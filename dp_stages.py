import types
from enum import Enum
from typing import Union, List, Optional

import numpy as np
import torch
from opacus import PerSampleGradientClipper, PrivacyEngine
from opacus.utils import clipping
from torch import nn
from torch.utils.data import DataLoader

from log import Logger


class Stages(Enum):
  """ Enumeration of the supported stages of the MSDPTrainer """

  STAGE_1 = 1
  """
    Input perturbation.
    This stage adds calibrated noise directly to the training data.
    Uses the following parameters:
      * eps: float
          Quantifies the degree of privacy added during this stage, 
          hence the amount of noise which will be injected into data.
      * max_grad_norm: Optional[Union[float, List[float]]]
          The maximum L2 norm to which the gradients will be clipped.
          Could be a single real value, to which all gradients will be
          clipped or a list of maximum norms per layer. 
          If provided, the input perturbation mechanism will be adapted 
  """

  STAGE_2 = 2
  """
    Gradient Perturbation.
    This stage wraps the optimizer and injects noise into the gradients
    Uses the following parameters:
      * noise_multiplier: float
          The ratio of the standard deviation of the Gaussian noise to
          the L2-sensitivity of the function to which the noise is added
      * max_grad_norm: Union[float, List[float]]
          The maximum L2 norm to which the gradients will be clipped.
          Could be a single real value, to which all gradients will be
          clipped or a list of maximum norms per layer
  """

  STAGE_3 = 3
  """
    Output (weights) Perturbation.
    Injects calibrated noise into the weights of the trained model 
    Uses the following parameters:
      * `eps: float`
          Quantifies the degree of privacy added during this stage, 
          hence the amount of noise which will be injected into the weights.
      * max_weight_norm: 
          The maximum L2 norm to which the weights will be clipped.
          Could be a single real value, to which all weights will be
          clipped or a list of maximum norms per layer
  """

  STAGE_4 = 4
  """
    Aggregated model output perturbation.
    Used in Federated Learning. Injects noise into the weights of the 
    aggregated model. 
    Uses the following parameters:
      * `eps: float`
          Quantifies the degree of privacy added during this stage, 
          hence the amount of noise which will be injected into the weights.
      * max_weight_norm: 
          The maximum L2 norm to which the weights will be clipped.
          Could be a single real value, to which all weights will be
          clipped or a list of maximum norms per layer
  """


class DPStage:
  def __init__(self, stage):
    self.stage = stage
    self.logger = None
    self.module = None

  def apply(self, *args, **kwargs):
    pass

  def add_logger(self, logger: Logger, parent_module: str):
    self.logger = logger
    self.module = f'{parent_module}->{self.stage}'

  def log(self, *msg):
    if self.logger is not None:
      self.logger.log(*msg, module=self.module)


class Stage1(DPStage):
  """
    Input perturbation.
    This stage adds calibrated noise directly to the training data.

    :param eps: float
          Quantifies the degree of privacy added during this stage,
          hence the amount of noise which will be injected into data.

    :param max_grad_norm: Optional[Union[float, List[float]]]
          The maximum L2 norm to which the gradients will be clipped.
          Could be a single real value, to which all gradients will be
          clipped or a list of maximum norms per layer.
          If provided, the input perturbation mechanism will be adapted
  """

  def __init__(self,
               eps: float,
               max_grad_norm: Optional[Union[float, List[float]]] = None):

    super(Stage1, self).__init__('STAGE_1')
    self.eps = eps
    self.max_grad_norm = max_grad_norm
    if max_grad_norm and isinstance(max_grad_norm, list):
      self.max_grad_norm = np.array(max_grad_norm).mean()

  def apply(self, data_loader: DataLoader, epochs: Optional[int] = 1) -> DataLoader:
    dataset = torch.tensor(np.moveaxis(data_loader.dataset.data, -1, 1)).float()

    noised_data = self._perturb_dataset(dataset, epochs) if self.max_grad_norm \
      else self._perturb_dataset_with_sensitivity(dataset)

    dataset = np.moveaxis(noised_data.numpy(), 1, -1)
    dataset = (dataset - dataset.min()) / (dataset.max() - dataset.min()) * 255
    data_loader.dataset.data = dataset.astype(np.uint8)
    data_loader.stage_1_attached = True
    return data_loader

  def _perturb_dataset(self, dataset: torch.Tensor, epochs: int) -> torch.Tensor:
    """
      Based on -> https://arxiv.org/abs/2002.08570
    """
    n = len(dataset)
    delta = 1 / (2 * len(dataset))
    std = self.max_grad_norm * np.sqrt(epochs * np.log(1/delta) / n * (n - 1)) / self.eps
    self.log(f"Input perturbation: std={std:.2e}, ({self.eps}, {delta})-DP")
    noise = torch.normal(0, std, size=dataset.shape)
    dataset += noise
    return dataset

  def _perturb_dataset_with_sensitivity(self, dataset: torch.Tensor) -> torch.Tensor:
    """
    Based on -> https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
             -> http://www.cs.cmu.edu/~anupamd/paper/NDSS2016.pdf
    """
    delta = 1 / (1.5 * len(dataset))
    sensitivity = self._get_sensitivity(dataset)
    eps = 25 * self.eps
    std = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / eps
    self.log(f"Input perturbation: ({eps:.2f}, {delta:.2e})-DP with std={std:.2e}")
    noise = torch.normal(0, std, size=dataset.shape)
    dataset += noise
    return dataset

  def _get_sensitivity(self, dataset: torch.Tensor) -> torch.Tensor:
    """
    :param dataset: [n, c, *] or [n, *]
    """
    if len(dataset.shape) == 2:
      norms = torch.norm(dataset, dim=1)
    elif len(dataset.shape) == 4:
      norms = torch.linalg.norm(dataset, dim=(1, 2, 3))
    else:
      raise ValueError("Unknown data shape to compute the sensitivity")
    sensitivity = torch.sqrt(torch.abs(norms.max(dim=0)[0] - norms.min(dim=0)[0]))
    self.log(f"Input perturbation: sensitivity: {sensitivity}")
    return sensitivity


class Stage2(DPStage):
  """
    Gradient Perturbation.
    This stage wraps the optimizer and injects noise into the gradients

    :param noise_multiplier:  float
          The ratio of the standard deviation of the Gaussian noise to
          the L2-sensitivity of the function to which the noise is added
    :param max_grad_norm:  Union[float, List[float]]
          The maximum L2 norm to which the gradients will be clipped.
          Could be a single real value, to which all gradients will be
          clipped or a list of maximum norms per layer
  """

  def __init__(self,
               noise_multiplier: float,
               max_grad_norm: Union[float, List[float]],
               ):
    super(Stage2, self).__init__('STAGE_2')
    self.noise_multiplier = noise_multiplier
    self.max_grad_norm = max_grad_norm
    self.privacy_engine = None

  def apply(self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            device: torch.device,
            batch_size: int,
            train_data_size: int,
            n_accumulated_steps: Optional[int] = 1
            ) -> torch.optim.Optimizer:
    """
      See -> https://github.com/pytorch/opacus
    """
    clipping = {"clip_per_layer": False, "enable_stat": True}
    self.privacy_engine = PrivacyEngine(
      model,
      sample_rate=(n_accumulated_steps * batch_size) / train_data_size,
      noise_multiplier=self.noise_multiplier,
      max_grad_norm=self.max_grad_norm,
      secure_rng=False,
      **clipping,
    )
    self.privacy_engine.attach(optimizer)

    return optimizer

  def detach(self):
    self.privacy_engine.detach()

  def get_privacy_spent(self, train_dataset_size: int):
    delta = 1 / (1.5 * train_dataset_size)
    return self.privacy_engine.get_privacy_spent(delta)[0], delta

class Stage3(DPStage):
  """
    Output (weights) Perturbation.
    Injects calibrated noise into the weights of the trained model

    :param eps:  float`
          Quantifies the degree of privacy added during this stage,
          hence the amount of noise which will be injected into the weights.
    :param max_weight_norm: Union[float, List[float]]
          The maximum L2 norm to which the weights will be clipped.
          Could be a single real value, to which all weights will be
          clipped or a list of maximum norms per layer
  """

  def __init__(self,
               eps: float,
               max_weight_norm: Union[float, List[float]]):
    super(Stage3, self).__init__('STAGE_3')
    self.eps = eps if eps else 1
    self.max_weight_norm = max_weight_norm
    self.exposures = 1

  def apply(self,
            model: nn.Module,
            training_dataset_size: float,
            ) -> nn.Module:
    """
    Based on -> https://arxiv.org/abs/1911.00222 for the uplink channel
    """
    # Compute probability of DP failure
    delta = 1 / (1.5 * training_dataset_size)
    # The standard deviation of the Gaussian noise. This is not calibrated yet.
    std = (self.exposures * np.sqrt(2 * np.log(1.25 / delta))) / self.eps
    self.log(f"Output Perturbation: ({self.eps:.2f}, {delta:.2e})-DP")  # with std={std:.2e}")

    # Sanitize the model's parameters to ensure privacy
    for i, p in enumerate(model.parameters()):
      if isinstance(self.max_weight_norm, list):
        clip_val = self.max_weight_norm[i]
      else:
        clip_val = self.max_weight_norm

      # Clip the weights to a maximum l2 norm
      p.data /= max(1, torch.norm(p.data) / clip_val)
      # Calibrate the noise
      max_sensitivity = 2 * clip_val / training_dataset_size
      sensitivity_calibrated_std = max_sensitivity * std
      # Inject noise to the weights
      noise = torch.normal(mean=0, std=sensitivity_calibrated_std, size=p.shape).to(p.data.device)
      p.data.add_(noise)
    return model

  def set_exposures(self, exposures: int):
    self.exposures = exposures


class Stage4(DPStage):
  """
    Aggregated model output perturbation

    :param eps:  float`
          Quantifies the degree of privacy added during this stage,
          hence the amount of noise which will be injected into the weights.
    :param max_weight_norm: Union[float, List[float]]
          The maximum L2 norm to which the weights will be clipped.
          Could be a single real value, to which all weights will be
          clipped or a list of maximum norms per layer
  """

  def __init__(self,
               eps: float,
               max_weight_norm: Union[float, List[float]]):
    super(Stage4, self).__init__('STAGE_4')
    self.eps = eps if eps else 1
    self.max_weight_norm = max_weight_norm

  def _get_std(self,
               rounds: int,
               clients: int,
               selected_clients: int,
               min_client_dataset_size: int,
               max_exposure: int,
               delta: float):
    """
    Based on -> https://arxiv.org/abs/1911.00222 for the donwlink channel
    """
    cs = clients / selected_clients
    b = -(rounds / self.eps) * np.log(1 - cs + cs * np.exp(-self.eps / rounds))
    g = - np.log(1 - 1 / cs + np.exp(-self.eps / (max_exposure * np.sqrt(selected_clients))) / cs)
    if rounds <= self.eps / g:
      return 0
    else:
      c = np.sqrt(2 * np.log(1.25 / delta))
      nom = 2 * c * self.max_weight_norm * np.sqrt((rounds ** 2) / (b ** 2) - selected_clients * max_exposure ** 2)
      return nom / (min_client_dataset_size * selected_clients * self.eps)

  def apply(self,
            model: nn.Module,
            rounds: int,
            clients: int,
            selected_clients: int,
            min_client_dataset_size: int,
            max_exposure: int,
            training_dataset_size: int
            ) -> nn.Module:
    self.log("Aggregated weights perturbation...")
    delta = 1 / (1.5 * training_dataset_size)
    std = self._get_std(rounds, clients, selected_clients, min_client_dataset_size, max_exposure, delta)

    # Sanitize the model's parameters to ensure privacy
    for i, p in enumerate(model.parameters()):
      if isinstance(self.max_weight_norm, list):
        clip_val = self.max_weight_norm[i]
      else:
        clip_val = self.max_weight_norm

      # Clip the weights to a maximum l2 norm
      p.data /= max(1, torch.norm(p.data) / clip_val)
      # Inject noise to the weights
      noise = torch.normal(mean=0, std=std, size=p.shape).to(p.data.device)
      p.data.add_(noise)
    return model
