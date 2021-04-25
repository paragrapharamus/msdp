import types
from enum import Enum
from typing import Union, List, Optional

import numpy as np
import torch
from opacus import PerSampleGradientClipper
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
  """

  def __init__(self, eps: float):
    super(Stage1, self).__init__('STAGE_1')
    self.eps = eps

  def apply(self, data_loader: DataLoader) -> DataLoader:
    dataset = torch.tensor(np.moveaxis(data_loader.dataset.data, -1, 1)).float()
    sensitivity = self._get_sensitivity(dataset)
    self.log(f"Input perturbation: sensitivity: {sensitivity}")
    noised_data = self._perturb_dataset(dataset, sensitivity)

    dataset = np.moveaxis(noised_data.numpy(), 1, -1)
    dataset = (dataset - dataset.min()) / (dataset.max() - dataset.min()) * 255
    data_loader.dataset.data = dataset.astype(np.uint8)
    data_loader.stage_1_attached = True
    return data_loader

  def _perturb_dataset(self, dataset: torch.Tensor, sensitivity: torch.Tensor) -> torch.Tensor:
    delta = 1 / (1.5 * len(dataset))
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
    return sensitivity


class Stage2(DPStage):
  """
    Gradient Perturbation.
    This stage wraps the optimizer and injects noise into the gradients
  """

  def __init__(self,
               noise_multiplier: float,
               max_grad_norm: Union[float, List[float]]):
    super(Stage2, self).__init__('STAGE_2')
    self.noise_multiplier = noise_multiplier
    self.max_grad_norm = max_grad_norm
    self.steps = 0

  def apply(self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            device: torch.device) -> torch.optim.Optimizer:
    self.model = model
    self.device = device
    # Opacus
    norm_clipper = (
      clipping.ConstantFlatClipper(self.max_grad_norm)
      if not isinstance(self.max_grad_norm, list)
      else clipping.ConstantPerLayerClipper(self.max_grad_norm)
    )

    self.clipper = PerSampleGradientClipper(self.model, norm_clipper)

    optimizer = self.wrap_optimizer(optimizer)
    optimizer.stage_2_attached = True
    return optimizer

  def wrap_optimizer(self, optimizer):
    def dp_zero_grad(self):
      self.dpstage._zero_grad()
      self.original_zero_grad()

    def dp_step(self, closure=None):
      self.dpstage._step()
      self.original_step(closure)

    optimizer.dpstage = self
    optimizer.original_step = optimizer.step
    optimizer.step = types.MethodType(dp_step, optimizer)

    optimizer.original_zero_grad = optimizer.zero_grad
    optimizer.zero_grad = types.MethodType(dp_zero_grad, optimizer)
    return optimizer

  def _zero_grad(self):
    self.clipper.zero_grad()

  def _step(self):
    self.steps += 1
    self.clipper.clip_and_accumulate()
    clip_values, batch_size = self.clipper.pre_step()

    params = (p for p in self.model.parameters() if p.requires_grad)
    for p, clip_value in zip(params, clip_values):
      noise = self._generate_noise(clip_value, p)
      noise /= batch_size
      p.grad += noise

  def _generate_noise(self,
                      max_grad_norm: float,
                      reference: nn.parameter.Parameter) -> torch.Tensor:
    if self.noise_multiplier > 0 and max_grad_norm > 0:
      return torch.normal(
        0,
        self.noise_multiplier * max_grad_norm,
        reference.grad.shape,
        device=self.device,
      )
    return torch.zeros(reference.grad.shape, device=self.device)


class Stage3(DPStage):
  """
    Output (weights) Perturbation.
    Injects calibrated noise into the weights of the trained model
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

    # Compute probability of DP failure
    delta = 1 / (1.5 * training_dataset_size)
    # The standard deviation of the Gaussian noise. This is not calibrated yet.
    std = (self.exposures * np.sqrt(2 * np.log(1.25 / delta))) / self.eps
    self.log(f"Output Perturbation: ({self.eps:.2f}, {delta:.2e})-DP with std={std:.2e}")

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
