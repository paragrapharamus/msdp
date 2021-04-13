import os
import shutil
import sys
import types

from typing import List, Union, Optional

import numpy as np
import torch
import torch.nn.functional as F
from opacus import PerSampleGradientClipper
from opacus.utils import clipping
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from log import Logger
from enum import Enum


class Stages(Enum):
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
    This stages wraps the optimizer and inject noise into the gradients
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


class MSDPStagesConfig:
  def __init__(self):
    self.stage_dict = {Stages.STAGE_1: None, Stages.STAGE_2: None, Stages.STAGE_3: None}

  # noinspection PyTypeChecker
  def add_stage(self, stage_type: Stages, param_dict):
    if stage_type == Stages.STAGE_1:
      self.stage_dict[stage_type] = Stage1(param_dict['eps'])
    elif stage_type == Stages.STAGE_2:
      self.stage_dict[stage_type] = Stage2(param_dict['noise_multiplier'],
                                           param_dict['max_grad_norm'])
    elif stage_type == Stages.STAGE_3:
      self.stage_dict[stage_type] = Stage3(param_dict['eps'],
                                           param_dict['max_weight_norm'])
    else:
      raise ValueError("Unknown stage type.")

  def add_stages(self, stages_dict: dict):
    """

    :param stages_dict:
    """
    for stage_type, param_dic in stages_dict.items():
      self.add_stage(stage_type, param_dic)

  def get_stages(self):
    stages = {}
    for stage_type, stage in self.stage_dict.items():
      if stage is not None:
        stages[stage_type] = stage
    return stages


class DPStage:
  def __init__(self, stage):
    self.stage = stage
    self.logger = None

  def apply(self, *args, **kwargs):
    pass

  def add_logger(self, logger: Logger):
    self.logger = logger

  def log(self, *msg):
    if self.logger is not None:
      self.logger.log(*msg, module=self.stage)


class Stage1(DPStage):
  def __init__(self, eps: float):
    super(Stage1, self).__init__('STAGE_1')
    self.eps = eps

  def apply(self, data_loader: DataLoader) -> DataLoader:
    # FIXME: fix for non-image data
    dataset = torch.tensor(np.moveaxis(data_loader.dataset.data, -1, 1)).float()
    sensitivity = self._get_sensitivity(dataset)
    self.log(f"Input perturbation: sensitivity: {sensitivity}")
    noised_data = self._perturb_dataset(dataset, sensitivity)

    dataset = np.moveaxis(noised_data.numpy(), 1, -1)
    dataset = (dataset - dataset.min()) / (dataset.max() - dataset.min()) * 255
    data_loader.dataset.data = dataset.astype(np.uint8)
    data_loader.stage_1_attached = True
    return data_loader

  def _perturb_dataset(self, dataset: torch.Tensor, sensitivity: torch.Tensor):
    delta = 1 / (1.5 * len(dataset))
    stds = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / (25 * self.eps)
    self.log(f"Input perturbation: noise std: {stds}")
    shape = [dataset.size(i) for i in range(len(dataset.shape))]
    shape[1] = 1
    noise = [torch.normal(0, std, size=shape) for std in stds]
    noise = torch.cat(noise, dim=1)
    dataset += noise
    return dataset

  def _get_sensitivity(self, dataset: torch.Tensor) -> torch.Tensor:
    """
    :param dataset: [n, c, *] or [n, *]
    """
    if len(dataset.shape) == 2:
      norms = torch.norm(dataset, dim=1)
      dim = 0
    else:
      norms = torch.norm(dataset, dim=(2, 3)).transpose(0, 1)
      dim = 1
    sensitivity = torch.sqrt(torch.abs(norms.max(dim=dim)[0] - norms.min(dim=dim)[0]))
    return sensitivity


class Stage2(DPStage):
  def __init__(self,
               noise_multiplier: float,
               max_grad_norm: Union[float, List[float]]):
    super(Stage2, self).__init__('STAGE_2')
    self.noise_multiplier = noise_multiplier
    self.max_grad_norm = max_grad_norm
    self.steps = 0

  def apply(self, model: nn.Module, optimizer: torch.optim, device: torch.device) -> torch.optim:
    self.model = model
    self.device = device
    # Opacus
    norm_clipper = (
      clipping.ConstantFlatClipper(self.max_grad_norm)
      if not isinstance(self.max_grad_norm, list)
      else clipping.ConstantPerLayerClipper(self.max_grad_norm)
    )

    self.clipper = PerSampleGradientClipper(self.model, norm_clipper)

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

  def _generate_noise(self, max_grad_norm: float, reference: nn.parameter.Parameter) -> torch.Tensor:
    if self.noise_multiplier > 0 and max_grad_norm > 0:
      return torch.normal(
        0,
        self.noise_multiplier * max_grad_norm,
        reference.grad.shape,
        device=self.device,
      )
    return torch.zeros(reference.grad.shape, device=self.device)


class Stage3(DPStage):
  def __init__(self,
               eps: float,
               max_weight_norm: Union[float, List[float]]):
    super(Stage3, self).__init__('STAGE_3')
    DEFAULT_MAX_NORM = 20
    self.eps = eps if eps else 1
    self.max_weight_norm = max_weight_norm if max_weight_norm else DEFAULT_MAX_NORM

  def apply(self, model: nn.Module,
            training_dataset_size: float,
            epochs: int,
            batch_size: int) -> nn.Module:
    # Compute probability of DP failure
    delta = 1 / (1.5 * training_dataset_size)
    # Number of times the training data is seen by the model
    exposure = int(epochs * training_dataset_size / batch_size)
    # The standard deviation of the Gaussian noise. This is not calibrated yet.
    std = (exposure * np.sqrt(2 * np.log(1.25 / delta))) / self.eps

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


class MSPDTrainer:
  """
  Multistage differentially private trainer
  """
  STATS_FMT = "[{:>5s}] loss: {:+.4f}, acc: {:.4f}"

  def __init__(self, model: nn.Module,
               optimizer: torch.optim,
               data_loaders: DataLoader,
               epochs: int,
               batch_size: int,
               device: torch.device,
               stages_config: Optional[MSDPStagesConfig] = None,
               logger: Optional[Logger] = None):

    self.model = model
    self.data_loaders = data_loaders
    if len(data_loaders) == 2:
      self.train_loader, self.test_loader = data_loaders
    elif len(data_loaders) == 3:
      self.train_loader, self.val_loader, self.test_loader = data_loaders
    else:
      raise Exception("Invalid number of loaders")

    self.epochs = epochs
    self.batch_size = batch_size
    self.device = device
    self.steps = 0
    self.logger = logger
    if logger is None:
      self.logger = Logger([sys.stdout, './msdp.log'])

    self.stages = dict() if stages_config is None else stages_config.get_stages()
    for _, stage in self.stages.items():
      stage.add_logger(self.logger)

    # Wrap the optimizer to perform DP training
    if Stages.STAGE_2 in self.stages:
      self.optimizer = self.stages[Stages.STAGE_2].apply(model, optimizer, device)
    else:
      self.optimizer = optimizer

  @staticmethod
  def accuracy(preds: np.ndarray, labels: np.ndarray):
    return (preds == labels).mean()

  def log(self, *msg):
    self.logger.log(*msg, module='MSPDTrainer')

  def train(self):
    # Inject noise to data
    if Stages.STAGE_1 in self.stages:
      self._stage_1_noise([self.train_loader])

    val_loader = self.val_loader if self.val_loader else self.test_loader

    best_acc = 0
    for epoch in range(self.epochs):
      prog_bar = tqdm(total=len(self.train_loader) * self.batch_size, file=sys.stdout)
      prog_bar.set_description("Epoch %d/%d" % (epoch + 1, self.epochs))
      losses = []
      accuracies = []

      # Model training
      self.model.train()
      for batch_idx, (data, targets) in enumerate(self.train_loader):
        data, targets = data.to(self.device), targets.to(self.device)

        output = self.model(data)
        loss = F.cross_entropy(output, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        losses.append(loss.item())
        accuracies.append(self.accuracy(torch.argmax(output, 1).detach().cpu().numpy(), targets.detach().cpu().numpy()))
        prog_bar.update(len(data))

      # Epoch statistics.
      epoch_loss = sum(losses) / len(losses)
      epoch_acc = sum(accuracies) / len(accuracies)
      prog_bar.close()
      self.log(f"Training epoch [{epoch + 1}/{self.epochs}] loss: {epoch_loss:.6f}, acc: {epoch_acc:.4f}")

      # Validation step
      acc = self.evaluate(val_loader)
      self.log(f"Validation Accuracy: {acc:.4f}\n")
      self.save_checkpoint(self.model.state_dict(), acc > best_acc, epoch + 1)
      best_acc = max(best_acc, acc)

    # Inject noise to model's weights
    if Stages.STAGE_3 in self.stages:
      self._stage_3_noise()

  def evaluate(self, loader):
    if hasattr(loader, 'stage_1_attached') and loader.stage_1_attached and Stages.STAGE_1 in self.stages:
      self._stage_1_noise([loader])
    self.model.eval()
    with torch.no_grad():
      accuracies = []
      for data, target in loader:
        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data)
        accuracies.append(self.accuracy(torch.argmax(output, 1).detach().cpu().numpy(), target.detach().cpu().numpy()))

      accuracy = sum(accuracies) / len(accuracies)
      return accuracy

  def train_and_test(self):
    self.train()
    acc = self.evaluate(self.test_loader)
    self.log(f"Final test Accuracy {acc:.4f}")

  @staticmethod
  def save_checkpoint(state, is_best, epoch, filename="checkpoint.pth"):
    checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')
    if not os.path.exists(checkpoint_dir):
      os.mkdir(os.path.join(os.getcwd(), 'checkpoints'))
    save_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, save_path)
    if is_best:
      shutil.copyfile(save_path, os.path.join(checkpoint_dir, f"model_best_epoch_{epoch}.pth"))

  def _stage_1_noise(self, loaders):
    for i, loader in enumerate(loaders):
      self.stages[Stages.STAGE_1].apply(loader)

  def _stage_3_noise(self):
    self.stages[Stages.STAGE_3].apply(self.model, len(self.train_loader.dataset), self.epochs, self.batch_size)
