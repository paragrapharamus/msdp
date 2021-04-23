import os
import sys
import types
from enum import Enum
from typing import List, Union, Optional, Type

import numpy as np
import pytorch_lightning as pl
import torch
from opacus import PerSampleGradientClipper
from opacus.utils import clipping
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader

from log import Logger

try:
  from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
  from libs.fedml.fedml_core.trainer.model_trainer import ModelTrainer


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
    std = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / (25 * self.eps)
    self.log(f"Input perturbation: noise std: {std}")
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
  """
    Output (weights) Perturbation.
    Injects calibrated noise into the weights of the trained model
  """

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
    self.log("Output Perturbation...")
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

  def __init__(self,
               model: pl.LightningModule,
               data_loaders: Union[DataLoader, List[DataLoader]],
               epochs: int,
               batch_size: int,
               device: torch.device,
               optimizer: Optional[torch.optim.Optimizer] = None,
               logger: Optional[Logger] = None,
               save_checkpoint: Optional[bool] = False,
               gpus: Optional[Union[int, List[int]]] = 1,
               id: Optional[int] = 0):

    self.id = f'MSDPTrainer_{id}'
    self.model = model

    trainer_callbacks = None
    if save_checkpoint:
      checkpoint_callback = ModelCheckpoint(
        monitor='valid_acc_epoch',
        dirpath=self._get_checkpoint_dir_path(),
        filename='checkpoint-{epoch:02d}-{valid_acc:.2f}',
        save_top_k=-1,
        mode='max',
      )
      trainer_callbacks = [checkpoint_callback]

    self.trainer = pl.Trainer(min_epochs=epochs,
                              max_epochs=epochs,
                              gpus=gpus,
                              callbacks=trainer_callbacks)

    self.optimizer = optimizer

    if isinstance(data_loaders, DataLoader):
      self.train_loader = data_loaders
    elif len(data_loaders) == 1:
      self.train_loader = data_loaders[0]
    elif len(data_loaders) == 2:
      self.train_loader, self.test_loader = data_loaders
    elif len(data_loaders) == 3:
      self.train_loader, self.val_loader, self.test_loader = data_loaders
    else:
      raise Exception("Invalid number of loaders")

    self.epochs = epochs
    self.batch_size = batch_size
    self.device = device
    self.steps = 0

    self.stages = dict()

    self.logger = logger
    if logger is None:
      self.logger = Logger([sys.stdout, './msdp.log'])

  def attach_stage(self, stage_type: Stages, stage_param_dict: dict):
    if stage_type == Stages.STAGE_1:
      self.stages[stage_type] = Stage1(stage_param_dict['eps'])
    elif stage_type == Stages.STAGE_2:
      self.stages[stage_type] = Stage2(stage_param_dict['noise_multiplier'],
                                       stage_param_dict['max_grad_norm'])
    elif stage_type == Stages.STAGE_3:
      self.stages[stage_type] = Stage3(stage_param_dict['eps'],
                                       stage_param_dict['max_weight_norm'])
    else:
      raise ValueError("Unknown stage type.")
    self.log(f"{stage_type} successfully attached.")

  @staticmethod
  def accuracy(preds: np.ndarray, labels: np.ndarray):
    return (preds == labels).mean()

  def log(self, *msg):
    self.logger.log(*msg, module=self.id)

  def log_warning(self, *msg):
    self.logger.log_waring(*msg, module=self.id)

  def train(self):

    self.model.to(self.device)

    if not self.optimizer:
      self.log_warning("Using a default optimizer. Please provide an optimizer for personalized training")
      self.optimizer = self.model.default_optimizer()

    for _, stage in self.stages.items():
      stage.add_logger(self.logger)

    loaders = [self.train_loader]
    if hasattr(self, 'val_loader'):
      loaders.append(self.val_loader)

    # Inject noise to data
    if Stages.STAGE_1 in self.stages and not hasattr(self.train_loader, 'stage_1_attached'):
      self._stage_1_noise(loaders)

    # Wrap the optimizer to perform DP training
    self._wrap_stage_2()

    self.trainer.fit(self.model, *loaders)

    # Inject noise to model's weights
    if Stages.STAGE_3 in self.stages:
      self._stage_3_noise()

    return self.model

  def test(self):
    if hasattr(self, 'test_loader'):
      loader = self.test_loader
      if not (hasattr(loader, 'stage_1_attached') and loader.stage_1_attached) and Stages.STAGE_1 in self.stages:
        self._stage_1_noise([loader])

      self.trainer.test(self.model, loader)

  def train_and_test(self):
    model = self.train()
    self.test()
    return model

  def update_optimizer(self, optimizer):
    self.optimizer = optimizer
    self._wrap_stage_2()

  @staticmethod
  def _get_checkpoint_dir_path():
    checkpoint_dir_base = os.path.join(os.getcwd(), 'checkpoints')
    checkpoint_dir = checkpoint_dir_base
    dir_id = 1
    while os.path.exists(checkpoint_dir):
      checkpoint_dir = f"{checkpoint_dir_base}_{dir_id}"
      dir_id += 1
    os.mkdir(checkpoint_dir)
    return checkpoint_dir

  def _stage_1_noise(self, loaders):
    for i, loader in enumerate(loaders):
      self.stages[Stages.STAGE_1].apply(loader)

  def _wrap_stage_2(self):
    if Stages.STAGE_2 in self.stages:
      self.model.add_optimizer(self.stages[Stages.STAGE_2].apply(self.model, self.optimizer, self.device))

  def _stage_3_noise(self):
    train_dataset_size = len(self.train_loader.dataset)
    self.stages[Stages.STAGE_3].apply(self.model, train_dataset_size, self.epochs, self.batch_size)
