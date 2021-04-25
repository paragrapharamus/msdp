import os
import sys
from typing import List, Union, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from dp_stages import Stages, Stage1, Stage2, Stage3
from log import Logger


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
               id: Optional[int] = 0,
               experiment_id: Optional[int] = None):

    self.id = f'MSDPTrainer_{id}'
    self.model = model

    self.trainer_callbacks = None
    if save_checkpoint:
      checkpoint_callback = ModelCheckpoint(
        monitor='valid_acc_epoch',
        dirpath=self._get_next_available_dir(os.getcwd(), 'checkpoints'),
        filename='checkpoint-{epoch:02d}-{valid_acc:.2f}',
        save_top_k=-1,
        mode='max',
      )
      self.trainer_callbacks = [checkpoint_callback]

    self.trainer = None

    # creating the TensorBoard logging
    __logs_dir = './lightning_logs'
    if not experiment_id:
      experiment_id = self._get_next_available_dir(__logs_dir, 'experiment', False, False)
    self.tensorboardlogger = TensorBoardLogger(save_dir=__logs_dir, version=self.id, name=experiment_id)

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
    self.gpus = gpus
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
    self.stages[stage_type].add_logger(self.logger, self.id)
    self.log(f"{stage_type} successfully attached.")

  def log(self, *msg):
    self.logger.log(*msg, module=self.id)

  def log_warning(self, *msg):
    self.logger.log_waring(*msg, module=self.id)

  def train(self):
    self.log(f"Started training...")
    self.trainer = pl.Trainer(min_epochs=self.epochs,
                              max_epochs=self.epochs,
                              gpus=self.gpus,
                              weights_summary=None,
                              logger=self.tensorboardlogger,
                              callbacks=self.trainer_callbacks)

    self.model.to(self.device)

    if not self.optimizer:
      self.log_warning("Using a default optimizer. Please provide an optimizer for personalized training")
      self.optimizer = self.model.default_optimizer()

    loaders = [self.train_loader]
    if hasattr(self, 'val_loader'):
      loaders.append(self.val_loader)

    # Inject noise to data
    if Stages.STAGE_1 in self.stages and not hasattr(self.train_loader, 'stage_1_attached'):
      self._stage_1_noise(loaders)

    # Wrap the optimizer to perform DP training
    if Stages.STAGE_2 in self.stages:
      # Attach stage 2 only once (add the model hooks), but update the optimizer
      # each time the model has been externally updated
      if hasattr(self.optimizer, 'stage_2_attached'):
        self.model.add_optimizer(self.stages[Stages.STAGE_2].wrap_optimizer(self.optimizer))
      else:
        self.model.add_optimizer(self.stages[Stages.STAGE_2].apply(self.model, self.optimizer, self.device))

    self.trainer.fit(self.model, *loaders)

    # Inject noise to model's weights
    if Stages.STAGE_3 in self.stages:
      self._stage_3_noise()

    self.model.cpu()
    return self.model

  def test(self):
    if hasattr(self, 'test_loader'):
      self.log("Started testing...")
      loader = self.test_loader
      if not (hasattr(loader, 'stage_1_attached') and loader.stage_1_attached) and Stages.STAGE_1 in self.stages:
        self._stage_1_noise([loader])

      self.model.to(self.device)
      results = self.trainer.test(self.model, loader)
      self.model.cpu()
      self.log(f"Test results: {results}")

  def train_and_test(self):
    model = self.train()
    self.test()
    return model

  def update_optimizer(self, optimizer):
    self.optimizer = optimizer

  @staticmethod
  def _get_next_available_dir(root, dir_name, absolute_path=True, create=True):
    checkpoint_dir_base = os.path.join(root, dir_name)
    dir_id = 1
    checkpoint_dir = f"{checkpoint_dir_base}_{dir_id}"
    while os.path.exists(checkpoint_dir):
      dir_id += 1
      checkpoint_dir = f"{checkpoint_dir_base}_{dir_id}"
    if create:
      os.mkdir(checkpoint_dir)
    if absolute_path:
      return checkpoint_dir
    else:
      return f"{dir_name}_{dir_id}"

  def _stage_1_noise(self, loaders):
    for i, loader in enumerate(loaders):
      self.stages[Stages.STAGE_1].apply(loader)

  def _stage_3_noise(self):
    self.stages[Stages.STAGE_3].apply(self.model, len(self.train_loader.dataset))
