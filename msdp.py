import os
import sys
from typing import List, Union, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from dp_stages import Stages, Stage1, Stage2, Stage3
from log import Logger
import time


class MSPDTrainer:
  """
  Multistage differentially private trainer

    :param model: The model to be trained
    :param data_loaders: the training dataloader of a list of dataloaders
    :param epochs: the number of epochs
    :param batch_size: the batch size
    :param device: the device on which data is processed
    :param optimizer: the model's optimizer
    :param logger: the logger
    :param save_checkpoint: if a checkpoint should be saved
    :param gpus: the number of gpus to train on or their indices
    :param id: the id of the trainer
    :param experiment_id: the id of the current experiment
    :param save_dir: where to save data if needed
  """
  STATS_FMT = "[{:>5s}] loss: {:+.4f}, acc: {:.4f}"

  def __init__(self,
               model: pl.LightningModule,
               data_loaders: Union[DataLoader, List[DataLoader]],
               epochs: int,
               batch_size: int,
               device: torch.device,
               virtual_batches: Optional[int] = 1,
               optimizer: Optional[torch.optim.Optimizer] = None,
               logger: Optional[Logger] = None,
               save_checkpoint: Optional[bool] = False,
               gpus: Optional[Union[int, List[int]]] = 1,
               id: Optional[int] = 0,
               experiment_id: Optional[int] = None,
               save_dir: Optional[str] = 'out/'):

    self.id = f'MSDPTrainer_{id}'
    self.start_time = time.time()
    self.model = model

    self.trainer_callbacks = None
    self.checkpoint_dir = None
    if save_checkpoint:
      self.checkpoint_dir = self._get_next_available_dir(save_dir, 'checkpoints',
                                                         absolute_path=True)
      checkpoint_callback = ModelCheckpoint(
        monitor='valid_acc_epoch',
        dirpath=self.checkpoint_dir,
        filename='checkpoint-{epoch:02d}-{valid_acc:.2f}',
        save_top_k=1,
        mode='max',
      )
    else:
      checkpoint_callback = ModelCheckpoint(
        monitor='valid_acc_epoch',
        filename='checkpoint-{epoch:02d}-{valid_acc:.2f}',
        save_top_k=0,
        mode='max',
      )
    self.trainer_callbacks = [checkpoint_callback]

    # Lazy init during training. Also, a new trainer instance must be created
    # before each FL round, in case the this model is used during a FL simulation
    self.trainer = None

    # creating the TensorBoard logging
    __logs_dir = f'{save_dir}/lightning_logs'
    if not experiment_id:
      experiment_id = self._get_next_available_dir(__logs_dir, 'experiment',
                                                   False, False, False)
    self.tensorboardlogger = TensorBoardLogger(save_dir=__logs_dir,
                                               version=self.id,
                                               name=experiment_id)

    if not self.checkpoint_dir:
      self.stat_dir = os.path.join(os.getcwd(),
                                   __logs_dir,
                                   experiment_id,
                                   self.id)

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
    self.virtual_batches = virtual_batches
    self.device = device
    self.gpus = gpus
    self.steps = 0

    self.stages = dict()

    self.logger = logger
    if logger is None:
      d = self.checkpoint_dir if self.checkpoint_dir else '.'
      self.logger = Logger([sys.stdout, f'{d}/msdp.log'])
    self.model.personal_log_fn = self.log

  def attach_stage(self, stage_type: Stages, stage_param_dict: dict):
    if stage_type == Stages.STAGE_1:
      self.stages[stage_type] = Stage1(stage_param_dict['eps'],
                                       stage_param_dict.get('max_grad_norm', None))
    elif stage_type == Stages.STAGE_2:
      self.stages[stage_type] = Stage2(stage_param_dict['noise_multiplier'],
                                       stage_param_dict['max_grad_norm'])
      # When training with Stage 2 attached, allow the model to accumulate
      # more batches before applying the gradient perturbation
      self.model.virtual_batches = self.virtual_batches
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
                              callbacks=self.trainer_callbacks,
                              progress_bar_refresh_rate=0,
                              num_sanity_val_steps=0,
                              deterministic=True)

    self.model.to(self.device)

    if not self.optimizer:
      self.log_warning("Using a default optimizer. Please provide an optimizer for personalized training")
      self.optimizer = self.model.default_optimizer()
    else:
      self.model.add_optimizer(self.optimizer)

    loaders = [self.train_loader]
    if hasattr(self, 'val_loader'):
      loaders.append(self.val_loader)

    # Inject noise to data
    if Stages.STAGE_1 in self.stages and not hasattr(self.train_loader, 'stage_1_attached'):
      applied = self.stages[Stages.STAGE_1].apply(self.train_loader, self.epochs)
      if not applied:
        self.model.batch_processing_hook = self._stage_1_on_batch

    # Wrap the optimizer to perform DP training
    if Stages.STAGE_2 in self.stages:
      self.model.add_optimizer(self.stages[Stages.STAGE_2].apply(self.model,
                                                                 self.optimizer,
                                                                 self.device,
                                                                 self.batch_size,
                                                                 len(self.train_loader.dataset),
                                                                 self.virtual_batches))
      self.model.privacy_info = self._log_privacy_info

    self.trainer.fit(self.model, *loaders)

    if Stages.STAGE_2 in self.stages:
      self._log_privacy_info()
      self.stages[Stages.STAGE_2].detach()

    # Inject noise to model's weights
    if Stages.STAGE_3 in self.stages:
      self._stage_3_noise()

    if self.checkpoint_dir:
      fp = os.path.join(self.checkpoint_dir, 'final.ckpt')
      self.trainer.save_checkpoint(fp)

    self._save_training_stats()

    self.model.cpu()
    return self.model

  def _log_privacy_info(self):
    eps, delta = self.stages[Stages.STAGE_2].get_privacy_spent(len(self.train_loader.dataset))
    self.log(f'Privacy from Stage 2: (ε={eps:.2f}, δ={delta:.2e})')

  def test(self):
    if hasattr(self, 'test_loader'):
      self.log("Started testing...")
      loader = self.test_loader

      self.model.to(self.device)
      results = self.trainer.test(self.model, loader)
      self.model.cpu()
      self.log(f"Test results: {results}")

      return results

  def train_and_test(self):
    model = self.train()
    self.test()
    return model

  def update_optimizer(self, optimizer):
    self.optimizer = optimizer

  def _save_training_stats(self):
    training_losses = np.array(self.model.training_losses)
    training_accuracies = np.array(self.model.training_accuracies)
    validation_accuracies = np.array(self.model.validation_accuracies)

    file_id = f'{self.id}_plot_stats.npy'
    if self.checkpoint_dir:
      fp = os.path.join(self.checkpoint_dir, file_id)
    else:
      fp = os.path.join(self.stat_dir, file_id)

    with open(fp, 'wb') as f:
      np.save(f, training_losses)
      np.save(f, training_accuracies)
      np.save(f, validation_accuracies)
    # self._save_time()

  def _save_time(self):
    times = self.model.training_times
    times.insert(0, self.start_time)
    times = np.array(times)
    times -= self.start_time
    file_id = f'{self.id}_times.npy'
    if self.checkpoint_dir:
      fp = os.path.join(self.checkpoint_dir, file_id)
    else:
      fp = os.path.join(self.stat_dir, file_id)

    with open(fp, 'wb') as f:
      np.save(f, times)

  def _load_training_stats(self):
    file_id = f'{self.id}_plot_stats'
    if self.checkpoint_dir:
      fp = os.path.join(self.checkpoint_dir, file_id)
    else:
      fp = os.path.join(os.getcwd(), file_id)

    with open(fp, 'rb') as f:
      training_losses = np.load(f)
      training_accuracies = np.load(f)
      validation_accuracies = np.load(f)
    return training_losses, training_accuracies, validation_accuracies

  @staticmethod
  def _get_next_available_dir(root,
                              dir_name,
                              absolute_path=True,
                              create=True,
                              id=False):
    checkpoint_dir_base = os.path.join(root, dir_name)
    dir_id = 1
    checkpoint_dir = f"{checkpoint_dir_base}_{dir_id}"
    while os.path.exists(checkpoint_dir):
      dir_id += 1
      checkpoint_dir = f"{checkpoint_dir_base}_{dir_id}"
    if create:
      os.mkdir(checkpoint_dir)
    if id:
      return (checkpoint_dir, dir_id) if absolute_path \
               else f"{dir_name}_{dir_id}", dir_id
    else:
      return checkpoint_dir if absolute_path \
        else f"{dir_name}_{dir_id}"

  def _stage_1_on_batch(self, data: torch.Tensor):
    # individual batch perturbation
    self.stages[Stages.STAGE_1].perturb_data(data, self.epochs, len(self.train_loader.dataset))

  def _stage_3_noise(self):
    self.stages[Stages.STAGE_3].apply(self.model, len(self.train_loader.dataset))
