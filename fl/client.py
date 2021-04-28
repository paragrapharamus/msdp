from typing import Union, Type, Optional, List
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from torch.utils.data import Dataset

from log import Logger
from msdp import MSPDTrainer
from dp_stages import Stages


class Client:
  """
  FL client

  :param id:
  :param model_class:
  :param dataloaders:
  :param epochs:
  :param batch_size:
  :param optimizer_class:
  :param learning_rate:
  :param weight_decay:
  :param device:
  :param optimizer_momentum:
  :param eps1:
  :param noise_multiplier:
  :param max_grad_norm:
  :param eps3:
  :param max_weight_norm:
  :param logger:
  :param experiment_id:
  """
  def __init__(self,
               id: int,
               model_class: Type[LightningModule],
               dataloaders: Union[DataLoader, List[DataLoader]],
               epochs: int,
               batch_size: int,
               optimizer_class: Type[torch.optim.Optimizer],
               learning_rate: float,
               weight_decay: float,
               device: torch.device,
               virtual_batches: Optional[int] = 1,
               optimizer_momentum: Optional[float] = 0.9,
               eps1: Optional[float] = None,
               noise_multiplier: Optional[float] = None,
               max_grad_norm: Optional[float] = None,
               eps3: Optional[float] = None,
               max_weight_norm: Optional[float] = None,
               logger: Optional[Logger] = None,
               experiment_id: Optional[int] = 0):

    self.id = id
    self.model_class = model_class
    self.device = device

    self.dataloaders = dataloaders
    if isinstance(dataloaders, DataLoader):
      self.training_data_size = len(dataloaders.dataset)
    else:
      self.training_data_size = len(dataloaders[0].dataset)

    self.batch_size = batch_size
    self.epochs = epochs

    # Lazy instantiation during training
    self.optimizer_class = optimizer_class
    self.learning_rate = learning_rate
    self.weight_decay = weight_decay
    self.optimizer_momentum = optimizer_momentum

    self.logger = logger
    self.log(f"Client {self.id} initiated")

    # Create the MSDP trainer
    self.model_trainer = MSPDTrainer(model=self.model_class(),
                                     data_loaders=dataloaders,
                                     epochs=epochs,
                                     batch_size=batch_size,
                                     device=device,
                                     virtual_batches=virtual_batches,
                                     id=id,
                                     logger=logger,
                                     experiment_id=experiment_id)

    # Attach the DP Stages if their parameters are provided
    if eps1:
      self.model_trainer.attach_stage(Stages.STAGE_1, {'eps': eps1,
                                                       'max_grad_norm': max_grad_norm})
    if noise_multiplier or max_grad_norm:
      self.model_trainer.attach_stage(Stages.STAGE_2, {'noise_multiplier': noise_multiplier,
                                                       'max_grad_norm': max_grad_norm})
    if eps3:
      self.model_trainer.attach_stage(Stages.STAGE_3, {'eps': eps3,
                                                       'max_weight_norm': max_weight_norm})

    # The number of times the clients took part in the aggregation
    self.exposures = 0

  # noinspection PyArgumentList
  def update_model(self, parameters):
    self.exposures += 1
    model_params = list(self.model_trainer.model.parameters())
    for i, param in enumerate(parameters):
      model_params[i].data.copy_(param.data)
    if self.optimizer_momentum:
      optimizer = self.optimizer_class(model_params, lr=self.learning_rate,
                                       weight_decay=self.weight_decay,
                                       momentum=self.optimizer_momentum)
    else:
      optimizer = self.optimizer_class(model_params, lr=self.learning_rate,
                                       weight_decay=self.weight_decay)
    self.model_trainer.update_optimizer(optimizer)

  def get_model_params(self):
    return self.model_trainer.model.named_parameters()

  def train(self):
    if Stages.STAGE_3 in self.model_trainer.stages:
      self.model_trainer.stages[Stages.STAGE_3].set_exposures(self.exposures)
    self.model_trainer.train()

  def test(self):
    self.model_trainer.test()

  def log(self, *msg):
    self.logger.log(*msg, module=f'Client_{self.id}')
