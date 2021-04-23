from typing import Union, Type, Optional, List
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from torch.utils.data import Dataset

from msdp import MSPDTrainer, Stages


class Client:
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
               optimizer_momentum: Optional[float] = 0.9,
               eps1: Optional[float] = None,
               noise_multiplier: Optional[float] = None,
               max_grad_norm: Optional[float] = None,
               eps3: Optional[float] = None,
               max_weight_norm: Optional[float] = None):
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

    # Create the MSDP trainer
    self.model_trainer = MSPDTrainer(model=self.model_class(),
                                     data_loaders=dataloaders,
                                     epochs=epochs,
                                     batch_size=batch_size,
                                     device=device,
                                     id=id)

    # Attach the DP Stages if their parameters are provided
    if eps1:
      self.model_trainer.attach_stage(Stages.STAGE_1, {'eps': eps1})
    if noise_multiplier or max_grad_norm:
      self.model_trainer.attach_stage(Stages.STAGE_2, {'noise_multiplier': noise_multiplier,
                                                       'max_grad_norm': max_grad_norm})
    if eps3:
      self.model_trainer.attach_stage(Stages.STAGE_3, {'eps': eps3,
                                                       'max_weight_norm': max_weight_norm})

  # noinspection PyArgumentList
  def update_model(self, parameters):
    model_params = list(self.model_trainer.model.parameters())
    for i, param in enumerate(parameters):
      model_params[i].data.copy_(param.data)
    if self.optimizer_momentum:
      optimizer = self.optimizer_class(model_params, lr=self.learning_rate,
                                       weight_decay=self.weight_decay,
                                       momentum=self.optimizer_momentum)
    else:
      optimizer = self.optimizer_class(model_params,  lr=self.learning_rate,
                                       weight_decay=self.weight_decay)
    self.model_trainer.update_optimizer(optimizer)

  def get_model_params(self):
    return self.model_trainer.model.named_parameters()

  def train(self):
    self.model_trainer.train()

  def test(self):
    self.model_trainer.test()
