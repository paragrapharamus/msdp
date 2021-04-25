from typing import Optional, Type, List
import numpy as np
from copy import deepcopy
from pytorch_lightning import LightningModule
import torch
from torch.utils.data import DataLoader

from dp_stages import Stage4
from fl.client import Client
from log import Logger


class Aggregator:
  def __init__(self,
               model_class: Type[LightningModule],
               clients: List[Client],
               clients_per_round: int,
               total_training_data_size: int,
               val_dataloader: DataLoader,
               test_dataloader: DataLoader,
               rounds: int,
               device: torch.device,
               logger: Optional[Logger] = None,
               epsilon: Optional[float] = None,
               max_weight_norm: Optional[float] = None):

    self.model_class = model_class
    self.model = None
    self.clients = clients
    self.clients_per_round = clients_per_round
    self.current_round_clients = []
    self.total_training_data_size = total_training_data_size
    self.curr_round_training_dataset_size = 0
    self.test_dataloader = test_dataloader
    self.val_dataloader = val_dataloader
    self.rounds = rounds
    self.device = device
    self.logger = logger

    self.dp_stage = None
    if epsilon:
      self.dp_stage = Stage4(epsilon, max_weight_norm)
      self.dp_stage.add_logger(logger, 'Aggregator')
      self.max_exposures = 0
      self.min_client_dataset = total_training_data_size

  def train_and_test(self):
    self.train()
    self.log("Testing... ")
    self.test(self.test_dataloader)

  def train(self):
    self.model = self.model_class()

    for r in range(self.rounds):
      self.log(f'******** ROUND {r + 1} ********')

      if self.dp_stage:
        self.max_exposures = 0
        self.min_client_dataset = self.total_training_data_size

      self._select_clients(r)
      self._send_model_and_train()
      self._aggregate_models()
      self.log("Validation... ")
      self.test(self.val_dataloader)

      if self.dp_stage:
        self.dp_stage.apply(self.model, self.rounds,
                            len(self.clients), self.clients_per_round,
                            self.min_client_dataset, self.max_exposures,
                            self.curr_round_training_dataset_size)

  def test(self, loader):
    losses = []
    self.model.to(self.device)
    with torch.no_grad():
      correct = 0
      for data, targets in loader:
        data, targets = data.to(self.device), targets.to(self.device)
        output = self.model(data)
        losses.append(self.model.compute_loss(output, targets).item())
        correct += (torch.argmax(output, 1) == targets).float().sum()
      accuracy = correct / len(loader.dataset)
      self.log(f"Global model accuracy: {accuracy:.3f}")
    self.model.cpu()

  def _send_model_and_train(self):
    for client in self.current_round_clients:
      params = deepcopy(list(self.model.parameters()))
      client.update_model(params)
      client.train()
      client.test()

  def _aggregate_models(self):
    global_model_contribution_weight = 0
    for client in self.current_round_clients:
      client_params = client.get_model_params()
      client_contribution_weight = client.training_data_size / self.curr_round_training_dataset_size

      if self.dp_stage:
        self.max_exposures = max(self.max_exposures, client.exposures)
        self.min_client_dataset = min(self.min_client_dataset, client.training_data_size)

      global_model_params = self.model.named_parameters()
      dict_global_model_params = dict(global_model_params)

      # Average the models into a global model
      with torch.no_grad():
        for name, param in client_params:
          if name in dict_global_model_params:
            averaged_weights = client_contribution_weight * param.data \
                               + global_model_contribution_weight * dict_global_model_params[name].data
            dict_global_model_params[name].data.copy_(averaged_weights)

      global_model_contribution_weight = 1

  def _select_clients(self, current_round):
    np.random.seed(current_round)
    client_indexes = np.random.choice(range(len(self.clients)), self.clients_per_round, replace=False)
    self.current_round_clients = []
    current_training_data = 0
    for ci in client_indexes:
      self.current_round_clients.append(self.clients[ci])
      current_training_data += self.clients[ci].training_data_size
    self.curr_round_training_dataset_size = current_training_data
    return self.current_round_clients

  def log(self, *msg):
    if self.logger:
      self.logger.log(*msg, module='Aggregator')
