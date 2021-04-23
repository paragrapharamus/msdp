from typing import Optional, Type, List
import numpy as np
from copy import deepcopy
from pytorch_lightning import LightningModule
import torch
from torch.utils.data import DataLoader

from fl.client import Client


class Aggregator:
  def __init__(self,
               model_class: Type[LightningModule],
               clients: List[Client],
               clients_per_round: int,
               total_training_data_size: int,
               test_dataloader: DataLoader,
               rounds: int,
               device: torch.device):

    self.model_class = model_class
    self.model = None
    self.clients = clients
    self.clients_per_round = clients_per_round
    self.current_round_clients = []
    self.total_training_data_size = total_training_data_size
    self.test_dataloader = test_dataloader
    self.rounds = rounds
    self.device = device

  def train_and_test(self):
    self.train()
    self.test()

  def train(self):
    self.model = self.model_class()

    for r in range(self.rounds):
      self._select_clients(r)
      self._send_model_and_train()
      self._aggregate_models()

  def test(self):
    accuracies = []
    losses = []
    self.model.to(self.device)
    with torch.no_grad():
      for data, targets in self.test_dataloader:
        data, targets = data.to(self.device), targets.to(self.device)
        output = self.model(data)
        losses.append(self.model.compute_loss(output, targets).item())
        acc = (torch.argmax(output, 1) == targets).float().sum().cpu()
        accuracies.append(acc)
    accuracies = np.array(accuracies)
    losses = np.array(losses)
    print(f"Global model test accuracy: {np.array(accuracies).mean():.3f}")

  def _send_model_and_train(self):
    for client in self.current_round_clients:
      params = deepcopy(list(self.model.parameters()))
      client.update_model(params)
      client.train()

  def _aggregate_models(self):
    global_model_contribution_weight = 0
    for client in self.current_round_clients:
      client_params = client.get_model_params()
      client_contribution_weight = client.training_data_size / self.total_training_data_size

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
    self.current_round_clients = [self.clients[i] for i in client_indexes]
    return self.current_round_clients
