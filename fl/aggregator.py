from typing import Optional, Type, List
import numpy as np
from copy import deepcopy
from pytorch_lightning import LightningModule, Trainer
import torch
from torch.utils.data import DataLoader

from dp_stages import Stage4
from fl.client import Client
from log import Logger


class Aggregator:
  """
  FL aggregator. It implements federated averaging.
    -> https://arxiv.org/pdf/1602.05629.pdf

  :param model_class: The type of the global model
  :param clients: The list of clients
  :param clients_per_round: The number of clients to be selected each round
  :param total_training_data_size: The number of training data points of
            all clients.
  :param val_dataloader: The global validation data loader.
  :param test_dataloader: The global test data loader.
  :param rounds: The number of rounds of training.
  :param device: The device used to train and test the models on
  :param logger: The logger
  :param epsilon: The privacy budget for global DP (Stage 4)
  :param max_weight_norm: The maximum L2 norm of the weights
            of the aggregated model.
  """

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
               max_weight_norm: Optional[float] = 20):

    self.model_class = model_class
    self.model = None
    self.trainer = Trainer(weights_summary=None,
                           progress_bar_refresh_rate=0,
                           gpus=1)
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

    self.val_accuracies = []

  def train_and_test(self):
    self.train()
    self.log("Testing... ")
    self.test(self.test_dataloader)

  def train(self):
    self.model = self.model_class()

    for r in range(self.rounds):
      self.log(f'******** ROUND {r + 1} ********')

      if self.dp_stage:
        # Reset the values, to be properly assigned during aggregation
        self.max_exposures = 0
        self.min_client_dataset = self.total_training_data_size

      self._select_clients(r)
      self._send_model_and_train()
      self._aggregate_models()
      self.log("Validation... ")
      results = self.test(self.val_dataloader)
      self.val_accuracies.append(results[0]['test_acc_epoch'])

      if self.dp_stage:
        self.dp_stage.apply(self.model, self.rounds,
                            len(self.clients), self.clients_per_round,
                            self.min_client_dataset, self.max_exposures,
                            self.curr_round_training_dataset_size)

        self.log("Post DP validation results... ")
        self.test(self.val_dataloader)

  def test(self, loader):
    self.model.to(self.device)
    results = self.trainer.test(self.model, loader)
    self.log(f"Global model results: {results}")
    self.model.cpu()
    return results

  def save_model(self, path):
    self.trainer.save_checkpoint(path)

  def save_stats(self, path):
    with open(path, 'wb') as f:
      np.save(f, np.array(self.val_accuracies))

  def _send_model_and_train(self):
    for client in self.current_round_clients:
      with torch.no_grad():
        params = deepcopy(list(self.model.parameters()))
      client.update_model(params)
      client.train()
      client.test()

  def _aggregate_models(self):
    global_model_contribution_weight = 0  # first, take the whole model
    for client in self.current_round_clients:
      client_params = client.get_model_params()
      client_contribution_weight = client.training_data_size / \
                                   self.curr_round_training_dataset_size

      if self.dp_stage:
        self.max_exposures = max(self.max_exposures, client.exposures)
        self.min_client_dataset = min(self.min_client_dataset,
                                      client.training_data_size)

      global_model_params = self.model.named_parameters()
      dict_global_model_params = dict(global_model_params)

      # Average the models into a global model
      with torch.no_grad():
        for name, param in client_params:
          if name in dict_global_model_params:
            averaged_weights = client_contribution_weight * param.data \
                               + global_model_contribution_weight \
                               * dict_global_model_params[name].data
            dict_global_model_params[name].data.copy_(averaged_weights)

      # use the whole global model and a contribution fraction from the client's model
      global_model_contribution_weight = 1

  def _select_clients(self, current_round):
    np.random.seed(current_round)
    client_indices = np.random.choice(range(len(self.clients)),
                                      self.clients_per_round,
                                      replace=False)
    self.current_round_clients = []
    current_training_data = 0
    for ci in client_indices:
      self.current_round_clients.append(self.clients[ci])
      current_training_data += self.clients[ci].training_data_size
    self.curr_round_training_dataset_size = current_training_data
    return self.current_round_clients

  def log(self, *msg):
    if self.logger:
      self.logger.log(*msg, module='Aggregator')
