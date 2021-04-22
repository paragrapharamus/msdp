from typing import Optional, Tuple, Type, Callable
from torchvision import datasets
import torch
import numpy as np

from datasets.dataset_util import get_dataset, get_dataset_transform, build_truncated_dataset
from fl.aggregator import Aggregator
from fl.client import Client
import pytorch_lightning as pl

from pytorch_lightning import LightningModule


class FLEnvironment:
  """ Federeated Learning simulated Environment

    :param dataset_name:
    :param dataset_root:
    :param num_clients:
    :param aggregator_class:
    :param rounds:
    :param alpha:
    :param args:
    """

  def __init__(self,
               model_class: pl.LightningModule,
               dataset_name: str,
               dataset_root: str,
               num_clients: int,
               aggregator_class: Callable[..., Aggregator],
               rounds: int,
               alpha: float,
               device: torch.device,
               client_optimizer_class: torch.optim,
               args=None):

    self.num_clients = num_clients
    self.rounds = rounds
    self.alpha = alpha
    self.device = device
    self.args = args

    # Allocate data for each client
    self.training_data_splits, self.global_test_data = self._split_dataset(dataset_name, dataset_root)

    self.clients = self._init_clients(client_optimizer_class)
    self.aggregator = aggregator_class(model_class, self.clients, rounds, device)

    self.aggregator.start()

  def _init_clients(self, optimizer_class):
    clients = []
    for client_id in range(self.num_clients):
      clients.append(Client(id=client_id,
                            data=self.training_data_splits[client_id],
                            epochs=self.args.epochs,
                            batch_size=self.args.batch_size,
                            optimizer_class=optimizer_class,
                            learning_rate=self.args.lr,
                            weight_decay=self.args.weight_decay,
                            momentum=self.args.momentum,
                            device=self.device,
                            eps1=self.args.eps1,
                            noise_multiplier=self.args.noise_multiplier,
                            max_grad_norm=self.args.max_grad_norm,
                            eps3=self.args.eps3,
                            max_weight_norm=self.args.max_weight_norm))
    return clients

  def _split_dataset(self, dataset_name: str, dataset_root: str):
    """
    Fetches and splits the training dataset foe each client

    :param dataset_name: the name of the dataset
    :param dataset_root: the root location of the dataset used to load or download it
    :return: the splits dictionary: dict[int, TruncatedDataset] and the global test dataset
    """
    train_dataset, test_dataset, allocation, num_classes = self._allocate_data(dataset_name,
                                                                               self.num_clients,
                                                                               alpha=self.alpha)
    splits = {}
    transform = get_dataset_transform(dataset_name)[0]
    for split_id in range(self.num_clients):
      splits[split_id] = build_truncated_dataset(dataset_name, dataset_root, train=True,
                                                 transform=transform, download=True,
                                                 indices=allocation[split_id])

    return splits, test_dataset

  @staticmethod
  def _allocate_data(dataset_name: str,
                     num_of_splits: int,
                     partition_method: Optional[str] = 'heterogeneous',
                     alpha: Optional[float] = 0.5):
    """
    Allocates the data into a given number of splits

    Based on FedML partition method https://github.com/FedML-AI/FedML

    :param dataset_name: the name of the dataset to be split
    :param partition_method: 'homogenous' or 'heterogeneous'
    :param num_of_splits: the number of splits in which the data is partitioned
    :param alpha: Dirichlet distribution concentration parameter. Controls the data distribution
                  among clients. See https://arxiv.org/pdf/1909.06335.pdf

    :return: The allocation dictionary dict[split_id, np.ndarray[indices]]
    """
    # Get the dataset
    train_dataset, test_dataset = get_dataset(dataset_name, validation_dataset=False)

    training_dataset_size = len(train_dataset.data)
    num_classes = len(train_dataset.classes)

    if partition_method == "heterogeneous":
      min_size = 0
      id2idxs = {}  # the allocation dictionary
      split_size = training_dataset_size // num_of_splits

      while min_size < 10:
        split_indices = [[] for _ in range(num_of_splits)]
        # for each class in the dataset
        for k in range(num_classes):
          class_indices = np.where(train_dataset.targets == k)[0]
          np.random.shuffle(class_indices)
          proportions = np.random.dirichlet(np.repeat(alpha, num_of_splits))

          #  Balance
          proportions = np.array([p * (len(idx_j) < split_size) for p, idx_j in zip(proportions, split_indices)])
          proportions = proportions / proportions.sum()
          proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
          split_indices = [idx_j + idx.tolist() for idx_j, idx in zip(split_indices, np.split(class_indices, proportions))]
          min_size = min([len(idx_j) for idx_j in split_indices])

      for split_idx in range(num_of_splits):
        np.random.shuffle(split_indices[split_idx])
        id2idxs[split_idx] = np.arrray(split_indices[split_idx])

    elif partition_method == "homogenous":
      indices = np.random.permutation(training_dataset_size)
      split_indices = np.array_split(indices, num_of_splits)
      id2idxs = {i: np.array(split_indices[i]) for i in range(num_of_splits)}

    else:
      raise ValueError("Invalid partition method")

    return train_dataset, test_dataset, id2idxs, num_classes
