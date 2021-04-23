from copy import deepcopy
from typing import Optional, Tuple, Type, List, Dict, Union

from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np

from datasets.dataset_util import get_dataset, get_dataset_transform, build_truncated_dataset
from fl.aggregator import Aggregator
from fl.client import Client
import pytorch_lightning as pl

from pytorch_lightning import LightningModule


class FLEnvironment:
  """ Federeated Learning simulated Environment

    """

  def __init__(self,
               model_class: Type[pl.LightningModule],
               dataset_name: str,
               dataset_root: str,
               num_clients: int,
               aggregator_class: Type[Aggregator],
               rounds: int,
               device: torch.device,
               client_optimizer_class: Type[torch.optim.Optimizer],
               alpha: Optional[float] = 0.5,
               clients_per_round: Optional[int] = 0,
               client_local_test_split: Optional[Union[bool, float]] = False,
               args=None):

    self.num_clients = num_clients
    self.rounds = rounds
    self.alpha = alpha
    self.device = device
    self.args = args

    # Allocate data for each client
    training_data_splits, global_test_data, n_training_data = self._split_dataset(dataset_name, dataset_root)

    # Build the global test dataloader
    self.use_cuda = not args.no_cuda and torch.cuda.is_available()
    train_kwargs = {"batch_size": self.args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if self.use_cuda:
      cuda_kwargs = {"num_workers": 1, "pin_memory": True}
      train_kwargs.update(cuda_kwargs)
      test_kwargs.update(cuda_kwargs)
    test_loader = DataLoader(global_test_data, **test_kwargs)

    # Initialize the clients
    self.clients = self._init_clients(model_class, client_optimizer_class, training_data_splits,
                                      client_local_test_split, train_kwargs, test_kwargs)
    clients_per_round = clients_per_round if clients_per_round > 0 else len(self.clients)

    # Initialize the aggregator
    self.aggregator = aggregator_class(model_class=model_class,
                                       clients=self.clients,
                                       clients_per_round=clients_per_round,
                                       test_dataloader=test_loader,
                                       total_training_data_size=n_training_data,
                                       rounds=rounds,
                                       device=device)

    # Start the FL simulation
    self.aggregator.train_and_test()

  def _init_clients(self, model_class: Type[pl.LightningModule],
                    optimizer_class: torch.optim,
                    training_data_splits: Dict[int, Dataset],
                    client_local_test_split: bool,
                    train_kwargs: dict,
                    test_kwargs: dict) -> List[Client]:
    """
    Creates the client instances with the specified parameters

    :param model_class:
    :param optimizer_class: The optimizer class used to optimize the client's model
    :param training_data_splits:
    :param client_local_test_split:
    :param train_kwargs:
    :param test_kwargs:
    :return: The list of clients
    """

    def get_dataloader(dataset, indices, kwargs):
      dataset.data = dataset.data[indices]
      dataset.targets = dataset.targets[indices]
      return DataLoader(dataset, **kwargs)

    clients = []
    for client_id in range(self.num_clients):
      # Create the data loaders
      client_dataset = training_data_splits[client_id]

      if client_local_test_split:
        dataset_size = len(client_dataset)
        if isinstance(client_local_test_split, bool):
          test_split_ratio = 0.1
        else:
          test_split_ratio = client_local_test_split \
            if client_local_test_split < 0.5 \
            else 1 - client_local_test_split

        # The data is already shuffled so we can simply split it.
        split_index = int(dataset_size * test_split_ratio)

        test_dataset = deepcopy(client_dataset)
        test_indices = np.array(range(split_index))
        test_loader = get_dataloader(test_dataset, test_indices, test_kwargs)

        train_indices = np.array(range(split_index, dataset_size))
        train_dataset = client_dataset
        train_loader = get_dataloader(train_dataset, train_indices, train_kwargs)

        dataloaders = [train_loader, test_loader]
      else:
        dataloaders = DataLoader(client_dataset, **train_kwargs)

      client = Client(id=client_id,
                      model_class=model_class,
                      dataloaders=dataloaders,
                      epochs=1,#self.args.epochs,
                      batch_size=self.args.batch_size,
                      optimizer_class=optimizer_class,
                      learning_rate=self.args.lr,
                      weight_decay=self.args.weight_decay,
                      device=self.device,
                      optimizer_momentum=self.args.momentum,
                      eps1=self.args.eps1,
                      noise_multiplier=self.args.noise_multiplier,
                      max_grad_norm=self.args.max_grad_norm,
                      eps3=self.args.eps3,
                      max_weight_norm=self.args.max_weight_norm)
      clients.append(client)
    return clients

  def _split_dataset(self, dataset_name: str, dataset_root: str):
    """
    Fetches and splits the training dataset foe each client

    :param dataset_name: the name of the dataset
    :param dataset_root: the root location of the dataset used to load or download it
    :return: the splits dictionary: dict[int, TruncatedDataset] and the global test dataset
    """
    train_dataset, test_dataset, allocation, _, n_train_data = self._allocate_data(dataset_name,
                                                                                   self.num_clients,
                                                                                   alpha=self.alpha)
    splits = {}
    transform = get_dataset_transform(dataset_name)[0]
    for split_id in range(self.num_clients):
      splits[split_id] = build_truncated_dataset(dataset_name, dataset_root, train=True,
                                                 transform=transform, download=True,
                                                 indices=allocation[split_id])

    return splits, test_dataset, n_train_data

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
    train_dataset.targets = np.array(train_dataset.targets)
    training_dataset_size = len(train_dataset.data)
    num_classes = len(train_dataset.classes)
    print("[FLEnv] Allocating data")
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
        id2idxs[split_idx] = np.array(split_indices[split_idx])

    elif partition_method == "homogenous":
      indices = np.random.permutation(training_dataset_size)
      split_indices = np.array_split(indices, num_of_splits)
      id2idxs = {i: np.array(split_indices[i]) for i in range(num_of_splits)}

    else:
      raise ValueError("Invalid partition method")

    return train_dataset, test_dataset, id2idxs, num_classes, len(train_dataset.data)
