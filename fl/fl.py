import os
import sys
from copy import deepcopy
from typing import Optional, Tuple, Type, List, Dict, Union

from matplotlib import pyplot as plt

from datasets.dataset_util import truncate_dataset, train_test_split
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from fl.aggregator import Aggregator
from fl.client import Client
from log import Logger


class FLEnvironment:
  """ Federeated Learning simulated Environment

    :param model_class: The class of the model to be trained.

    :param train_dataset: The training dataset.

    :param test_dataset: The test dataset used to test the global model.

    :param num_clients: The number of clients to take part into training.

    :param aggregator_class: The type of aggregator.

    :param rounds: Number of rounds to run the federated simulation

    :param device: the device on which the models will be trained and tested.

    :param client_optimizer_class: the optimizer type.

    :param partition_method: The partition method used to split the training
            data for the clients. Could be ;homogeneous' or 'heterogeneous'.
              * If 'homogeneous', then each client will get a uniformly
              distributed slice of data (including uniformly distributed
              class partition).
              * If 'heterogeneous', then LDA will be used to allocate the data.

    :param alpha: Dirichlet distribution concentration parameter. Controls the
            data distribution among clients. If alpha=infinity, then the
            allocation will be uniform, and when alpha=0, the data allocation
            is non i.i.d. See https://arxiv.org/pdf/1909.06335.pdf

    :param clients_per_round: The number of clients selected each round.
            If set to 0, the total number of clients will be used.

    :param client_local_test_split: The ratio used for the local test splits.
            If 'True' then a default value of 0.1 will be used.

    :param logger: The logger

    :param args: additional arguments for clients and trainers
  """

  def __init__(self,
               model_class: Type[pl.LightningModule],
               train_dataset: Dataset,
               test_dataset: Dataset,
               num_clients: int,
               aggregator_class: Type[Aggregator],
               rounds: int,
               device: torch.device,
               client_optimizer_class: Type[torch.optim.Optimizer],
               partition_method: Optional[str] = 'heterogeneous',
               alpha: Optional[float] = 1,
               clients_per_round: Optional[int] = 0,
               client_local_test_split: Optional[Union[bool, float]] = False,
               logger: Optional[Logger] = None,
               args=None):

    self.num_clients = num_clients
    self.rounds = rounds
    self.alpha = alpha
    self.device = device
    self.args = args

    self.logger = logger
    if logger is None:
      self.logger = Logger([sys.stdout, f'{args.save_model_path}/fl.log'])
    self.logger.log(self.args)

    # Allocate data for each client and ge the global val/test splits
    training_data_splits, val_dataset = self._split_dataset(train_dataset,
                                                            partition_method)

    # Build the global test dataloader
    self.use_cuda = not args.no_cuda and torch.cuda.is_available()
    train_kwargs = {"batch_size": self.args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if self.use_cuda:
      cuda_kwargs = {"num_workers": 1, "pin_memory": True}
      train_kwargs.update(cuda_kwargs)
      test_kwargs.update(cuda_kwargs)
    val_loader = DataLoader(val_dataset, **train_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)

    # Initialize the clients
    self.clients, n_training_data = self._init_clients(model_class,
                                                       client_optimizer_class,
                                                       training_data_splits,
                                                       client_local_test_split,
                                                       train_kwargs,
                                                       test_kwargs)
    clients_per_round = clients_per_round if clients_per_round > 0 else len(self.clients)

    # Initialize the aggregator
    if args.stage4:
      eps = args.eps3
    else:
      eps = None
    self.aggregator = aggregator_class(model_class=model_class,
                                       clients=self.clients,
                                       clients_per_round=clients_per_round,
                                       val_dataloader=val_loader,
                                       test_dataloader=test_loader,
                                       total_training_data_size=n_training_data,
                                       rounds=rounds,
                                       epsilon=eps,
                                       max_weight_norm=args.max_weight_norm_aggregated,
                                       device=device,
                                       logger=self.logger)

    # Start the FL simulation
    self.log("FL simulation started...")
    self.aggregator.train_and_test()
    if args.save_model:
      self.aggregator.save_model(
        os.path.join(args.save_model_path, 'final.ckpt')
      )
      self.aggregator.save_stats(
        os.path.join(args.save_model_path, 'stats.npy')
      )

  def log(self, *msg):
    self.logger.log(*msg, module='FLEnv')

  def get_model(self):
    return self.aggregator.model

  def _init_clients(self, model_class: Type[pl.LightningModule],
                    optimizer_class: torch.optim,
                    training_data_splits: Dict[int, Dataset],
                    client_local_test_split: bool,
                    train_kwargs: dict,
                    test_kwargs: dict) -> Tuple[List[Client], int]:
    """
    Creates the client instances with the specified parameters

    :param model_class:
    :param optimizer_class: The optimizer class used to optimize the client's model
    :param training_data_splits:
    :param client_local_test_split:
    :param train_kwargs:
    :param test_kwargs:
    :return: The list of clients and the total number of training datapoints
    """
    self.log("Creating the clients...")
    clients = []
    n_training_data = 0

    client_class_distributions = dict()
    for client_id in range(self.num_clients):
      # Create the data loaders
      client_dataset = training_data_splits[client_id]

      present_classes = np.unique(client_dataset.targets)
      class_distribution = np.bincount(client_dataset.targets)
      client_class_distributions[client_id] = (present_classes, class_distribution)

      # Generate local test splits
      if client_local_test_split:
        train_dataset, test_dataset = train_test_split(client_dataset,
                                                       client_local_test_split)
        train_loader = DataLoader(train_dataset, **train_kwargs)
        test_loader = DataLoader(test_dataset, **test_kwargs)
        dataloaders = [train_loader, test_loader]
        n_training_data += len(train_dataset)
      else:
        dataloaders = DataLoader(client_dataset, **train_kwargs)
        n_training_data += len(client_dataset)

      client = Client(id=client_id,
                      model_class=model_class,
                      dataloaders=dataloaders,
                      epochs=self.args.epochs,
                      batch_size=self.args.batch_size,
                      optimizer_class=optimizer_class,
                      learning_rate=self.args.lr,
                      weight_decay=self.args.weight_decay,
                      device=self.device,
                      virtual_batches=self.args.virtual_batches,
                      optimizer_momentum=self.args.momentum,
                      eps1=self.args.eps1,
                      noise_multiplier=self.args.noise_multiplier,
                      max_grad_norm=self.args.max_grad_norm,
                      eps3=self.args.eps3,
                      max_weight_norm=self.args.max_weight_norm,
                      logger=self.logger,
                      experiment_id=self.args.experiment_id,
                      args=self.args)
      clients.append(client)

    self._plot_client_class_distribution(client_class_distributions, client_dataset.classes)
    return clients, n_training_data

  def _split_dataset(self,
                     train_dataset: Dataset,
                     partition_method: str
                     ) -> Tuple[Dict[int, Dataset], Dataset]:
    """
    Fetches and splits the training dataset foe each client

    :param train_dataset: the training dataset
    :param partition_method: 'homogeneous' or 'heterogeneous'
    :return: * split data for each client id
             * the validation data split
    """
    (train_dataset, val_dataset,
     allocation, _) = self._allocate_data(train_dataset,
                                          self.num_clients,
                                          partition_method=partition_method,
                                          alpha=self.alpha)
    splits = {}
    self.log('Building client datasets...')
    for split_id in range(self.num_clients):
      ds = deepcopy(train_dataset)
      ds.data = ds.data[allocation[split_id]]
      ds.targets = ds.targets[allocation[split_id]]
      splits[split_id] = ds

    return splits, val_dataset

  def _allocate_data(self,
                     train_dataset: Dataset,
                     num_of_splits: int,
                     partition_method: Optional[str] = 'heterogeneous',
                     alpha: Optional[float] = 0.5):
    """
    Allocates the training data into a given number of splits,
    keeping a validation split

    Based on FedML partition method https://github.com/FedML-AI/FedML

    :param train_dataset: the training dataset
    :param partition_method: 'homogeneous' or 'heterogeneous'
    :param num_of_splits: the number of splits in which the data is partitioned
    :param alpha: Dirichlet distribution concentration parameter.
                  Controls the data distribution among clients.
                  See https://arxiv.org/pdf/1909.06335.pdf

    :return: The allocation dictionary dict[split_id, np.ndarray[indices]]
    """
    # Get the dataset
    train_dataset, val_dataset = train_test_split(train_dataset, 0.1, True)

    train_dataset.targets = train_dataset.targets
    training_dataset_size = len(train_dataset.data)
    num_classes = len(train_dataset.classes)

    self.log(f'Train data size: {training_dataset_size}, '
             f'Global Val data size: {len(val_dataset.data)}')
    self.log(f"Allocating data using {partition_method} splits...")
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
          proportions = np.array([p * (len(idx_j) < split_size)
                                  for p, idx_j in zip(proportions, split_indices)])
          proportions = proportions / proportions.sum()
          proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
          split_indices = [idx_j + idx.tolist()
                           for idx_j, idx in
                           zip(split_indices, np.split(class_indices, proportions))]
          min_size = min([len(idx_j) for idx_j in split_indices])

      for split_idx in range(num_of_splits):
        np.random.shuffle(split_indices[split_idx])
        id2idxs[split_idx] = np.array(split_indices[split_idx])

    elif partition_method == "homogeneous":
      indices = np.random.permutation(training_dataset_size)
      split_indices = np.array_split(indices, num_of_splits)
      id2idxs = {i: np.array(split_indices[i]) for i in range(num_of_splits)}

    else:
      raise ValueError("Invalid partition method")
    return train_dataset, val_dataset, id2idxs, num_classes

  def _plot_client_class_distribution(
      self,
      client_class_distributions: Dict[int, Tuple[np.ndarray, np.ndarray]],
      class_names: List[str]
  ):

    def draw_plot(per_client_distribution, class_names):
      """
      Source
      https://matplotlib.org/stable/gallery/lines_bars_and_markers/
          horizontal_barchart_distribution.html
      """
      labels = list(per_client_distribution.keys())
      data = np.array(list(per_client_distribution.values()))
      data_cum = data.cumsum(axis=1)
      category_colors = plt.get_cmap('tab20c')(
        np.linspace(0.15, 0.85, data.shape[1]))

      fig, ax = plt.subplots(figsize=(11, 5))
      ax.invert_yaxis()
      ax.xaxis.set_visible(False)
      ax.set_xlim(0, np.sum(data, axis=1).max())

      for i, (colname, color) in enumerate(zip(class_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        try:
          ax.bar_label(rects, label_type='center', color=text_color)
        except:
          pass
      ax.legend(ncol=len(class_names), bbox_to_anchor=(0, 1),
                loc='lower left', fontsize='small')

      return fig, ax

    total_num_classes = len(class_names)
    per_client_distribution = dict()

    for client, (present_cls, client_cls_distr) in client_class_distributions.items():
      k = f'Client {client}'
      per_client_distribution[k] = [0] * total_num_classes
      for i in range(len(present_cls)):
        per_client_distribution[k][present_cls[i]] = client_cls_distr[i]

    fig, ax = draw_plot(per_client_distribution, class_names)
    fig.savefig(f"{self.args.save_dir}/clients_class_distributions.png")


