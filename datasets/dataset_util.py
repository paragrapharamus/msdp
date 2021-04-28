from copy import deepcopy
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, sampler, Dataset
from torchvision import datasets
from torchvision import transforms


def load_dataset(dataset_name, args):
  use_cuda = not args.no_cuda and torch.cuda.is_available()
  train_kwargs = {"batch_size": args.batch_size}
  test_kwargs = {"batch_size": args.test_batch_size}
  if use_cuda:
    cuda_kwargs = {"num_workers": 1, "pin_memory": True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
    print('GPU: ' + str(torch.cuda.get_device_name(0)))
  else:
    print('CPU')

  if dataset_name == 'cifar10':
    return load_cifar10(test_kwargs, train_kwargs)
  else:
    raise NotImplementedError('Unsupported dataset')


def get_dataset(dataset_name, validation_dataset=True):
  if dataset_name == 'cifar10':
    return get_cifar10_dataset(validation_dataset=validation_dataset)
  else:
    raise NotImplementedError('Unsupported dataset')


def get_dataset_transform(dataset_name):
  if dataset_name == 'cifar10':
    return get_cifar10_transforms()
  else:
    raise NotImplementedError('Unsupported dataset')


def get_cifar10_transforms():
  transform_trainval = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                                std=[0.2023, 0.1994, 0.2010])])
  transform_test = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])
  return transform_trainval, transform_test


def get_cifar10_dataset(validation_dataset=True):
  transform_trainval, transform_test = get_cifar10_transforms()

  train_dataset = datasets.CIFAR10("../data", train=True, download=True, transform=transform_trainval)
  test_dataset = datasets.CIFAR10("../data", train=False, transform=transform_test)

  if validation_dataset:
    valid_dataset = datasets.CIFAR10("../data", train=True, download=True, transform=transform_trainval)
    return train_dataset, valid_dataset, test_dataset
  else:
    return train_dataset, test_dataset


def truncate_dataset(ds: Dataset, idxs: np.ndarray):
  """
    Returns the truncated dataset that contains data
    corresponding to the given indices
  """
  ds.data = ds.data[idxs]
  if not isinstance(ds.targets, np.ndarray):
    ds.targets = np.array(ds.targets)
  ds.targets = ds.targets[idxs]
  return ds


def merge_datasets(ds1: Dataset, ds2: Dataset):
  """
    Merges the two datasets without altering them
  """
  ds = deepcopy(ds1)
  ds.data = np.concatenate([ds1.data, ds2.data])
  ds.targets = np.concatenate([ds1.targets, ds2.targets])
  idxs = np.random.permutation(len(ds.data))
  ds.data = ds.data[idxs]
  ds.targets = ds.targets[idxs]
  return ds


def train_test_split(dataset: Dataset,
                     test_split_ratio: Optional[Union[float, bool]] = 0.1,
                     shuffle=False):
  dataset_size = len(dataset)
  if isinstance(test_split_ratio, bool):
    test_split_ratio = 0.1
  else:
    # make sure the smaller fraction is for the test split
    test_split_ratio = test_split_ratio \
      if test_split_ratio < 0.5 \
      else 1 - test_split_ratio

  if shuffle:
    indices = np.random.permutation(dataset_size)
    dataset.data = dataset.data[indices]
    if not isinstance(dataset.targets, np.ndarray):
      dataset.targets = np.array(dataset.targets)
    dataset.targets = dataset.targets[indices]

  split_index = int(dataset_size * test_split_ratio)
  # Get the test data split
  test_indices = np.array(range(split_index))
  test_dataset = truncate_dataset(deepcopy(dataset), test_indices)
  # Get the train data split
  train_indices = np.array(range(split_index, dataset_size))
  train_dataset = truncate_dataset(dataset, train_indices)
  return train_dataset, test_dataset


def load_cifar10(test_kwargs, train_kwargs):
  train_dataset, test_dataset = get_cifar10_dataset(validation_dataset=False)
  train_dataset, valid_dataset = train_test_split(train_dataset, shuffle=True)
  # Define the data loaders
  valid_kwargs = test_kwargs.copy()
  train_loader = DataLoader(train_dataset, **train_kwargs)
  valid_loader = DataLoader(valid_dataset, **valid_kwargs)
  test_loader = DataLoader(test_dataset, **test_kwargs)
  return train_loader, valid_loader, test_loader


def build_truncated_dataset(dataset_name: str,
                            root: str,
                            train: bool,
                            download: bool,
                            indices: np.ndarray,
                            transform=None) -> datasets:
  """
  Builds the specified torchvision dataset and truncates it by keeping only the specified indices
  """

  if dataset_name == 'cifar10':
    dataset_class = datasets.CIFAR10
  elif dataset_name == 'mnist':
    dataset_class = datasets.MNIST
  else:
    raise ValueError("Unknown dataset name")

  class TruncatedDataset(dataset_class):
    def __init__(self, root, indices, train=True, download=True, transform=None):
      super(TruncatedDataset, self).__init__(root, train=train, download=download, transform=transform)
      self.data = self.data[indices]
      if isinstance(self.targets, np.ndarray):
        self.target = self.targets[indices]
      else:
        # list
        self.targets = np.array(self.targets)[indices]
      self.dataset_class = dataset_class

  return TruncatedDataset(root, indices, train, download, transform)
