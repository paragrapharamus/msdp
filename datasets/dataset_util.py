import os
from copy import deepcopy
from typing import Optional, Union, Tuple, Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import image as mpimg
from torch.utils.data import DataLoader, sampler, Dataset
from torchvision import datasets
from torchvision import transforms


class Cifar10Dataset(datasets.CIFAR10):
  transform_trainval = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                                std=[0.2023, 0.1994, 0.2010])])
  transform_test = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

  input_shape = (1, 3, 32, 32)
  num_classes = 10
  name = 'cifar10'

  def __init__(self, *args, **kwargs) -> None:
    super(Cifar10Dataset, self).__init__(*args, **kwargs)

    self.data = np.moveaxis(self.data.data, -1, 1)
    self.targets = np.array(self.targets)

  def __getitem__(self, index: int) -> Tuple[Any, Any]:
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target = self.data[index], self.targets[index]

    img = np.moveaxis(img, 0, -1)

    # doing this so that it is consistent with all other datasets
    # to return a PIL Image
    img = Image.fromarray(img)

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target


class MNISTDataset(datasets.MNIST):
  transform_trainval = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,))])
  transform_test = transform_trainval
  input_shape = (1, 1, 28, 28)
  num_classes = 10
  name = 'mnist'

  def __init__(self, *args, **kwargs) -> None:
    super(MNISTDataset, self).__init__(*args, **kwargs)
    self.data = self.data.numpy()
    self.targets = self.targets.numpy()

  def __getitem__(self, index: int) -> Tuple[Any, Any]:
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target = self.data[index], int(self.targets[index])

    # doing this so that it is consistent with all other datasets
    # to return a PIL Image
    img = Image.fromarray(img, mode='L')

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target


class DRDataset(Dataset):
  transform_trainval = transforms.Compose([
    transforms.Resize(265),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])
  transform_test = transform_trainval

  input_shape = (1, 3, 224, 224)
  num_classes = 4
  name = 'dr'

  def __init__(self, data_dir, train=True, **kwargs):
    super(DRDataset, self).__init__()
    data_dir = os.path.join(data_dir, 'diabetic_retinopathy')
    self.image_dir = os.path.join(data_dir, 'images/')
    self.train = train
    if train:
      _df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    else:
      _df = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    self.data = np.array(list(map(lambda id: id + '.png', _df.id_code[:])))
    self.targets = np.array(_df.diagnosis)
    self.classes = ['No DR', 'Mild DR', 'Moderate DR', 'Severe  DR', 'Proliferative  DR']
    self.transform = DRDataset.transform_trainval

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    img_name = self.data[index]
    label = int(self.targets[index])
    img_path = os.path.join(self.image_dir, img_name)
    image = Image.open(img_path)
    # image = (image + 1) * 127.5
    if self.transform is not None:
      image = self.transform(image)
    return image.numpy(), label


def load_dataset(dataset_name, args):
  use_cuda = not args.no_cuda and torch.cuda.is_available()
  train_kwargs = {"batch_size": args.batch_size}
  test_kwargs = {"batch_size": args.test_batch_size}
  if use_cuda:
    cuda_kwargs = {"num_workers": 2, "pin_memory": True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
    print('GPU: ' + str(torch.cuda.get_device_name(0)))
  else:
    print('CPU')

  if dataset_name == 'cifar10':
    return load_cifar10(train_kwargs, test_kwargs)
  elif dataset_name == 'mnist':
    return load_mnist(train_kwargs, test_kwargs)
  elif dataset_name == 'dr':
    return load_dr(train_kwargs, test_kwargs)
  else:
    raise NotImplementedError('Unsupported dataset')


def _load(train_dataset, valid_dataset, test_dataset, train_kwargs, test_kwargs):
  # Define the data loaders
  valid_kwargs = test_kwargs.copy()
  train_loader = DataLoader(train_dataset, **train_kwargs)
  valid_loader = DataLoader(valid_dataset, **valid_kwargs)
  test_loader = DataLoader(test_dataset, **test_kwargs)
  return train_loader, valid_loader, test_loader


def load_cifar10(train_kwargs, test_kwargs):
  train_dataset, valid_dataset, test_dataset = get_cifar10_dataset(validation_dataset=True)
  return _load(train_dataset, valid_dataset, test_dataset, train_kwargs, test_kwargs)


def load_mnist(train_kwargs, test_kwargs):
  train_dataset, valid_dataset, test_dataset = get_mnist_dataset(validation_dataset=True)
  return _load(train_dataset, valid_dataset, test_dataset, train_kwargs, test_kwargs)


def load_dr(train_kwargs, test_kwargs):
  train_dataset, valid_dataset, test_dataset = get_dr_dataset(validation_dataset=True)
  return _load(train_dataset, valid_dataset, test_dataset, train_kwargs, test_kwargs)


def get_dataset(dataset_name, validation_dataset=True):
  if dataset_name == 'cifar10':
    return get_cifar10_dataset(validation_dataset=validation_dataset)
  elif dataset_name == 'mnist':
    return get_mnist_dataset(validation_dataset=validation_dataset)
  elif dataset_name == 'dr':
    return get_dr_dataset(validation_dataset=validation_dataset)
  else:
    raise NotImplementedError('Unsupported dataset')


def _get_dataset(dataset_class, validation_dataset=True):
  train_dataset = dataset_class("../data", train=True, download=True, transform=dataset_class.transform_trainval)
  test_dataset = dataset_class("../data", train=False, transform=dataset_class.transform_test)
  if validation_dataset:
    train_dataset, valid_dataset = train_test_split(train_dataset, shuffle=True)
    return train_dataset, valid_dataset, test_dataset
  else:
    return train_dataset, test_dataset


def get_cifar10_dataset(validation_dataset=True):
  return _get_dataset(Cifar10Dataset, validation_dataset)


def get_mnist_dataset(validation_dataset=True):
  return _get_dataset(MNISTDataset, validation_dataset)


def get_dr_dataset(validation_dataset=True):
  return _get_dataset(DRDataset, validation_dataset)


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
                     shuffle: bool = False,
                     seed: int = 42
                     ):
  dataset_size = len(dataset)
  if isinstance(test_split_ratio, bool):
    test_split_ratio = 0.1
  else:
    # make sure the smaller fraction is for the test split
    test_split_ratio = test_split_ratio \
      if test_split_ratio < 0.5 \
      else 1 - test_split_ratio

  if shuffle:
    indices = np.random.RandomState(seed=seed).permutation(dataset_size)
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
    dataset_class = Cifar10Dataset
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
