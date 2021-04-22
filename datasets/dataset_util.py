import numpy as np
import torch
from torch.utils.data import DataLoader, sampler
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


def load_cifar10(test_kwargs, train_kwargs):
  train_dataset, valid_dataset, test_dataset = get_cifar10_dataset()
  training_dataset_size = len(train_dataset)
  indices = list(range(training_dataset_size))
  split = int(np.floor(0.1 * training_dataset_size))
  np.random.shuffle(indices)
  # Split the training dataset
  train_idx, valid_idx = indices[split:], indices[:split]
  train_sampler = sampler.SubsetRandomSampler(train_idx)
  valid_sampler = sampler.SubsetRandomSampler(valid_idx)
  # Define the data loaders
  train_kwargs["sampler"] = train_sampler
  valid_kwargs = test_kwargs.copy()
  valid_kwargs["sampler"] = valid_sampler
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
      self.targets = np.array(self.targets)[indices]
      self.dataset_class = dataset_class

  return TruncatedDataset(root, indices, train, download, transform)
