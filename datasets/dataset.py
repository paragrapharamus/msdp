import numpy as np
from torch.utils.data import DataLoader, sampler, random_split
from torchvision import datasets
from torchvision import transforms


def get_cifar10_dataset():
  transform_trainval = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                                std=[0.2023, 0.1994, 0.2010])])
  transform_test = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])
  train_dataset = datasets.CIFAR10("../data", train=True, download=True, transform=transform_trainval)
  valid_dataset = datasets.CIFAR10("../data", train=True, download=True, transform=transform_trainval)
  test_dataset = datasets.CIFAR10("../data", train=False, transform=transform_test)
  return train_dataset, valid_dataset, test_dataset


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
