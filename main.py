import argparse
import os
import warnings
from copy import deepcopy

import pytorch_lightning as pl
import torch
from torch.utils.data import random_split

from attacks import model_extraction, membership_inference
from datasets.dataset_util import *
from fl.aggregator import Aggregator
from fl.fl import FLEnvironment
from models import Cifar10Net
from msdp import MSPDTrainer
from dp_stages import Stages
from config import ExperimentConfig


def train(args):
  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  dataloaders = load_dataset('cifar10', args)
  model = Cifar10Net().to(device)
  optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

  trainer = MSPDTrainer(model=model,
                        optimizer=optimizer,
                        data_loaders=dataloaders,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        device=device,
                        save_checkpoint=True
                        )

  trainer.attach_stage(Stages.STAGE_1, {'eps': args.eps1, 'max_grad_norm': args.max_grad_norm})
  trainer.attach_stage(Stages.STAGE_2, {'noise_multiplier': args.noise_multiplier, 'max_grad_norm': args.max_grad_norm})
  trainer.attach_stage(Stages.STAGE_3, {'eps': args.eps3, 'max_weight_norm': args.max_weight_norm})

  model = trainer.train_and_test()
  return model


def attack_model(model, num_data=10000):
  def truncate(ds, size):
    idxs = np.random.permutation(len(ds))[:size]
    ds.data = ds.data[idxs]
    ds.targets = np.array(ds.targets)[idxs]
    return ds

  def merge(ds1, ds2):
    ds = deepcopy(ds1)
    ds.data = np.concatenate([ds1.data, ds2.data])
    ds.targets = np.concatenate([ds1.targets, ds2.targets])
    idxs = np.random.permutation(len(ds.data))
    ds.data = ds.data[idxs]
    ds.targets = ds.targets[idxs]
    return ds

  # Obtain seed (or public) data to be used in extraction
  train_dataset, _, test_dataset = get_cifar10_dataset()
  train_dataset = truncate(train_dataset, num_data // 2)
  test_dataset = truncate(test_dataset, num_data // 2)
  dataset = merge(train_dataset, test_dataset)

  split_ratio = 0.1
  train_data, test_data = random_split(dataset,
                                       [int((1 - split_ratio) * len(dataset)),
                                        int(split_ratio * len(dataset))])
  train_data = [train_data.dataset[i] for i in train_data.indices]
  test_data = [test_data.dataset[i] for i in test_data.indices]

  data_shape = (1, 3, 32, 32)

  # model_extraction(model=model,
  #                  query_limit=num_data,
  #                  victim_input_shape=data_shape,
  #                  number_of_targets=10,
  #                  attacker_input_shape=data_shape,
  #                  synthesizer_name="copycat",
  #                  substitute_architecture=Cifar10Net,
  #                  attack_train_data=train_data,
  #                  attack_test_data=test_data,
  #                  max_epochs=30)

  membership_inference(model=model,
                       query_limit=num_data,
                       victim_input_shape=data_shape,
                       number_of_targets=10,
                       attacker_input_shape=data_shape,
                       synthesizer_name="copycat",
                       substitute_architecture=Cifar10Net,
                       attack_train_data=train_data,
                       attack_test_data=test_data,
                       max_epochs=25)


def train_opacus(args):
  from dp.opacus_dp import opacus_training
  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  dataloaders = load_dataset('cifar10', args)
  model = Cifar10Net().to(device)
  opacus_training(model, dataloaders, args)


def main():
  args = ExperimentConfig()

  # Deterministic, reproducible behaviour
  pl.seed_everything(args.seed)
  if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  # train_dataset, test_dataset = get_dataset('cifar10', False)
  # args.experiment_id = _get_next_available_dir('./lightning_logs', 'experiment', False, False)
  # FLEnvironment(model_class=Cifar10Net,
  #               train_dataset=train_dataset,
  #               test_dataset=test_dataset,
  #               num_clients=2,
  #               aggregator_class=Aggregator,
  #               rounds=1,
  #               device=device,
  #               client_optimizer_class=torch.optim.SGD,
  #               clients_per_round=0,
  #               client_local_test_split=0.1,
  #               partition_method='homogeneous',
  #               alpha=10,
  #               args=args)

  # model = train(args)
  model = Cifar10Net.load_from_checkpoint('checkpoints_opacus_private/opacus_model.pth')
  attack_model(model)
  # train_opacus(args)


def _get_next_available_dir(root, dir_name, absolute_path=True, create=True):
  checkpoint_dir_base = os.path.join(root, dir_name)
  dir_id = 1
  checkpoint_dir = f"{checkpoint_dir_base}_{dir_id}"
  while os.path.exists(checkpoint_dir):
    dir_id += 1
    checkpoint_dir = f"{checkpoint_dir_base}_{dir_id}"
  if create:
    os.mkdir(checkpoint_dir)
  if absolute_path:
    return checkpoint_dir
  else:
    return f"{dir_name}_{dir_id}"



if __name__ == '__main__':
  warnings.filterwarnings("ignore")
  main()
