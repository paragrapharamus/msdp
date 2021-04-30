import argparse
import os
import warnings
from copy import deepcopy

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import random_split

from attacks import model_extraction, membership_inference_black_box, membership_inference_mia, model_extraction_2
from datasets.dataset_util import *
from fl.aggregator import Aggregator
from fl.fl import FLEnvironment
from models import Cifar10Net, AttackModel
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

  # trainer.attach_stage(Stages.STAGE_1, {'eps': args.eps1, 'max_grad_norm': args.max_grad_norm})
  # trainer.attach_stage(Stages.STAGE_2, {'noise_multiplier': args.noise_multiplier, 'max_grad_norm': args.max_grad_norm})
  # trainer.attach_stage(Stages.STAGE_3, {'eps': args.eps3, 'max_weight_norm': args.max_weight_norm})

  model = trainer.train_and_test()
  return model


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

  train_dataset, test_dataset = get_dataset('cifar10', False)
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
  model = Cifar10Net.load_from_checkpoint('checkpoints_8/checkpoint-epoch=05-valid_acc=0.72.ckpt').to(device)
  # membership_inference_mia(model=model,
  #                      model_cls=Cifar10Net,
  #                      attack_model_cls=AttackModel,
  #                      num_classes=10,
  #                      train_dataset=train_dataset,
  #                      test_dataset=test_dataset,
  #                      shadow_epochs=10,
  #                      shadow_dataset_size=4500,
  #                      attack_test_dataset_size=4000,
  #                      num_shadows=3,
  #                      attack_epochs=20,
  #                      seed=args.seed,
  #                      use_cuda=True)

  # membership_inference_black_box(model=model,
  #                                loss=nn.CrossEntropyLoss(),
  #                                optimizer_class=torch.optim.Adam,
  #                                train_dataset=train_dataset,
  #                                test_dataset=test_dataset,
  #                                input_shape=(3, 32, 32),
  #                                num_classes=10,
  #                                attack_train_ratio=0.5,
  #                                attack_test_ratio=0.5,
  #                                )

  model_extraction_2(model=model,
                     attack_model_cls=Cifar10Net,
                     loss=nn.CrossEntropyLoss(),
                     optimizer_class=torch.optim.Adam,
                     input_shape=(3, 32, 32),
                     num_classes=10,
                     test_dataset=test_dataset,
                     batch_size=args.batch_size,
                     epochs=5,
                     query_limit=9000,
                     )

  # model_extraction(model=model,
  #                  train_dataset=train_dataset,
  #                  test_dataset=test_dataset,
  #                  query_limit=len(test_dataset),
  #                  victim_input_shape=(1, 3, 32, 32),
  #                  num_classes=10,
  #                  attacker_input_shape=(1, 3, 32, 32),
  #                  synthesizer_name='copycat',
  #                  substitute_architecture=Cifar10Net,
  #                  max_epochs=25
  #                  )
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
