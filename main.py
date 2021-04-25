import argparse
import os

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


def _parse_args():
  parser = argparse.ArgumentParser()
  ##########################
  # Basic training arguments
  ##########################
  parser.add_argument(
    "--batch-size",
    type=int,
    default=256,
    metavar="N",
    help="input batch size for training (default: 64)",
  )
  parser.add_argument(
    "--minibatch_size",
    type=int,
    default=1,
    metavar="N",
    help="input minibatch size for training (default: 1)",
  )
  parser.add_argument(
    "--test-batch-size",
    type=int,
    default=500,
    metavar="N",
    help="input batch size for testing (default: 100)",
  )
  parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    metavar="N",
    help="number of epochs to train (default: 20)",
  )
  parser.add_argument(
    "--lr",
    type=float,
    default=0.02,
    metavar="LR",
    help="learning rate (default: 0.02)",
  )
  parser.add_argument(
    "--gamma",
    type=float,
    default=0.7,
    metavar="M",
    help="Learning rate step gamma (default: 0.7)",
  )
  parser.add_argument(
    "--wd",
    "--weight-decay",
    default=5e-4,
    type=float,
    metavar="W",
    help="SGD weight decay (default: 5e-4)",
    dest="weight_decay",
  )
  parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="MOMENTUM",
    help="momentum"
  )
  parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    default=False,
    help="quickly check a single pass",
  )
  parser.add_argument(
    "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
  )

  parser.add_argument(
    "--save-model-path",
    type=str,
    default='./model.pth',
    help="path for Saving the current Model",
  )
  parser.add_argument(
    "--save-model",
    action="store_true",
    default=True,
    help="For Saving the current Model",
  )
  parser.add_argument(
    "--trained-model-path",
    type=str,
    default='./checkpoints/checkpoint.pth',
    help="the path of the trained model"
  )
  ##########################
  # MSDP arguments
  ##########################
  # Stage 1
  parser.add_argument(
    "--eps1",
    type=float,
    default=1,
    metavar="EPS1",
    help="Stage 1 Epsilon (Input perturbation)"
  )
  # Stage 2
  parser.add_argument(
    "--noise_multiplier",
    type=float,
    default=0.5,
    metavar="NOISE_MULTIPLIER",
    help="noise_multiplier (default: 1.0)",
  )
  parser.add_argument(
    "--max_grad_norm",
    type=float,
    default=2,
    metavar="MAX_GRAD_NORM",
    help="l2 gradient clipping norm (default: 1.0)",
  )
  # Stage 3
  parser.add_argument(
    "--eps3",
    type=float,
    default=1,
    metavar="EPS3",
    help="Stage 3 Epsilon (Output perturbation)"
  )
  parser.add_argument(
    "--max_weight_norm",
    type=float,
    default=2,
    metavar="MAX_WEIGHT_NORM",
    help="l2 weights clipping norm (default: 1.0)",
  )
  # Stage 4
  parser.add_argument(
    "--eps4",
    type=float,
    default=1,
    metavar="EPS3",
    help="Stage 4 Epsilon (Aggregation perturbation)"
  )
  parser.add_argument(
    "--max_weight_norm_aggregated",
    type=float,
    default=2,
    help="L2 weights clipping norm for stage 4 (default: 2.0)",
  )
  ##########################
  # FL arguments
  ##########################
  parser.add_argument(
    "--num-rounds",
    type=int,
    default=25,
    help="The number of federated learning rounds (default: 25)",
  )
  parser.add_argument(
    "--num-clients",
    type=int,
    default=10,
    help="The number of clients to take part in the FL (default: 10)",
  )
  parser.add_argument(
    "--clients-per-round",
    type=int,
    default=10,
    help="The number of clients to be selected each round (default: 10)",
  )
  parser.add_argument(
    "--partition-method",
    type=str,
    default='heterogeneous',
    help="The method for data partitioning. (default: 'heterogeneous')",
  )
  parser.add_argument(
    "--alpha",
    type=float,
    default=1,
    help="Controls the data heterogeneity if `partition-method='heterogeneous'` (default: 1.0)",
  )
  parser.add_argument(
    "--client-local-test-split",
    type=float,
    default=0.1,
    help="The fraction of the local training data to be used for testing (default: 0.1)",
  )
  args = parser.parse_args()
  return args


def train(args):
  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  dataloaders = load_dataset('cifar10', args)
  model = Cifar10Net().to(device)
  optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

  trainer = MSPDTrainer(model=model,
                        optimizer=optimizer,
                        data_loaders=dataloaders,
                        epochs=2,#args.epochs,
                        batch_size=args.batch_size,
                        device=device,
                        save_checkpoint=True
                        )

  # trainer.attach_stage(Stages.STAGE_1, {'eps': args.eps1})
  # trainer.attach_stage(Stages.STAGE_2, {'noise_multiplier': args.noise_multiplier, 'max_grad_norm': args.max_grad_norm})
  trainer.attach_stage(Stages.STAGE_3, {'eps': args.eps3, 'max_weight_norm': args.max_weight_norm})

  model = trainer.train_and_test()
  return model


def attack_model(model):
  # Obtain seed (or public) data to be used in extraction
  test_dataset = get_cifar10_dataset()[2]
  split_ratio = 0.1
  train_data, test_data = random_split(test_dataset,
                                       [int((1 - split_ratio) * len(test_dataset)),
                                        int(split_ratio * len(test_dataset))])
  train_data = [train_data.dataset[i] for i in train_data.indices]
  test_data = [test_data.dataset[i] for i in test_data.indices]
  data_shape = (1, 3, 32, 32)

  # model_extraction(model=model,
  #                  query_limit=10000,
  #                  victim_input_shape=data_shape,
  #                  number_of_targets=10,
  #                  attacker_input_shape=data_shape,
  #                  synthesizer_name="copycat",
  #                  substitute_architecture=Cifar10Net,
  #                  attack_train_data=train_data,
  #                  attack_test_data=test_data,
  #                  max_epochs=30)

  membership_inference(model=model,
                       query_limit=10000,
                       victim_input_shape=data_shape,
                       number_of_targets=10,
                       attacker_input_shape=data_shape,
                       synthesizer_name="copycat",
                       substitute_architecture=Cifar10Net,
                       attack_train_data=train_data,
                       attack_test_data=test_data,
                       max_epochs=50)


def main():
  args = _parse_args()

  # Deterministic, reproducible behaviour
  pl.seed_everything(args.seed)
  if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  train_dataset, test_dataset = get_dataset('cifar10', False)
  args.experiment_id = _get_next_available_dir('./lightning_logs', 'experiment', False, False)
  FLEnvironment(model_class=Cifar10Net,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                num_clients=2,
                aggregator_class=Aggregator,
                rounds=1,
                device=device,
                client_optimizer_class=torch.optim.SGD,
                clients_per_round=0,
                client_local_test_split=0.1,
                partition_method='homogeneous',
                alpha=10,
                args=args)

  # model = train(args)
  # model = Cifar10Net.load_from_checkpoint('./checkpoints/checkpoint-epoch=29-valid_acc=0.71.ckpt')
  # attack_model(model)


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
  main()
