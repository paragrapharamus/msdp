import json
from typing import Type

import os
import warnings

import pytorch_lightning as pl
from torch import nn

from attacks import model_extraction, membership_inference_black_box, model_extraction_knockoffnets
from config import ExperimentConfig
from datasets.dataset_util import *
from dp.opacus_dp import opacus_training
from dp_stages import Stages
from fl.aggregator import Aggregator
from fl.fl import FLEnvironment
from log import Logger
from models import Cifar10Net
from msdp import MSPDTrainer


def _set_seed(seed=42):
  pl.seed_everything(seed)
  if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True


def expertiment(exp):
  def decorator():
    _set_seed(42)
    print(exp.__name__)
    exp()

  return decorator


def attack_model(args: ExperimentConfig,
                 device: torch.device,
                 dataset_name: str,
                 architecture: Type[pl.LightningModule],
                 model: pl.LightningModule = None,
                 checkpoint_path: str = None,
                 logger: Logger = None):
  if not model:
    model = architecture.load_from_checkpoint(checkpoint_path)
  model = model.to(device)

  train_dataset, _, test_dataset = get_dataset(dataset_name)

  if args.membership_inference:
    _set_seed()
    membership_inference_black_box(model=model,
                                   loss=nn.CrossEntropyLoss(),
                                   optimizer_class=torch.optim.Adam,
                                   train_dataset=train_dataset,
                                   test_dataset=test_dataset,
                                   input_shape=train_dataset.input_shape,
                                   num_classes=train_dataset.num_classes,
                                   attack_train_size=5000,
                                   attack_test_size=5000,
                                   logger=logger
                                   )
  if args.model_extraction:
    _set_seed()
    fidelity = 0
    runs = 3
    for _ in range(runs):
      fidelity += model_extraction(model=model,
                                   train_dataset=train_dataset,
                                   test_dataset=test_dataset,
                                   query_limit=len(test_dataset),
                                   victim_input_shape=train_dataset.input_shape,
                                   num_classes=train_dataset.num_classes,
                                   attacker_input_shape=train_dataset.input_shape,
                                   synthesizer_name='copycat',
                                   substitute_architecture=architecture,
                                   max_epochs=25,
                                   logger=logger
                                   )
    msg = f"Model extraction AVERAGE fidelity: {100 * fidelity / runs:.2f}"
    logger.log(msg) if logger else print(msg)

  if args.knockoffnet_extraction:
    _set_seed()
    model_extraction_knockoffnets(model=model,
                                  attack_model_cls=architecture,
                                  loss=nn.CrossEntropyLoss(),
                                  optimizer_class=torch.optim.Adam,
                                  input_shape=train_dataset.input_shape,
                                  num_classes=train_dataset.num_classes,
                                  test_dataset=test_dataset,
                                  batch_size=args.batch_size,
                                  epochs=25,
                                  query_limit=9000,
                                  logger=logger
                                  )


@expertiment
def non_private_training_on_cifar10():
  args = ExperimentConfig()
  args.name = "Non private training on CIFAR10"
  args.stage1 = False
  args.stage2 = False
  args.stage3 = False
  args.stage4 = False
  args.batch_size = 128
  args.epochs = 15

  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  dataloaders = load_dataset('cifar10', args)
  model = Cifar10Net().to(device)
  optimizer = torch.optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)

  trainer = MSPDTrainer(model=model,
                        optimizer=optimizer,
                        data_loaders=dataloaders,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        device=device,
                        save_checkpoint=True
                        )
  trainer.logger.log(args)

  model = trainer.train_and_test()
  attack_model(args, device, 'cifar10', Cifar10Net, model, logger=trainer.logger)


@expertiment
def msdp_training_on_cifar10():
  args = ExperimentConfig()
  args.name = f"MSDP on CIFAR10, Stage1={args.stage1}, Stage2={args.stage2}, Stage3={args.stage3}"

  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  dataloaders = load_dataset('cifar10', args)
  model = Cifar10Net().to(device)
  optimizer = torch.optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)

  trainer = MSPDTrainer(model=model,
                        optimizer=optimizer,
                        data_loaders=dataloaders,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        device=device,
                        save_checkpoint=True
                        )

  trainer.logger.log(args)

  if args.stage1:
    trainer.attach_stage(Stages.STAGE_1,
                         {'eps': args.eps1,
                          'max_grad_norm': args.max_grad_norm})
  if args.stage2:
    trainer.attach_stage(Stages.STAGE_2,
                         {'noise_multiplier': args.noise_multiplier,
                          'max_grad_norm': args.max_grad_norm})
  if args.stage3:
    trainer.attach_stage(Stages.STAGE_3,
                         {'eps': args.eps3,
                          'max_weight_norm': args.max_weight_norm})

  model = trainer.train_and_test()

  attack_model(args, device, 'cifar10', Cifar10Net, model, logger=trainer.logger)


@expertiment
def opacus_training_on_cifar10():
  args = ExperimentConfig()
  args.name = "Opacus training on CIFAR10"
  args.noise_multiplier = 0.5

  print(args)

  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  dataloaders = load_dataset('cifar10', args)
  model = Cifar10Net().to(device)
  model = opacus_training(model, dataloaders, args)

  attack_model(args, device, 'cifar10', Cifar10Net, model)


@expertiment
def fl_simulation_on_cifar10():
  args = ExperimentConfig()
  args.name = "FL Simulation on CIFAR10"

  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  train_dataset, test_dataset = get_dataset('cifar10', False)
  args.experiment_id = _get_next_available_dir('out/lightning_logs', 'experiment', False, False)

  args.save_model_path = _get_next_available_dir('out/', 'checkpoints', True, True)
  fl_simulator = FLEnvironment(model_class=Cifar10Net,
                               train_dataset=train_dataset,
                               test_dataset=test_dataset,
                               num_clients=args.num_clients,
                               aggregator_class=Aggregator,
                               rounds=args.num_rounds,
                               device=device,
                               client_optimizer_class=torch.optim.SGD,
                               clients_per_round=args.clients_per_round,
                               client_local_test_split=args.client_local_test_split,
                               partition_method=args.partition_method,
                               alpha=args.alpha,
                               args=args)

  model = fl_simulator.get_model()

  attack_model(args, device, 'cifar10', Cifar10Net, model, logger=fl_simulator.logger)


def run_experiments():
  experiments = [
    msdp_training_on_cifar10,
    # opacus_training_on_cifar10
  ]

  for exp in experiments:
    exp()


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


def attack_test():
  _set_seed()
  args = ExperimentConfig()

  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  args.membership_inference = False
  args.model_extraction = False

  attack_model(args, device, 'cifar10', architecture=Cifar10Net, model=Cifar10Net(),)#
               # checkpoint_path='out/checkpoints_1/final.ckpt')


if __name__ == '__main__':
  warnings.filterwarnings("ignore")
  run_experiments()
  # attack_test()
