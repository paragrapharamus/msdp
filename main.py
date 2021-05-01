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
from models import Cifar10Net
from msdp import MSPDTrainer


def _set_seed(seed=42):
  pl.seed_everything(seed)
  if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True


def expertiment(exp):
  def decorator():
    _set_seed(42)
    exp()

  return decorator


def attack_model(args: ExperimentConfig,
                 device: torch.device,
                 dataset_name: str,
                 model=None,
                 checkpoint_path=None,
                 architecture=None):
  if not model:
    model = architecture.load_from_checkpoint(checkpoint_path).to(device)

  train_dataset, test_dataset = get_dataset(dataset_name, validation_dataset=False)

  if args.membership_inference:
    membership_inference_black_box(model=model,
                                   loss=nn.CrossEntropyLoss(),
                                   optimizer_class=torch.optim.Adam,
                                   train_dataset=train_dataset,
                                   test_dataset=test_dataset,
                                   input_shape=train_dataset.input_shape,
                                   num_classes=train_dataset.num_classes,
                                   attack_train_ratio=0.5,
                                   attack_test_ratio=0.5,
                                   )
  if args.model_extraction:
    model_extraction(model=model,
                     train_dataset=train_dataset,
                     test_dataset=test_dataset,
                     query_limit=len(test_dataset),
                     victim_input_shape=train_dataset.input_shape,
                     num_classes=train_dataset.num_classes,
                     attacker_input_shape=train_dataset.input_shape,
                     synthesizer_name='copycat',
                     substitute_architecture=architecture,
                     max_epochs=25
                     )

  if args.knockoffnet_extraction:
    model_extraction_knockoffnets(model=model,
                                  attack_model_cls=architecture,
                                  loss=nn.CrossEntropyLoss(),
                                  optimizer_class=torch.optim.Adam,
                                  input_shape=train_dataset.input_shape,
                                  num_classes=10,
                                  test_dataset=test_dataset,
                                  batch_size=args.batch_size,
                                  epochs=5,
                                  query_limit=9000,
                                  )


@expertiment
def non_private_training_on_cifar10():
  args = ExperimentConfig()
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
  model = trainer.train_and_test()

  attack_model(args, device, 'cifar10', model)


@expertiment
def msdp_training_on_cifar10(stage_1=True, stage_2=True, stage_3=True):
  args = ExperimentConfig()
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
  if stage_1:
    trainer.attach_stage(Stages.STAGE_1,
                         {'eps': args.eps1,
                          'max_grad_norm': args.max_grad_norm})
  if stage_2:
    trainer.attach_stage(Stages.STAGE_2,
                         {'noise_multiplier': args.noise_multiplier,
                          'max_grad_norm': args.max_grad_norm})
  if stage_3:
    trainer.attach_stage(Stages.STAGE_3,
                         {'eps': args.eps3,
                          'max_weight_norm': args.max_weight_norm})

  model = trainer.train_and_test()
  attack_model(args, device, 'cifar10', model)


@expertiment
def opacus_training_on_cifar10():
  args = ExperimentConfig()

  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  dataloaders = load_dataset('cifar10', args)
  model = Cifar10Net().to(device)
  opacus_training(model, dataloaders, args)


@expertiment
def fl_simulation_on_cifar10():
  args = ExperimentConfig()

  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  train_dataset, test_dataset = get_dataset('cifar10', False)
  args.experiment_id = _get_next_available_dir('out/lightning_logs', 'experiment', False, False)
  fl_simulator = FLEnvironment(model_class=Cifar10Net,
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

  model = fl_simulator.get_model()

  attack_model(args, device, 'cifar10', model)


def run_experiments():
  experiments = [
    non_private_training_on_cifar10
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


if __name__ == '__main__':
  warnings.filterwarnings("ignore")
  run_experiments()
