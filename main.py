from typing import Type

import os
import warnings

import numpy as np
import pytorch_lightning as pl
from torch import nn
from matplotlib import pyplot as plt

from attacks import model_extraction, membership_inference_black_box, model_extraction_knockoffnets
from config import ExperimentConfig
from datasets.dataset_util import *
from dp.opacus_dp import opacus_training
from dp_stages import Stages
from fl.aggregator import Aggregator
from fl.fl import FLEnvironment
from log import Logger
from models import Cifar10Net, MnistCNNNet, MnistFCNet
from msdp import MSPDTrainer


def _set_seed(seed=42):
  pl.seed_everything(seed)
  if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True


def experiment(exp):
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

  attack_results = dict()

  if args.membership_inference:
    _set_seed()
    acc, p, r = membership_inference_black_box(model=model,
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
    attack_results['MIA'] = {'accuracy': acc, 'precision': p, 'recall': r}

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
    fidelity = fidelity / runs
    msg = f"Model extraction AVERAGE fidelity: {100 * fidelity:.2f}"
    logger.log(msg) if logger else print(msg)

    attack_results['MEA'] = {'fidelity': fidelity}

  if args.knockoffnet_extraction:
    _set_seed()
    fidelity = model_extraction_knockoffnets(model=model,
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
    attack_results['MEA_KnockOffNet'] = {'fidelity': fidelity}

  return attack_results


def _train_msdp_and_attack(args, model_cls, dataset_name):
  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  dataloaders = load_dataset(dataset_name, args)
  model = model_cls().to(device)
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
  attack_results = attack_model(args, device, dataset_name, model_cls, model, logger=trainer.logger)

  return model, attack_results


@experiment
def non_private_training_on_cifar10():
  args = ExperimentConfig()
  args.name = "Non private training on CIFAR10"
  args.stage1 = False
  args.stage2 = False
  args.stage3 = False
  args.stage4 = False
  args.batch_size = 64
  args.epochs = 25
  model_cls = Cifar10Net

  _train_msdp_and_attack(args, model_cls, 'cifar10')


@experiment
def msdp_training_on_cifar10():
  args = ExperimentConfig()
  args.name = f"MSDP on CIFAR10, Stage1={args.stage1}, Stage2={args.stage2}, Stage3={args.stage3}"
  args.eps1 = 10
  args.noise_multiplier = 0.3
  args.max_grad_norm = 5
  args.virtual_batches = 1
  args.eps3 = 1
  args.max_weight_norm = 20
  args.batch_size = 256
  args.test_batch_size = 1000
  args.epochs = 25
  args.lr = 0.02
  args.gamma = 0.7
  args.weight_decay = 5e-4
  args.momentum = 0.9
  model_cls = Cifar10Net

  _train_msdp_and_attack(args, model_cls, 'cifar10')


@experiment
def msdp_stage_effect_on_cifar10():
  def _reset_privacy_params_to_default(args):
    args.eps1 = 10
    args.noise_multiplier = 0.3
    args.eps3 = 1

  args = ExperimentConfig()
  args.name = f"MSDP on CIFAR10, Stage1={args.stage1}, Stage2={args.stage2}, Stage3={args.stage3}"
  args.eps1 = 10
  args.noise_multiplier = 0.3
  args.max_grad_norm = 5
  args.virtual_batches = 1
  args.eps3 = 1
  args.max_weight_norm = 20
  args.batch_size = 256
  args.test_batch_size = 1000
  args.epochs = 25
  args.lr = 0.02
  args.gamma = 0.7
  args.weight_decay = 5e-4
  args.momentum = 0.9
  model_cls = Cifar10Net

  eps1_range = [0.5, 2, 4, 8, 10, 15, 20]
  noise_multiplier_range = [0.1, 0.3, 0.5, 0.7, 0.8, 1]
  eps3_range = [0.25, 0.5, 0.75, 1, 2, 3]

  ranges = {'eps1': eps1_range, 'noise_multiplier': noise_multiplier_range, 'eps3': eps3_range}

  for name, rng in ranges.items():
    print('=' * 80)
    test_acc = []
    mea_fid = []
    mia_acc = []
    for value in rng:
      _reset_privacy_params_to_default(args)
      args.set_value(name, value)

      model, attack_results = _train_msdp_and_attack(args, model_cls, 'cifar10')
      test_acc.append(model.test_accuracy)
      mea_fid.append(attack_results['MEA']['fidelity'])
      mia_acc.append(attack_results['MIA']['accuracy'])

    with open(f'./{name}.npy', 'wb') as f:
      np.save(f, np.array(test_acc))
      np.save(f, np.array(mea_fid))
      np.save(f, np.array(mia_acc))


@experiment
def opacus_training_on_cifar10():
  args = ExperimentConfig()
  args.name = "Opacus training on CIFAR10"
  args.batch_size = 256
  args.epochs = 25
  args.noise_multiplier = 0.5
  args.max_grad_norm = 6
  args.lr = 0.02
  args.gamma = 0.7
  args.weight_decay = 5e-4
  args.momentum = 0.9
  args.stage1 = False
  args.stage2 = False  # using the opacus library here
  args.stage3 = False

  print(args)

  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  dataloaders = load_dataset('cifar10', args)
  model = Cifar10Net().to(device)
  model = opacus_training(model, dataloaders, args)

  attack_model(args, device, 'cifar10', Cifar10Net, model)


@experiment
def fl_simulation_on_cifar10():
  args = ExperimentConfig()
  args.name = "FL Simulation on CIFAR10"
  model_cls = Cifar10Net

  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  train_dataset, test_dataset = get_dataset('cifar10', False)
  args.experiment_id = _get_next_available_dir('out/lightning_logs', 'experiment', False, False)

  args.save_model_path = _get_next_available_dir('out/', 'checkpoints', True, True)
  fl_simulator = FLEnvironment(model_class=model_cls,
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

  attack_model(args, device, 'cifar10', model_cls, model, logger=fl_simulator.logger)


@experiment
def non_private_training_on_mnist():
  args = ExperimentConfig()
  args.name = "Non private training on MNIST"
  args.stage1 = False
  args.stage2 = False
  args.stage3 = False
  args.stage4 = False
  args.batch_size = 64
  args.epochs = 10
  model_cls = MnistFCNet

  _train_msdp_and_attack(args, model_cls, 'mnist')


@experiment
def msdp_training_on_mnist():
  args = ExperimentConfig()
  args.name = f"MSDP on MNIST, Stage1={args.stage1}, Stage2={args.stage2}, Stage3={args.stage3}"
  args.eps1 = 25
  args.noise_multiplier = 0.6
  args.max_grad_norm = 6
  args.virtual_batches = 1
  args.eps3 = 2
  args.max_weight_norm = 20
  args.batch_size = 256
  args.test_batch_size = 1000
  args.epochs = 15
  args.lr = 0.02
  args.gamma = 0.7
  args.weight_decay = 5e-4
  args.momentum = 0.9
  model_cls = MnistFCNet

  _train_msdp_and_attack(args, model_cls, 'mnist')


@experiment
def opacus_training_on_mnist():
  args = ExperimentConfig()
  args.name = "Opacus training on MNIST"
  args.batch_size = 256
  args.epochs = 15
  args.noise_multiplier = 0.8
  args.max_grad_norm = 6
  args.lr = 0.02
  args.gamma = 0.7
  args.weight_decay = 5e-4
  args.momentum = 0.9
  args.stage1 = False
  args.stage2 = False  # using the opacus library here
  args.stage3 = False
  model_cls = MnistFCNet

  print(args)

  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  dataloaders = load_dataset('mnist', args)
  model = model_cls().to(device)
  model = opacus_training(model, dataloaders, args)

  attack_model(args, device, 'mnist', model_cls, model)


@experiment
def fl_simulation_on_mnist():
  args = ExperimentConfig()
  args.name = "FL Simulation on MNIST"
  model_cls = MnistCNNNet

  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  train_dataset, test_dataset = get_dataset('mnist', False)
  args.experiment_id = _get_next_available_dir('out/lightning_logs', 'experiment', False, False)

  args.save_model_path = _get_next_available_dir('out/', 'checkpoints', True, True)
  fl_simulator = FLEnvironment(model_class=model_cls,
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

  attack_model(args, device, 'mnist', model_cls, model, logger=fl_simulator.logger)


def run_experiments():
  experiments = [
    # non_private_training_on_mnist,
    msdp_stage_effect_on_cifar10,
    # opacus_training_on_mnist,
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
  model_cls = MnistCNNNet

  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  args.membership_inference = False
  args.model_extraction = True
  # args.knockoffnet_extraction = True

  attack_model(args, device, 'mnist', architecture=model_cls,  # model=model_cls(), )  #
               checkpoint_path='out/opacus_training/opacus_model.pth')


def _plot(data_dict, y_label, title):
  fig, ax = plt.subplots()  # (figsize=(10,10))

  for name, data in data_dict.items():
    ax.plot(data, label=name)

  ax.legend()
  plt.xlabel('Epoch')
  plt.ylabel(y_label)
  plt.title(title)
  plt.show()


def load_and_plot():
  def fetch(fs, metric_name):
    metric_data = dict()
    for f in fs:
      metric_data[f['name']] = f[metric_name]
    return metric_data

  metrics = ['train_loss', 'train_acc', 'val_acc']

  msdp = {'name': 'MSDP', 'fp': "out/MNIST/fc/checkpoints_6/MSDPTrainer_0_plot_stats.npy"}
  opacus = {'name': 'Opacus', 'fp': "out/MNIST/fc/opacus_training/opacus_training_stats.npy"}
  non_private = {'name': 'Non-Private', 'fp': "out/MNIST/fc/checkpoints_1/MSDPTrainer_0_plot_stats.npy"}
  title = 'MLP on MNIST'

  # msdp = {'name': 'MSDP', 'fp': "out/0_slurm_1/checkpoints_2/MSDPTrainer_0_plot_stats.npy"}
  # non_private = {'name': 'Non-Private', 'fp': "out/0_slurm_1/checkpoints_1/MSDPTrainer_0_plot_stats.npy"}
  # opacus = {'name': 'Opacus', 'fp': "out/0_slurm_1/opacus_training/opacus_training_stats.npy"}
  # title = 'ResNet-18 on CIFAR10'

  files = [msdp, opacus, non_private]

  for data_file in files:
    data = dict()
    with open(data_file['fp'], 'rb') as f:
      for metric in metrics:
        data[metric] = np.load(f)
        if metric in ['train_acc', 'val_acc'] and data_file['name'] != 'Opacus':
          data[metric] = data[metric][1:]
    data_file.update(**data)

  for metric in metrics:
    metric_data = fetch(files, metric)
    _plot(metric_data, metric, title)


if __name__ == '__main__':
  warnings.filterwarnings("ignore")
  # load_and_plot()
  run_experiments()
  # attack_test()
