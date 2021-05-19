from typing import Type

import os
import warnings

import numpy as np
import pytorch_lightning as pl
import torch.optim
from torch import nn
from matplotlib import pyplot as plt
from shutil import move, rmtree

from attacks import (model_extraction,
                     membership_inference_black_box,
                     model_extraction_knockoffnets,
                     model_inversion
                     )
from config import ExperimentConfig
from datasets.dataset_util import *
from torchvision.utils import make_grid
from dp_stages import Stages
from fl.aggregator import Aggregator
from fl.fl import FLEnvironment
from log import Logger
from models import Cifar10Net, MnistCNNNet, MnistFCNet, SqueezeNetDR, MnistClassifierCNN
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
                 logger: Logger = None,
                 include_val_split=True):
  if not model:
    model = architecture.load_from_checkpoint(checkpoint_path)
  model = model.to(device)

  datasets = get_dataset(dataset_name, include_val_split)
  if include_val_split:
    train_dataset, _, test_dataset = datasets
  else:
    train_dataset, test_dataset = datasets

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
    for _ in range(args.runs):
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
    fidelity = fidelity / args.runs
    msg = f"Model extraction AVERAGE fidelity: {100 * fidelity:.2f}%"
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

  if args.model_inversion:
    _set_seed()
    if train_dataset.name == 'cifar10':
      evaluator = None
    elif train_dataset.name == 'mnnist':
      evaluator = MnistClassifierCNN()
    else:
      evaluator = None

    attack_success, inverted_data = model_inversion(model=model,
                                                    loss=nn.CrossEntropyLoss(),
                                                    optimizer_class=torch.optim.SGD,
                                                    input_shape=train_dataset.input_shape,
                                                    num_classes=train_dataset.num_classes,
                                                    test_dataset=test_dataset,
                                                    batch_size=args.batch_size,
                                                    epochs=10,
                                                    evaluator=evaluator,
                                                    confidence_threshold=0.5,
                                                    logger=logger
                                                    )
    msg = f"Model Inversion AVERAGE success: {100 * attack_success:.2f}%"
    logger.log(msg) if logger else print(msg)
    attack_results['Model_Inversion'] = {'success': attack_success}

    if inverted_data is not None:
      grid = make_grid(torch.tensor(inverted_data),
                       nrow=2, padding=2, normalize=False,
                       range=None, scale_each=False, pad_value=0)
      grid = grid.cpu().numpy()
      plt.imsave(os.path.join(args.save_dir, 'inversion_result.png'), grid)

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
                        virtual_batches=args.virtual_batches,
                        device=device,
                        save_checkpoint=True
                        )

  trainer.logger.log(str(args))

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


def _train_fl_and_attack(args, model_cls, dataset_name):
  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  train_dataset, test_dataset = get_dataset(dataset_name, False)

  args.experiment_id = _get_next_available_dir(f'{args.save_dir}/lightning_logs', 'experiment', False, False)
  args.save_model_path = _get_next_available_dir(args.save_dir, 'checkpoints', True, True)

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

  attack_results = attack_model(args, device, dataset_name, model_cls,
                                model, logger=fl_simulator.logger, include_val_split=False)

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
  eps3_range = [0.001, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 2, 3]

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
      np.save(f, np.array(rng))
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
  args.stage2 = True  # using the opacus library here
  args.stage3 = False

  _train_msdp_and_attack(args, Cifar10Net, 'cifar10')


@experiment
def nonprivate_fl_on_cifar10():
  local = True
  if local:
    save_dir = results_dir = 'out/'
  else:
    save_dir = '/tmp/va4317/out/'
    os.makedirs(save_dir, exist_ok=True)
    results_dir = '/vol/bitbucket/va4317/msdp/out/'

  args = ExperimentConfig()
  args.name = "Non-Private on CIFAR10"
  args.save_dir = save_dir
  args.num_rounds = 10
  args.epochs = 10
  args.stage1 = False
  args.stage2 = False
  args.stage3 = False
  args.stage4 = False

  model_cls = Cifar10Net

  _train_fl_and_attack(args, model_cls, 'cifar10')

  # # move logs to results_dir if non-local training
  if not local:
    file_names = os.listdir(save_dir)
    for file_name in file_names:
      move(os.path.join(save_dir, file_name), results_dir)
    rmtree(save_dir)


@experiment
def msdpfl_on_cifar10():
  local = True
  if local:
    save_dir = results_dir = 'out/'
  else:
    save_dir = '/tmp/va4317/out/'
    os.makedirs(save_dir, exist_ok=True)
    results_dir = '/vol/bitbucket/va4317/msdp/out/'

  args = ExperimentConfig()
  args.name = "MSDPFL on CIFAR10"
  args.save_dir = save_dir
  args.num_rounds = 10
  args.epochs = 15
  args.eps1 = 10
  args.noise_multiplier = 0.5
  args.max_grad_norm = 6
  args.eps3 = 20
  args.max_weight_norm = 20
  args.max_weight_norm_aggregated = 20

  model_cls = Cifar10Net

  _train_fl_and_attack(args, model_cls, 'cifar10')

  # # move logs to results_dir if non-local training
  if not local:
    file_names = os.listdir(save_dir)
    for file_name in file_names:
      move(os.path.join(save_dir, file_name), results_dir)
    rmtree(save_dir)


@experiment
def fl_opacus_on_cifar10():
  local = True
  if local:
    save_dir = results_dir = 'out/'
  else:
    save_dir = '/tmp/va4317/out/'
    os.makedirs(save_dir, exist_ok=True)
    results_dir = '/vol/bitbucket/va4317/msdp/out/'

  args = ExperimentConfig()
  args.name = "FL Opacus on CIFAR10"
  args.save_dir = save_dir
  args.num_rounds = 10
  args.epochs = 15
  args.stage1 = False
  args.stage2 = True
  args.stage3 = False
  args.stage4 = False

  args.noise_multiplier = 0.75
  args.max_grad_norm = 6

  model_cls = Cifar10Net

  _train_fl_and_attack(args, model_cls, 'cifar10')

  # move logs to results_dir if non-local training
  if not local:
    file_names = os.listdir(save_dir)
    for file_name in file_names:
      move(os.path.join(save_dir, file_name), results_dir)
    rmtree(save_dir)


@experiment
def non_private_training_on_mnist():
  args = ExperimentConfig()
  args.name = "Non private training on MNIST"
  args.stage1 = False
  args.stage2 = False
  args.stage3 = False
  args.stage4 = False
  args.batch_size = 128
  args.epochs = 10
  model_cls = MnistCNNNet

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
  model_cls = MnistCNNNet

  _train_msdp_and_attack(args, model_cls, 'mnist')


@experiment
def opacus_training_on_mnist():
  args = ExperimentConfig()
  args.name = "Opacus training on MNIST"
  args.batch_size = 256
  args.epochs = 15
  args.noise_multiplier = .75
  args.max_grad_norm = 6
  args.lr = 0.02
  args.gamma = 0.7
  args.weight_decay = 5e-4
  args.momentum = 0.9
  args.stage1 = False
  args.stage2 = True
  args.stage3 = False
  model_cls = MnistCNNNet

  _train_msdp_and_attack(args, model_cls, 'mnist')


@experiment
def nonprivate_fl_on_mnist():
  args = ExperimentConfig()
  args.name = "Non-Private on MNIST"
  args.num_rounds = 10
  args.epochs = 5
  args.stage1 = False
  args.stage2 = False
  args.stage3 = False
  args.stage4 = False
  args.num_clients = 10
  args.alpha = 10
  model_cls = MnistCNNNet

  _train_fl_and_attack(args, model_cls, 'mnist')


@experiment
def msdpfl_on_mnist():
  args = ExperimentConfig()
  args.name = "MSDPFL on CIFAR10"
  args.num_rounds = 10
  args.epochs = 5
  args.eps1 = 10
  args.noise_multiplier = 0.7
  args.max_grad_norm = 6
  args.eps3 = 10
  args.max_weight_norm = 20
  args.max_weight_norm_aggregated = 20
  args.num_clients = 10
  args.alpha = 10

  model_cls = MnistCNNNet

  _train_fl_and_attack(args, model_cls, 'mnist')


@experiment
def fl_opacus_on_mnist():
  args = ExperimentConfig()
  args.name = "FL Opacus on MNIST"
  args.num_rounds = 10
  args.epochs = 5
  args.stage1 = False
  args.stage2 = True
  args.stage3 = False
  args.stage4 = False
  args.noise_multiplier = 1.1
  args.max_grad_norm = 5
  args.num_clients = 10
  args.alpha = 10
  model_cls = MnistCNNNet

  _train_fl_and_attack(args, model_cls, 'mnist')


@experiment
def msdpfl_on_cifar_client_variation():
  args = ExperimentConfig()
  args.name = f"MSDPFL on CIFAR with client variation"
  args.num_rounds = 10
  args.epochs = 15
  args.eps1 = 10
  args.noise_multiplier = 0.5
  args.max_grad_norm = 6
  args.eps3 = 20
  args.max_weight_norm = 20
  args.max_weight_norm_aggregated = 20
  args.alpha = 50
  model_cls = Cifar10Net

  clients_range = [5, 10, 20, 30, 50, 75, 100]

  ranges = {'num_clients': clients_range}

  for name, rng in ranges.items():
    print('=' * 80)
    test_acc = []
    mea_fid = []
    mia_acc = []
    for value in rng:
      args.set_value(name, value)

      model, attack_results = _train_fl_and_attack(args, model_cls, 'cifar10')
      test_acc.append(model.test_accuracy)
      mea_fid.append(attack_results['MEA']['fidelity'])
      mia_acc.append(attack_results['MIA']['accuracy'])

    with open(f'./{name}.npy', 'wb') as f:
      np.save(f, np.array(rng))
      np.save(f, np.array(test_acc))
      np.save(f, np.array(mea_fid))
      np.save(f, np.array(mia_acc))


@experiment
def non_private_training_on_dr():
  args = ExperimentConfig()
  args.name = "Non private training on DR"
  args.stage1 = False
  args.stage2 = False
  args.stage3 = False
  args.stage4 = False
  args.batch_size = 256
  args.test_batch_size = 256
  args.epochs = 60
  args.lr = 0.002
  args.membership_inference = False
  args.model_extraction = False
  model_cls = SqueezeNetDR

  _train_msdp_and_attack(args, model_cls, 'dr')


@experiment
def msdp_training_on_dr():
  args = ExperimentConfig()
  args.name = f"MSDP on DR"
  args.eps1 = 8
  args.noise_multiplier = 0.35
  args.max_grad_norm = 4.5
  args.virtual_batches = 1
  args.eps3 = 1
  args.max_weight_norm = 18
  args.stage2 = False
  args.stage3 = False
  args.batch_size = 100
  args.test_batch_size = 100
  args.epochs = 20
  args.lr = 0.002
  args.gamma = 0.7
  args.weight_decay = 5e-4
  args.momentum = 0.9
  args.lr = 0.002
  args.membership_inference = False
  args.model_extraction = False
  model_cls = SqueezeNetDR

  _train_msdp_and_attack(args, model_cls, 'dr')


@experiment
def nonprivate_fl_training_on_dr():
  args = ExperimentConfig()
  args.name = "Non-Private FL on DR"
  args.num_rounds = 60
  args.epochs = 2
  args.batch_size = 256
  args.test_batch_size = 256
  args.lr = 0.002
  args.num_clients = 10
  args.partition_method = 'homogeneous'
  args.clients_per_round = 5
  args.stage1 = False
  args.stage2 = False
  args.stage3 = False
  args.stage4 = False
  args.membership_inference = False
  args.model_extraction = False
  model_cls = SqueezeNetDR

  _train_fl_and_attack(args, model_cls, 'dr')


@experiment
def msdpfl_training_on_dr():
  args = ExperimentConfig()
  args.name = "MSDPFL on DR"
  args.num_rounds = 25
  args.epochs = 1
  args.batch_size = 64
  args.lr = 0.002
  args.stage1 = True
  args.stage2 = True
  args.stage3 = True
  args.stage4 = True
  args.eps1 = 10
  args.noise_multiplier = 0.6
  args.max_grad_norm = 4
  args.eps3 = 3
  args.max_weight_norm = 15
  args.num_clients = 10
  args.partition_method = 'homogeneous'
  args.clients_per_round = 5
  args.membership_inference = False
  args.model_extraction = False
  model_cls = SqueezeNetDR

  _train_fl_and_attack(args, model_cls, 'dr')


def run_experiments():
  experiments = [
    opacus_training_on_mnist
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

  args.membership_inference = True
  args.model_extraction = False
  # args.knockoffnet_extraction = True

  attack_model(args, device, 'mnist', architecture=model_cls,
               checkpoint_path='out_centralMSDP/MNIST/cnn/opacus_training/final.ckpt')


def _plot(data_dict, x_label, y_label, title):
  fig, ax = plt.subplots()  # (figsize=(10,10))

  if 'x_ticks' in data_dict:
    x_values = data_dict.pop('x_ticks')
  else:
    x_values = None

  max_x_range_len = 0
  for name, data in data_dict.items():
    if x_values is not None:
      ax.plot(list(range(len(x_values))), data, label=name)
    else:
      ax.plot(list(range(len(data))), data, label=name)
      max_x_range_len = max(max_x_range_len, len(data))

  if x_values is not None:
    plt.xticks(list(range(len(x_values))), x_values)
  else:
    ticks = list(range(max_x_range_len))
    values = list(range(1, max_x_range_len + 1))
    plt.xticks(ticks, values)
  ax.legend()
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.title(title)
  plt.show()


def load_and_plot_privacy_param_variation():
  # eps1 = {'name': 'eps_1', 'fp': './eps1.npy'}
  # noise_multiplier = {'name': 'noise_multiplier', 'fp': './noise_multiplier.npy'}
  # eps3 = {'name': 'eps_3', 'fp': './eps3.npy'}
  #
  # files = [eps1, noise_multiplier, eps3]

  num_c = {'name': 'num_clients', 'fp': './num_clients.npy'}
  files = [num_c]
  curve_names = ['Test accuracy', 'MEA fidelity', 'MIA accuracy']

  for data_file in files:
    data = dict()
    with open(data_file['fp'], 'rb') as f:
      data['x_ticks'] = np.load(f)
      for curve in curve_names:
        data[curve] = np.load(f)
    _plot(data, data_file['name'], 'Privacy and Utility', 'Small CNN on Cifar10')


def load_and_plot_learning_curves():
  def fetch(fs, metric_name):
    metric_data = dict()
    for f in fs:
      metric_data[f['name']] = f[metric_name]
    return metric_data

  metrics = ['val_acc']

  # msdp = {'name': 'MSDPFL', 'fp': "out/MNIST/checkpoints_2/stats.npy"}
  # opacus = {'name': 'Opacus FL', 'fp': "out/MNIST/checkpoints_4/stats.npy"}
  # non_private = {'name': 'Non-Private FL', 'fp': "out/MNIST/checkpoints_1/stats.npy"}
  # title = 'FL on MNIST'

  msdp = {'name': 'MSDPFL', 'fp': "out/CIFAR10/checkpoints_3/stats.npy"}
  opacus = {'name': 'Opacus FL', 'fp': "out/CIFAR10/checkpoints_1/stats.npy"}
  np_centralised = {'name': 'Non-Private Centralised', 'fp': "out_centralMSDP/DR/checkpoints_1/MSDPTrainer_0_plot_stats.npy"}
  np_fl = {'name': 'Non-Private FL', 'fp': "outFL/DR/checkpoints_2/stats.npy"}
  title = 'FL training on DR'

  files = [np_fl]

  for data_file in files:
    data = dict()
    with open(data_file['fp'], 'rb') as f:
      for metric in metrics:
        data[metric] = np.load(f)
        # if metric in ['train_acc', 'val_acc'] and data_file['name'] != 'Opacus':
        #   data[metric] = data[metric][1:]
    data_file.update(**data)

  for metric in metrics:
    metric_data = fetch(files, metric)
    _plot(metric_data, 'Rounds', metric, title)


if __name__ == '__main__':
  warnings.filterwarnings("ignore")
  # load_and_plot_privacy_param_variation()
  # load_and_plot_learning_curves()
  run_experiments()
  # attack_test()
