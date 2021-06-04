from typing import Type

import warnings

import pytorch_lightning as pl
import torch.optim
from torch import nn
from matplotlib import pyplot as plt

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
from models import Cifar10Net, MnistCNNNet, MnistFCNet, SqueezeNetDR, MnistClassifierCNN, Cifar10Classifier, Cifar10ResNet
from msdp import MSPDTrainer
from utils import *


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
                                   test_dataset=test_dataset,
                                   query_limit=250,
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
      evaluator = Cifar10Classifier()
    elif train_dataset.name == 'mnist':
      evaluator = MnistClassifierCNN()
    else:
      evaluator = None

    attack_success, inverted_data = model_inversion(model=model,
                                                    loss=nn.CrossEntropyLoss(),
                                                    optimizer_class=torch.optim.Adam,
                                                    input_shape=train_dataset.input_shape,
                                                    num_classes=train_dataset.num_classes,
                                                    test_dataset=test_dataset,
                                                    batch_size=args.batch_size,
                                                    epochs=20,
                                                    evaluator=evaluator,
                                                    confidence_threshold=0.5,
                                                    logger=logger
                                                    )
    msg = f"Model Inversion AVERAGE success: {100 * attack_success:.2f}%"
    logger.log(msg) if logger else print(msg)
    attack_results['Model_Inversion'] = {'success': attack_success}

    if inverted_data is not None:
      grid = make_grid(torch.tensor(inverted_data),
                       nrow=5, padding=2, normalize=True,
                       value_range=(0, 255), scale_each=False, pad_value=0)
      grid = np.transpose(grid.cpu().numpy(), (1, 2, 0))
      plt.imsave(os.path.join(args.save_dir, f'inversion_result_{args.name}.png'), grid)

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

  args.experiment_id = get_next_available_dir(f'{args.save_dir}/lightning_logs', 'experiment', False, False)
  args.save_model_path = get_next_available_dir(args.save_dir, 'checkpoints', True, True)

  fl_simulator = FLEnvironment(model_class=model_cls,
                               train_dataset=train_dataset,
                               test_dataset=test_dataset,
                               num_clients=args.num_clients,
                               aggregator_class=Aggregator,
                               rounds=args.num_rounds,
                               device=device,
                               client_optimizer_class=args.client_optimizer_class,
                               clients_per_round=args.clients_per_round,
                               client_local_test_split=args.client_local_test_split,
                               partition_method=args.partition_method,
                               alpha=args.alpha,
                               args=args)

  if args.cross_client_validation:
    fl_simulator.cross_client_validation(os.path.join(args.save_model_path, 'val_heatmap.png'))

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
  args.name = f"MSDP on CIFAR10"
  args.eps1 = 2.5
  args.noise_multiplier = 1
  args.max_grad_norm = 1.75
  args.virtual_batches = 1
  args.eps3 = 1
  args.max_weight_norm = 15
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
  args.name = f"MSDP on CIFAR10"
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
  args.noise_multiplier = 2
  args.max_grad_norm = 2
  args.lr = 0.005
  args.gamma = 0.7
  args.weight_decay = 5e-4
  args.momentum = 0.9
  args.stage1 = False
  args.stage2 = True  # using the opacus library here
  args.stage3 = False

  _train_msdp_and_attack(args, Cifar10Net, 'cifar10')


@experiment
def nonprivate_fl_on_cifar10():
  args = ExperimentConfig()
  args.name = "Non-Private on CIFAR10"
  args.num_rounds = 15
  args.epochs = 5
  args.stage1 = False
  args.stage2 = False
  args.stage3 = False
  args.stage4 = False
  args.clients_per_round = 3
  args.alpha = 0.1
  model_cls = Cifar10Net

  _train_fl_and_attack(args, model_cls, 'cifar10')


@experiment
def msdpfl_on_cifar10():
  args = ExperimentConfig()
  args.name = "MSDPFL on CIFAR10"
  args.num_rounds = 10
  args.epochs = 10
  args.eps1 = 5
  args.noise_multiplier = 1.5
  args.max_grad_norm = 2
  args.eps3 = 15
  args.max_weight_norm = 12
  args.clients_per_round = 5
  args.alpha = 50

  model_cls = Cifar10Net

  _train_fl_and_attack(args, model_cls, 'cifar10')


@experiment
def fl_opacus_on_cifar10():
  args = ExperimentConfig()
  args.name = "FL Opacus on CIFAR10"
  args.num_rounds = 15
  args.epochs = 5
  args.stage1 = False
  args.stage2 = True
  args.stage3 = False
  args.stage4 = False
  args.noise_multiplier = 2
  args.max_grad_norm = 2
  args.clients_per_round = 5
  args.alpha = 0.3

  model_cls = Cifar10Net

  _train_fl_and_attack(args, model_cls, 'cifar10')


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
  model_cls = MnistFCNet

  _train_msdp_and_attack(args, model_cls, 'mnist')


@experiment
def msdp_training_on_mnist():
  args = ExperimentConfig()
  args.name = f"MSDP on MNIST"
  args.eps1 = 10
  args.stage1 = False
  args.noise_multiplier = 0.6
  args.max_grad_norm = 2
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
  args.max_grad_norm = 2
  args.lr = 0.02
  args.gamma = 0.7
  args.weight_decay = 5e-4
  args.momentum = 0.9
  args.stage1 = False
  args.stage2 = True
  args.stage3 = False
  model_cls = MnistFCNet

  _train_msdp_and_attack(args, model_cls, 'mnist')


@experiment
def nonprivate_fl_on_mnist():
  args = ExperimentConfig()
  args.name = "Non-Private on MNIST"
  args.num_rounds = 20
  args.epochs = 3
  args.stage1 = False
  args.stage2 = False
  args.stage3 = False
  args.stage4 = False
  args.num_clients = 10
  args.clients_per_round = 5
  args.alpha = 0.2
  model_cls = MnistCNNNet

  _train_fl_and_attack(args, model_cls, 'mnist')


@experiment
def msdpfl_on_mnist():
  args = ExperimentConfig()
  args.name = "MSDPFL on MNIST"
  args.num_rounds = 10
  args.epochs = 5
  args.eps1 = 10
  args.noise_multiplier = 0.7
  args.max_grad_norm = 6
  args.eps3 = 10
  args.max_weight_norm = 11
  args.num_clients = 10
  args.clients_per_round = 5
  args.alpha = 0.2

  model_cls = MnistCNNNet

  _train_fl_and_attack(args, model_cls, 'mnist')


@experiment
def fl_opacus_on_mnist():
  args = ExperimentConfig()
  args.name = "FL Opacus on MNIST"
  args.num_rounds = 20
  args.epochs = 3
  args.stage1 = False
  args.stage2 = True
  args.stage3 = False
  args.stage4 = False
  args.noise_multiplier = 2.5
  args.max_grad_norm = 2
  args.num_clients = 10
  args.clients_per_round = 5
  args.alpha = 0.2
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
  args.eps1 = 5
  args.noise_multiplier = 0.4
  args.max_grad_norm = 5
  args.virtual_batches = 10
  args.eps3 = 1
  args.max_weight_norm = 18
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
def opacus_training_on_dr():
  args = ExperimentConfig()
  args.name = f"Opacus on DR"
  args.noise_multiplier = 2
  args.virtual_batches = 10
  args.stage1 = False
  args.stage2 = True
  args.stage3 = False
  args.batch_size = 100
  args.test_batch_size = 100
  args.epochs = 60
  args.lr = 0.002
  args.gamma = 0.7
  args.weight_decay = 5e-4
  args.momentum = 0.9
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


@experiment
def opacus_fl_training_on_dr():
  args = ExperimentConfig()
  args.name = "OpacusFL on DR"
  args.num_rounds = 60
  args.epochs = 5
  args.batch_size = 130
  args.virtual_batches = 1
  args.lr = 0.002
  args.stage1 = False
  args.stage2 = True
  args.stage3 = False
  args.stage4 = False
  args.noise_multiplier = 2
  args.max_grad_norm = 2
  args.num_clients = 10
  args.partition_method = 'homogeneous'
  args.clients_per_round = 5
  args.membership_inference = False
  args.model_extraction = False
  model_cls = SqueezeNetDR

  _train_fl_and_attack(args, model_cls, 'dr')


def run_experiments():
  experiments = [
    # nonprivate_fl_on_mnist,
    # fl_opacus_on_mnist,
    msdpfl_on_cifar10
  ]

  for exp in experiments:
    exp()


def attack_test():
  _set_seed()
  args = ExperimentConfig()
  model_cls = Cifar10Net
  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  args.membership_inference = False
  args.model_extraction = False
  args.model_inversion = True
  # args.knockoffnet_extraction = True

  args.name = 'msdpfl_cifar-low'
  attack_model(args, device, 'cifar10', architecture=model_cls,
               checkpoint_path='outFL/CIFAR10/low_eps/msdpfl/final.ckpt')


if __name__ == '__main__':
  warnings.filterwarnings("ignore")
  # load_and_plot_privacy_param_variation()
  # load_and_plot_learning_curves()
  # load_and_plot_dr()
  run_experiments()
  # attack_test()
