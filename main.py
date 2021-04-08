import argparse
import torch
from datasets.dataset import *
from membership_inference import membership_inference_attack

if torch.cuda.is_available():
  torch.backends.cudnn.deterministic = True
RAND_SEED = 42
np.random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)


def _parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
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
    default=100,
    metavar="N",
    help="input batch size for testing (default: 100)",
  )
  parser.add_argument(
    "--epochs",
    type=int,
    default=20,
    metavar="N",
    help="number of epochs to train (default: 20)",
  )
  parser.add_argument(
    "--lr",
    type=float,
    default=0.15,
    metavar="LR",
    help="learning rate (default: 0.15)",
  )

  parser.add_argument(
    "--max_norm",
    type=float,
    default=10,
    metavar="MAX_NORM",
    help="l2 clipping norm (default: 10.0)",
  )

  parser.add_argument(
    "--noise_multiplier",
    type=float,
    default=1.1,
    metavar="NOISE_MULTIPLIER",
    help="noise_multiplier (default: 1.0)",
  )

  parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="MOMENTUM",
    help="momentum"
  )

  parser.add_argument(
    "--gamma",
    type=float,
    default=0.7,
    metavar="M",
    help="Learning rate step gamma (default: 0.7)",
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
    "--log-interval",
    type=int,
    default=100,
    metavar="N",
    help="how many batches to wait before logging training status",
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
    default='./model.pth',
    help="the path of the trained model"
  )

  args = parser.parse_args()
  return args

from models import Cifar10Net
from dp_optim import DPSGD
from msdp import MSPDTrainer

def train(args):
  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  train_kwargs = {"batch_size": args.batch_size}
  test_kwargs = {"batch_size": args.test_batch_size}
  if use_cuda:
    cuda_kwargs = {"num_workers": 1, "pin_memory": True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

  dataloaders = load_cifar10(test_kwargs, train_kwargs)

  model = Cifar10Net().to(device)
  optimizer = DPSGD(
    l2_norm_clip=args.max_norm,
    noise_multiplier=args.noise_multiplier,
    minibatch_size=args.batch_size,
    microbatch_size=args.minibatch_size,
    params=model.parameters(),
    lr=args.lr)

  trainer = MSPDTrainer(model, optimizer, dataloaders, ['STAGE_1', 'STAGE_3'], 1, args.batch_size,
                        args.minibatch_size, device, './model.pth')

  trainer.train_and_test()

def main():
  args = _parse_args()
  # membership_inference_attack(args)
  train(args)


if __name__ == '__main__':
  main()
