import argparse

from dataset import *
from membership_inference import membership_inference_attack
from models import *

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
    default=0,
    metavar="NOISE_MULTIPLIER",
    help="noise_multiplier (default: 0)",
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


def main():
  args = _parse_args()
  membership_inference_attack(args)


if __name__ == '__main__':
  main()
