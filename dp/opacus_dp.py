import argparse
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torch.utils.tensorboard as tensorboard
import torchvision.models as models
import torchvision.transforms as transforms
from opacus import PrivacyEngine
from opacus.utils import stats
from opacus.utils.module_modification import convert_batchnorm_modules
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from models import Cifar10Net


class Config:
  def __init__(self):
    self.log_dir = ''
    self.data_root = '../data'
    self.checkpoint_file = './opacus_model.pth'
    self.device = 'cuda'
    self.seed = 42
    self.resume = ""

    self.disable_dp = False
    self.delta = 1e-5
    self.max_per_sample_grad_norm = 2
    self.sigma = 0.3
    self.print_freq = 10
    self.weight_decay = 5e-4
    self.momentum = 0.9
    self.lr = 0.01
    self.n_accumulation_steps = 1
    self.batch_size = 128
    self.batch_size_test = 500
    self.start_epoch = 1
    self.epochs = 20


def _save_checkpoint(state, is_best, filename="checkpoint.pth"):
  torch.save(state, filename)


def _accuracy(preds, labels):
  return (preds == labels).mean()


def _train(args, model, train_loader, optimizer, epoch, device):
  model.train()
  criterion = nn.CrossEntropyLoss()

  losses = []
  top1_acc = []

  for i, (images, target) in enumerate(tqdm(train_loader)):
    # print(f"batch_size: {len(images)}")
    images = images.to(device)
    target = target.to(device)

    # compute output
    output = model(images)
    loss = criterion(output, target)
    preds = np.argmax(output.detach().cpu().numpy(), axis=1)
    labels = target.detach().cpu().numpy()

    # measure accuracy and record loss
    acc1 = _accuracy(preds, labels)

    losses.append(loss.item())
    top1_acc.append(acc1)
    stats.update(stats.StatType.TRAIN, acc1=acc1)

    # compute gradient and do SGD step
    loss.backward()

    # make sure we take a step after processing the last mini-batch in the
    # epoch to ensure we start the next epoch with a clean state
    if ((i + 1) % args.n_accumulation_steps == 0) or ((i + 1) == len(train_loader)):
      optimizer.step()
      optimizer.zero_grad()
    else:
      optimizer.virtual_step()

    if i % args.print_freq == 0:
      if not args.disable_dp:
        epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(
          args.delta
        )
        print(
          f"\tTrain Epoch: {epoch} \t"
          f"Loss: {np.mean(losses):.6f} "
          f"Acc@1: {np.mean(top1_acc):.6f} "
          f"(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha}"
        )
      else:
        print(
          f"\tTrain Epoch: {epoch} \t"
          f"Loss: {np.mean(losses):.6f} "
          f"Acc@1: {np.mean(top1_acc):.6f} "
        )

  return np.array(losses).mean(), np.array(top1_acc).mean()


def _test(type, model, test_loader, device):
  model.eval()
  criterion = nn.CrossEntropyLoss()
  losses = []
  top1_acc = []

  with torch.no_grad():
    for images, target in tqdm(test_loader):
      images = images.to(device)
      target = target.to(device)

      output = model(images)
      loss = criterion(output, target)
      preds = np.argmax(output.detach().cpu().numpy(), axis=1)
      labels = target.detach().cpu().numpy()
      acc1 = _accuracy(preds, labels)

      losses.append(loss.item())
      top1_acc.append(acc1)

  top1_avg = np.mean(top1_acc)
  stats.update(stats.StatType.TEST, acc1=top1_avg)

  print(f"\t{type} set:" f"Loss: {np.mean(losses):.6f} " f"Acc@1: {top1_avg :.6f} ")
  return np.mean(top1_acc)


def opacus_training(model, dataloaders, global_args):
  args = Config()
  args.epochs = global_args.epochs
  args.batch_size = global_args.batch_size
  args.max_per_sample_grad_norm = global_args.max_grad_norm
  args.sigma = global_args.noise_multiplier

  # The following few lines, enable stats gathering about the run
  # 1. where the stats should be logged
  stats.set_global_summary_writer(
    tensorboard.SummaryWriter(os.path.join("/tmp/stat", args.log_dir))
  )
  # 2. enable stats
  stats.add(
    # stats about gradient norms aggregated for all layers
    stats.Stat(stats.StatType.GRAD, "AllLayers", frequency=0.1),
    # stats about gradient norms per layer
    stats.Stat(stats.StatType.GRAD, "PerLayer", frequency=0.1),
    # stats about clipping
    stats.Stat(stats.StatType.GRAD, "ClippingStats", frequency=0.1),
    # stats on training accuracy
    stats.Stat(stats.StatType.TRAIN, "accuracy", frequency=0.01),
    # stats on validation accuracy
    stats.Stat(stats.StatType.TEST, "accuracy"),
  )

  # The following lines enable stat gathering for the clipping process
  # and set a default of per layer clipping for the Privacy Engine
  clipping = {"clip_per_layer": False, "enable_stat": True}

  best_acc1 = 0
  device = torch.device(args.device)
  # model = convert_batchnorm_modules(model)

  train_loader, val_loader, test_loader = dataloaders
  model = model.to(device)
  optimizer = optim.SGD(model.parameters(),
                        lr=args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay,
                        )

  privacy_engine = PrivacyEngine(
    model,
    sample_rate=args.n_accumulation_steps * args.batch_size / (0.9 * len(train_loader.dataset)),
    alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
    noise_multiplier=args.sigma,
    max_grad_norm=args.max_per_sample_grad_norm,
    secure_rng=False,
    **clipping,
  )
  privacy_engine.attach(optimizer)

  training_losses, training_accuracies = [], []
  validation_accuracies = []
  for epoch in range(args.start_epoch, args.epochs + 1):
    tr_loss, tr_acc = _train(args, model, train_loader, optimizer, epoch, device)
    top1_acc = _test('Validation', model, val_loader, device)
    training_losses.append(tr_loss)
    training_accuracies.append(tr_acc)
    validation_accuracies.append(top1_acc)

    # remember best acc@1 and save checkpoint
    is_best = top1_acc > best_acc1
    best_acc1 = max(top1_acc, best_acc1)

    _save_checkpoint(
      {
        "epoch": epoch + 1,
        "arch": "Cifar10Net",
        "state_dict": model.state_dict(),
        "best_acc1": best_acc1,
        "optimizer": optimizer.state_dict(),
      },
      is_best,
      filename=args.checkpoint_file,
    )

  fp = './opacus_training_stats'
  np.save(fp, np.array(training_losses))
  np.save(fp, np.array(training_accuracies))
  np.save(fp, np.array(validation_accuracies))

  top1_acc = _test('Test', model, test_loader, device)
  print(f"Test set accuracy: {top1_acc:.2f}")
  return model
