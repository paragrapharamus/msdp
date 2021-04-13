import os
import shutil
import sys
import types

from typing import List, Union, Optional

import numpy as np
import torch
import torch.nn.functional as F
from opacus import PerSampleGradientClipper
from opacus.utils import clipping
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from log import Logger
from enum import Enum


class Stages(Enum):
  STAGE_1 = 1
  ''' Input Perturbation '''
  STAGE_2 = 2
  ''' Gradient Perturbation  '''
  STAGE_3 = 3
  ''' Output (weights) Perturbation '''


class MSPDTrainer:
  """
  Multistage differentially private trainer
  """
  STATS_FMT = "[{:>5s}] loss: {:+.4f}, acc: {:.4f}"

  def __init__(self, model: nn.Module,
               optimizer: torch.optim,
               data_loaders: DataLoader,
               epochs: int,
               batch_size: int,
               device: torch.device,
               stages: List[Stages],
               max_norm: Union[float, List[float]],
               noise_multiplier: float,
               epsilon: Optional[float],
               logger: Optional[Logger] = None):

    self.model = model
    self.data_loaders = data_loaders
    if len(data_loaders) == 2:
      self.train_loader, self.test_loader = data_loaders
    elif len(data_loaders) == 3:
      self.train_loader, self.val_loader, self.test_loader = data_loaders
    else:
      raise Exception("Invalid number of loaders")
    self.stages = stages
    self.epochs = epochs
    self.batch_size = batch_size
    self.device = device
    self.max_norm = max_norm
    self.noise_multiplier = noise_multiplier
    self.epsilon = epsilon
    self.steps = 0
    self.logger = logger
    if logger is None:
      self.logger = Logger([sys.stdout, './msdp.log'])

    # For vectorized per-example gradient clipping
    self._wrap_optimizer(optimizer)

  @staticmethod
  def accuracy(preds: np.ndarray, labels: np.ndarray):
    return (preds == labels).mean()

  def train(self):
    # Inject noise to data
    if Stages.STAGE_1 in self.stages:
      self._stage_1_noise()

    best_acc = 0
    for epoch in range(self.epochs):
      prog_bar = tqdm(total=len(self.train_loader) * self.batch_size, file=sys.stdout)
      prog_bar.set_description("Epoch %d/%d" % (epoch + 1, self.epochs))
      losses = []
      accuracies = []

      # Model training
      self.model.train()
      for batch_idx, (data, targets) in enumerate(self.train_loader):
        data, targets = data.to(self.device), targets.to(self.device)

        output = self.model(data)
        loss = F.cross_entropy(output, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        losses.append(loss.item())
        accuracies.append(self.accuracy(torch.argmax(output, 1).detach().cpu().numpy(), targets.detach().cpu().numpy()))
        prog_bar.update(len(data))

      # Epoch statistics.
      epoch_loss = sum(losses) / len(losses)
      epoch_acc = sum(accuracies) / len(accuracies)
      prog_bar.close()
      self.logger.log(f"Training epoch [{epoch + 1}/{self.epochs}] loss: {epoch_loss:.6f}, acc: {epoch_acc:.4f}")

      # Validation step
      acc = self.evaluate(self.val_loader)
      self.logger.log(f"Validation Accuracy: {acc:.4f}\n")
      self.save_checkpoint(self.model.state_dict, acc > best_acc, epoch + 1)
      best_acc = max(best_acc, acc)

    # Inject noise to model's weights
    if Stages.STAGE_3 in self.stages:
      self._stage_3_noise()

  def evaluate(self, loader):
    self.model.eval()
    with torch.no_grad():
      accuracies = []
      for data, target in loader:
        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data)
        accuracies.append(self.accuracy(torch.argmax(output, 1).detach().cpu().numpy(), target.detach().cpu().numpy()))

      accuracy = sum(accuracies) / len(accuracies)
      return accuracy

  def train_and_test(self):
    self.train()
    acc = self.evaluate(self.test_loader)
    self.logger.log(f"Final test Accuracy {acc:.4f}")

  @staticmethod
  def save_checkpoint(state, is_best, epoch, filename="checkpoint.pth"):
    checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')
    if not os.path.exists(checkpoint_dir):
      os.mkdir(os.path.join(os.getcwd(), 'checkpoints'))
    save_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, save_path)
    if is_best:
      shutil.copyfile(save_path, os.path.join(checkpoint_dir, f"model_best_epoch_{epoch}.pth"))

  def _stage_1_noise(self):
    self.logger.log("Performing input perturbation...")
    for loader in tqdm(self.data_loaders):
      self._perturb_loader_data(loader)

  def _perturb_loader_data(self, data_loader: DataLoader):
    # FIXME: fix for non-image data

    dataset = torch.tensor(np.moveaxis(data_loader.dataset.data, -1, 1)).float()
    sensitivity = self._get_sensitivity(dataset)
    noised_data = self._perturb_dataset(dataset, sensitivity)

    dataset = np.moveaxis(noised_data.numpy(), 1, -1)
    dataset = (dataset - dataset.min()) / (dataset.max() - dataset.min()) * 255
    data_loader.dataset.data = dataset.astype(np.uint8)
    return data_loader

  @staticmethod
  def _perturb_dataset(dataset: torch.Tensor, sensitivity: torch.Tensor, eps: Optional[float] = 1):
    delta = 1 / (1.5 * len(dataset))
    stds = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / (25 * eps)
    shape = [dataset.size(i) for i in range(len(dataset.shape))]
    shape[1] = 1
    noise = [torch.normal(0, std, size=shape) for std in stds]
    noise = torch.cat(noise, dim=1)
    dataset += noise
    return dataset

  def _get_sensitivity(self, dataset: torch.Tensor) -> torch.Tensor:
    """
    :param dataset: [n, c, *] or [n, *]
    """
    if len(dataset.shape) == 2:
      norms = torch.norm(dataset, dim=1)
      dim = 0
    else:
      norms = torch.norm(dataset, dim=(2, 3)).transpose(0, 1)
      dim = 1
    sensitivity = torch.sqrt(torch.abs(norms.max(dim=dim)[0] - norms.min(dim=dim)[0]))
    return sensitivity

  def _stage_3_noise(self, clipping_norm: Optional[Union[float, List[float]]] = None):
    """
    Output perturbation on the model's weights

    'Federated learning with differential privacy: Algorithms and performance analysis', Wei et al., 2019
    :param clipping_norm:

    """
    delta = 1 / (1.5 * len(self.train_loader.dataset))
    max_sensitivity = 2 * clipping_norm / len(self.train_loader.dataset)
    exposure = len(self.train_loader) * self.epochs
    std = (max_sensitivity * exposure * np.sqrt(2 * np.log(1.25 / delta))) / self.epsilon
    self.logger.log(f"Output Perturbation -> max_sensitivity: {max_sensitivity:.4f}, "
                    f"std: {std:.4f}, "
                    f"max_norm: {clipping_norm}")
    # Sanitize
    for i, p in enumerate(self.model.parameters()):
      if clipping_norm is not None:
        if isinstance(clipping_norm, list):
          clip_val = clipping_norm[i]
        else:
          clip_val = clipping_norm
        p.data /= max(1, torch.norm(p.data) / clip_val)
      noise = torch.normal(mean=0, std=std, size=p.shape).to(p.data.device)
      p.data.add_(noise)

  def _wrap_optimizer(self, optimizer: torch.optim):
    # Opacus
    norm_clipper = (
      clipping.ConstantFlatClipper(self.max_norm)
      if not isinstance(self.max_norm, list)
      else clipping.ConstantPerLayerClipper(self.max_norm)
    )

    self.clipper = PerSampleGradientClipper(self.model, norm_clipper)

    def dp_zero_grad(self):
      self.msdp_trainer._zero_grad()
      self.original_zero_grad()

    def dp_step(self, closure=None):
      self.msdp_trainer._step()
      self.original_step(closure)

    optimizer.msdp_trainer = self
    optimizer.original_step = optimizer.step
    optimizer.step = types.MethodType(dp_step, optimizer)

    optimizer.original_zero_grad = optimizer.zero_grad
    optimizer.zero_grad = types.MethodType(dp_zero_grad, optimizer)

    self.optimizer = optimizer

  def _zero_grad(self):
    self.clipper.zero_grad()

  def _step(self):
    self.steps += 1
    self.clipper.clip_and_accumulate()
    clip_values, batch_size = self.clipper.pre_step()

    params = (p for p in self.model.parameters() if p.requires_grad)
    for p, clip_value in zip(params, clip_values):
      noise = self._generate_noise(clip_value, p)
      noise /= batch_size
      p.grad += noise

  def _generate_noise(self, max_grad_norm: float, reference: nn.parameter.Parameter) -> torch.Tensor:
    if self.noise_multiplier > 0 and max_grad_norm > 0:
      return torch.normal(
        0,
        self.noise_multiplier * max_grad_norm,
        reference.grad.shape,
        device=self.device,
      )
    return torch.zeros(reference.grad.shape, device=self.device)
