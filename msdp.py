import torch
from torch.utils.data import TensorDataset
from torch import nn
from tqdm import tqdm

from utils import minibatch_loader


class MSPDTrainer:
  STATS_FMT = "[{:>5s}] loss: {:+.4f}, acc: {:.4f}"

  def __init__(self, model, optimizer, data_loaders, stages, epochs, batch_size, minibatch_size, device, save_path=None):
    """

    :param model:
    :param optimizer:
    :param data_loaders:
    :param stages:
    :param epochs:
    :param batch_size:
    :param minibatch_size:
    :param device:
    :param save_path:
    """
    self.model = model
    self.optimizer = optimizer
    if len(data_loaders) == 2:
      self.train_loader, self.test_loader = data_loaders
    elif len(data_loaders) == 3:
      self.train_loader, self.val_loader, self.test_loader = data_loaders
    else:
      raise Exception("Invalid number of loaders")
    self.data_loaders = data_loaders
    self.stages = stages
    self.epochs = epochs
    self.batch_size = batch_size
    self.minibatch_size = minibatch_size
    self.device = device
    self.save_path = save_path

  def train(self):

    # Inject noise to data
    if 'STAGE_1' in self.stages:
      self._stage_1_noise()

    for epoch in range(self.epochs):
      print("Epoch %d/%d" % (epoch + 1, self.epochs))

      prog_bar = tqdm(total=len(self.train_loader.dataset))
      running_loss = 0
      running_correct_preds = 0

      # Model training
      self.model.train()

      for batch_idx, batch in enumerate(self.train_loader):
        self.optimizer.zero_grad()
        correct_preds = 0
        batch_loss = 0
        for data, target in minibatch_loader(TensorDataset(*batch), self.minibatch_size):
          data, target = data.to(self.device), target.to(self.device)
          self.optimizer.zero_minibatch_grad()

          output = self.model(data)
          loss = nn.NLLLoss()(output, target)
          loss.backward()
          self.optimizer.minibatch_step()

          correct_preds += float(torch.sum(torch.argmax(output) == target))
          batch_loss += loss.item()

        self.optimizer.step()
        running_loss += batch_loss
        running_correct_preds += correct_preds

        batch_stats_str = MSPDTrainer.STATS_FMT.format(
          'Training',
          batch_loss / len(batch),
          correct_preds / len(batch),
        )
        prog_bar.set_description(batch_stats_str)
        prog_bar.update(len(batch))

      # Epoch statistics.
      epoch_loss = running_loss / len(self.data_loaders['train'].dataset)
      epoch_acc = running_correct_preds / len(self.data_loaders['train'].dataset)
      prog_bar.close()
      print(f"Training loss: {epoch_loss:.6f}, Training acc: {epoch_acc:.4f}")

      # Validation step
      print("Validation Step:")
      self.evaluate(self.data_loaders['val'])

    # Inject noise to model's weights
    if 'STAGE_3' in self.stages:
      self._stage_3_noise()

    if self.save_path is not None:
      torch.save(self.model.state_dict, self.save_path)

  def evaluate(self, loader):
    self.model.eval()
    with torch.no_grad():
      correct_preds = 0
      for data, target in tqdm(loader):
        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data)
        correct_preds += float(torch.sum(torch.argmax(output, 1) == target))

      print(f"Accuracy: {correct_preds / len(loader.dataset)}\n")

  def train_and_test(self):
    self.train()
    print("Final test")
    self.evaluate(self.data_loaders['test'])

  def _stage_1_noise(self):
    pass

  def _stage_3_noise(self):
    pass
