import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class Cifar10Net(pl.LightningModule):
  def __init__(self):
    super(Cifar10Net, self).__init__()

    NUM_CLASSES = 10
    self.classifier = nn.Sequential(
      nn.Conv2d(3, 32, kernel_size=3, padding=1),
      nn.ReLU(True),
      nn.MaxPool2d(2),
      nn.Conv2d(32, 64, kernel_size=3, padding=1),
      nn.ReLU(True),
      nn.MaxPool2d(2),
      nn.Conv2d(64, 128, kernel_size=3, padding=1),
      nn.ReLU(True),
      nn.MaxPool2d(2),
      nn.Conv2d(128, NUM_CLASSES, kernel_size=4),
      nn.LogSoftmax(dim=1)
    )

    self.train_acc = pl.metrics.Accuracy()
    self.valid_acc = pl.metrics.Accuracy()
    self.test_acc = pl.metrics.Accuracy()

  def forward(self, x):
    return self.classifier(x.float()).view(-1, 10)

  def training_step(self, batch, batch_idx):
    data, targets = batch
    data, targets = data.to(self.device), targets.to(self.device)
    output = self(data)
    loss = F.cross_entropy(output, targets)
    self.log('train_acc_step', self.train_acc(torch.argmax(output, 1), targets))
    return loss

  def training_epoch_end(self, outs):
    # log epoch metric
    self.log('train_acc_epoch', self.train_acc.compute())

  def validation_step(self, batch, batch_idx):
    data, targets = batch
    data, targets = data.to(self.device), targets.to(self.device)
    output = self(data)
    loss = F.cross_entropy(output, targets)
    self.log('valid_acc_step', self.valid_acc(torch.argmax(output, 1), targets))
    self.log('valid_loss', loss, on_step=True)

  def validation_epoch_end(self, outputs):
    self.log('valid_acc_epoch', self.valid_acc.compute())

  def test_step(self, batch, batch_idx):
    data, targets = batch
    data, targets = data.to(self.device), targets.to(self.device)
    output = self(data)
    loss = F.cross_entropy(output, targets)
    self.log('test_acc_step', self.test_acc(torch.argmax(output, 1), targets))
    self.log('test_loss', loss, on_step=True)

  def test_epoch_end(self, outputs):
    self.log('test_acc_epoch', self.train_acc.compute())

  def add_optimizer(self, optimizer):
    self._optimizer = optimizer

  def configure_optimizers(self):
    if hasattr(self, "optimizer"):
      optimizer = self._optimizer
    else:
      optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    return optimizer


class AttackModel(nn.Module):
  def __init__(self, num_classes):
    super(AttackModel, self).__init__()
    self.classifier = nn.Sequential(
      nn.Linear(num_classes, 128),
      nn.ReLU(True),
      nn.Dropout(0.3),
      nn.Linear(128, 64),
      nn.ReLU(True),
      nn.Dropout(0.2),
      nn.Linear(64, 64),
      nn.ReLU(True),
      nn.Linear(64, 1),
      nn.Sigmoid()
    )

  def forward(self, x):
    return self.classifier(x.float())
