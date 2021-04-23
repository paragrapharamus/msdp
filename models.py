import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class BaseNet(pl.LightningModule):
  @property
  def automatic_optimization(self) -> bool:
    return False

  def forward(self, x):
    return self.classifier(x.float()).view(-1, 10)

  def training_step(self, batch, batch_idx):
    opt = self.optimizers()
    opt.zero_grad()
    data, targets = batch
    data, targets = data.to(self.device), targets.to(self.device)
    output = self(data)
    loss = self.loss_fn(output, targets)
    self.manual_backward(loss, opt)
    opt.step()
    self.log('train_acc', self.train_acc(torch.argmax(output, 1), targets), on_step=True, on_epoch=False)
    self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False)
    return loss

  def validation_step(self, batch, batch_idx):
    data, targets = batch
    data, targets = data.to(self.device), targets.to(self.device)
    output = self(data)
    loss = self.loss_fn(output, targets)
    self.log('valid_acc', self.valid_acc(torch.argmax(output, 1), targets), on_step=True, on_epoch=True)
    self.log('valid_loss', loss, on_step=True, on_epoch=True)

  def test_step(self, batch, batch_idx):
    data, targets = batch
    data, targets = data.to(self.device), targets.to(self.device)
    output = self(data)
    loss = self.loss_fn(output, targets)
    self.log('test_acc', self.test_acc(torch.argmax(output, 1), targets), on_step=True, on_epoch=True)
    self.log('test_loss', loss, on_step=True, on_epoch=False)

  def add_optimizer(self, optimizer):
    self._optimizer = optimizer

  def configure_optimizers(self):
    if hasattr(self, "_optimizer"):
      optimizer = self._optimizer
    else:
      optimizer = self.default_optimizer()
    return optimizer

  def default_optimizer(self):
    return torch.optim.Adam(self.parameters(), lr=1e-3)

  @staticmethod
  def compute_loss(output, targets):
    return F.cross_entropy(output, targets)


class Cifar10Net(BaseNet):
  def __init__(self, *args):
    super(BaseNet, self).__init__()

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
