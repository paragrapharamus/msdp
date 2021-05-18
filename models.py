import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from opacus.utils.module_modification import convert_batchnorm_modules
from torchvision.models import resnet18, squeezenet1_1


class MSDPBase(pl.LightningModule):
  def __init__(self, *args):
    super(MSDPBase, self).__init__()

    self.module = None

    self.train_acc = pl.metrics.Accuracy()
    self.valid_acc = pl.metrics.Accuracy()
    self.test_acc = pl.metrics.Accuracy()

    self.personal_log_fn = None
    self.batch_processing_hook = None

    self.training_losses = []
    self.training_accuracies = []
    self.validation_accuracies = []
    self.test_accuracy = -1

  @property
  def automatic_optimization(self) -> bool:
    return False

  def forward(self, x):
    return self.module(x)

  def training_step(self, batch, batch_idx):
    opt = self.optimizers()
    data, targets = batch
    data, targets = data.to(self.device), targets.to(self.device)
    if self.batch_processing_hook is not None:
      self.batch_processing_hook(data)
    output = self(data)
    loss = self.compute_loss(output, targets)
    self.manual_backward(loss, opt)
    if hasattr(self, 'virtual_batches'):
      if ((batch_idx + 1) % self.virtual_batches == 0) or ((batch_idx + 1) == len(self.train_dataloader())):
        opt.step()
        opt.zero_grad()
      else:
        opt.virtual_step()
    else:
      opt.step()
      opt.zero_grad()
    self.log('train_acc', self.train_acc(torch.argmax(output, 1), targets))
    self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
    return {'train_loss': loss}

  def training_epoch_end(self, outputs):
    accuracy = self.train_acc.compute()
    self.log('train_acc_epoch', accuracy)
    epoch_loss = torch.stack([x['train_loss'] for x in outputs]).mean()
    self.training_losses.append(epoch_loss.item())
    self.training_accuracies.append(accuracy.item())
    if self.personal_log_fn:
      self.personal_log_fn(f"Train epoch {self.current_epoch} - loss: {epoch_loss:.4f}, accuracy: {accuracy:.2f}")

  def validation_step(self, batch, batch_idx):
    data, targets = batch
    data, targets = data.to(self.device), targets.to(self.device)
    output = self(data)
    loss = self.compute_loss(output, targets)
    self.log('valid_acc', self.valid_acc(torch.argmax(output, 1), targets))
    self.log('valid_loss', loss, on_step=False, on_epoch=True)
    return {'valid_loss': loss}

  def validation_epoch_end(self, outputs):
    accuracy = self.valid_acc.compute()
    self.log('valid_acc_epoch', accuracy)
    self.validation_accuracies.append(accuracy.item())
    if self.personal_log_fn:
      epoch_loss = torch.stack([x['valid_loss'] for x in outputs]).mean()
      self.personal_log_fn(f"Validation epoch {self.current_epoch} - loss: {epoch_loss:.4f}, accuracy: {accuracy:.2f}")

  def test_step(self, batch, batch_idx):
    data, targets = batch
    data, targets = data.to(self.device), targets.to(self.device)
    output = self(data)
    loss = self.compute_loss(output, targets)
    self.log('test_acc', self.test_acc(torch.argmax(output, 1), targets))
    self.log('test_loss', loss, on_step=False, on_epoch=True)
    return {'test_loss': loss}

  def test_epoch_end(self, outputs):
    accuracy = self.test_acc.compute()
    self.log('test_acc_epoch', accuracy)
    if self.personal_log_fn:
      epoch_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
      self.personal_log_fn(f"Test - loss: {epoch_loss:.4f}, accuracy: {accuracy:.2f}")
    self.test_accuracy = accuracy.item()

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


class Cifar10Net(MSDPBase):
  def __init__(self, *args):
    super(Cifar10Net, self).__init__(args)

    self._NUM_CLASSES = 10
    self.module = nn.Sequential(
      nn.Conv2d(3, 32, kernel_size=3, padding=1),
      nn.ReLU(True),
      nn.MaxPool2d(2),
      nn.Conv2d(32, 64, kernel_size=3, padding=1),
      nn.ReLU(True),
      nn.MaxPool2d(2),
      nn.Conv2d(64, 128, kernel_size=3, padding=1),
      nn.ReLU(True),
      nn.MaxPool2d(2),
      nn.Conv2d(128, self._NUM_CLASSES, kernel_size=4),
      # nn.Softmax(dim=1),
      # nn.LogSoftmax(dim=1)
    )

  def forward(self, x):
    return self.module(x).view(-1, self._NUM_CLASSES)


class Cifar10ResNet(MSDPBase):
  def __init__(self, *args):
    super(Cifar10ResNet, self).__init__(args)
    self._NUM_CLASSES = 10
    self.module = convert_batchnorm_modules(resnet18(num_classes=self._NUM_CLASSES))

  def forward(self, x):
    return self.module(x).view(-1, self._NUM_CLASSES)


class MnistCNNNet(MSDPBase):
  def __init__(self, *args):
    super(MnistCNNNet, self).__init__(args)

    self._NUM_CLASSES = 10
    self.module = nn.Sequential(
      nn.Conv2d(1, 32, kernel_size=3),
      nn.ReLU(True),
      nn.MaxPool2d(2),
      nn.Conv2d(32, 64, kernel_size=3),
      nn.ReLU(True),
      nn.MaxPool2d(2),
      nn.Conv2d(64, 128, kernel_size=3),
      nn.Conv2d(128, self._NUM_CLASSES, kernel_size=3),
      # nn.Softmax(dim=1),
      # nn.LogSoftmax(dim=1)
    )

  def forward(self, x):
    return self.module(x).view(-1, self._NUM_CLASSES)


class MnistFCNet(MSDPBase):
  def __init__(self, *args):
    super(MnistFCNet, self).__init__(args)

    self._NUM_CLASSES = 10
    self.module = nn.Sequential(
      nn.Linear(28 * 28, 256),
      nn.ReLU(True),
      nn.Linear(256, 128),
      nn.ReLU(True),
      nn.Linear(128, 10),
      # nn.Softmax(dim=1)
    )

  def forward(self, x):
    return self.module(x.view(-1, 28 * 28)).view(-1, self._NUM_CLASSES)


class SqueezeNetDR(MSDPBase):
  def __init__(self, *args):
    super(SqueezeNetDR, self).__init__(args)
    self._NUM_CLASSES = 5
    self.module = squeezenet1_1(pretrained=True)
    self.module.classifier[1] = nn.Conv2d(512, 5, kernel_size=(1, 1), stride=(1, 1))
    self.module.num_classes = self._NUM_CLASSES
    self.module = convert_batchnorm_modules(self.module)

  def forward(self, x):
    return self.module(x).view(-1, self._NUM_CLASSES)


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
