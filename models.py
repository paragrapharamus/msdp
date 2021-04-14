import torch.nn as nn


class Cifar10Net(nn.Module):
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

  def forward(self, x):
    return self.classifier(x.float()).view(-1, 10)


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
