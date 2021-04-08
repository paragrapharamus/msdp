from torch.utils.data import DataLoader


def minibatch_loader(minibatch, minibatch_size, drop_last=True):
  return DataLoader(minibatch, batch_size=minibatch_size, drop_last=drop_last)
