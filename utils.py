import os

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def minibatch_loader(minibatch, minibatch_size, drop_last=True):
  return DataLoader(minibatch, batch_size=minibatch_size, drop_last=drop_last)


def get_next_available_dir(root, dir_name, absolute_path=True, create=True):
  checkpoint_dir_base = os.path.join(root, dir_name)
  dir_id = 1
  checkpoint_dir = f"{checkpoint_dir_base}_{dir_id}"
  while os.path.exists(checkpoint_dir):
    dir_id += 1
    checkpoint_dir = f"{checkpoint_dir_base}_{dir_id}"
  if create:
    os.mkdir(checkpoint_dir)
  if absolute_path:
    return checkpoint_dir
  else:
    return f"{dir_name}_{dir_id}"


def _plot(data_dict, x_label, y_label, title):
  fig, ax = plt.subplots()  # (figsize=(10,10))

  if 'x_ticks' in data_dict:
    x_values = data_dict.pop('x_ticks')
    if len(x_values) > 20:
      x_values = None  # too crowded to read on the figure
  else:
    x_values = None

  max_x_range_len = 0
  for name, data in data_dict.items():
    if x_values is not None:
      ax.plot(list(range(len(x_values))), data, label=name)
    else:
      ax.plot(data, label=name)

  if x_values is not None:
    plt.xticks(list(range(len(x_values))), x_values)

  ax.legend()
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.title(title)
  plt.show()
  return fig


def load_and_plot_privacy_param_variation():
  eps1 = {'name': 'eps_1', 'fp': './eps1.npy', }
  noise_multiplier = {'name': 'noise_multiplier', 'fp': './noise_multiplier.npy'}
  eps3 = {'name': 'eps_3', 'fp': './eps3.npy'}

  files = [eps1, noise_multiplier, eps3]

  curve_names = ['Test accuracy', 'MEA fidelity', 'MIA accuracy']

  for data_file in files:
    data = dict()
    with open(data_file['fp'], 'rb') as f:
      data['x_ticks'] = np.load(f)
      for curve in curve_names:
        data[curve] = np.load(f)
      # data['x_ticks'] = np.array(data_file['rng'])
    _plot(data, data_file['name'], 'Privacy and Utility', 'Small CNN on Cifar10')


def load_and_plot_learning_curves():
  def fetch(fs, metric_name):
    metric_data = dict()
    for f in fs:
      metric_data[f['name']] = f[metric_name]
    return metric_data

  metrics = ['val_acc']

  msdp = {'name': 'MSDPFL', 'fp': "outFL/MNIST/low_eps/msdpfl/stats.npy"}
  opacus = {'name': 'Opacus FL', 'fp': "outFL/MNIST/low_eps/opacusfl/stats.npy"}
  non_p = {'name': 'Non-Private FL', 'fp': "outFL/MNIST/npfl/stats.npy"}

  title = 'Highly private FL training on MNIST'

  files = [msdp, opacus, non_p]

  for data_file in files:
    data = dict()
    with open(data_file['fp'], 'rb') as f:
      for metric in metrics:
        data[metric] = np.load(f)
    data_file.update(**data)


  for metric in metrics:
    metric_data = fetch(files, metric)
    f = _plot(metric_data, 'Epochs', metric, title)
    if metric == 'val_acc':
      f.savefig(f"./val_acc.png", bbox_inches='tight')


def load_and_plot_dr():
  def fetch(fs, metric_name):
    metric_data = dict()
    for f in fs:
      metric_data[f['name']] = f[metric_name]
    return metric_data

  def dr_plot(data_dict, x_label, y_label, title):
    fig, ax = plt.subplots()  # (figsize=(10,10))
    for name, data in data_dict.items():
      ax.plot(data, label=name)
    ax.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

  metrics = {'centralised': ['train_loss', 'train_acc', 'val_acc'],
             'fl': ['val_acc']
             }

  msdp = {'name': 'MSDP', 'fp': "out_centralMSDP/DR/msdp/MSDPTrainer_0_plot_stats.npy"}
  msdpfl = {'name': 'MSDPFL', 'fp': "outFL/DR/msdpfl/stats.npy"}
  opacus = {'name': 'Opacus', 'fp': "out_centralMSDP/DR/opacus/MSDPTrainer_0_plot_stats.npy"}
  opacusfl = {'name': 'OpacusFL', 'fp': "outFL/DR/opacus_fl/stats.npy"}
  non_p = {'name': 'Non-Private', 'fp': "out_centralMSDP/DR/np/MSDPTrainer_0_plot_stats.npy"}
  non_pfl = {'name': 'Non-Private FL', 'fp': "outFL/DR/np_fl/stats.npy"}
  title = 'FL training on DR'

  central = [msdp, opacus, non_p]
  fl = [msdpfl, opacusfl, non_pfl]
  files = central + fl

  for data_file in files:
    data = dict()
    if data_file in central:
      metric_type = 'centralised'
    else:
      metric_type = 'fl'
    with open(data_file['fp'], 'rb') as f:
      for metric in metrics[metric_type]:
        data[metric] = np.load(f)

    data_file.update(**data)

  for metric in ['val_acc']:
    metric_data = fetch(files, metric)
    dr_plot(metric_data, 'Epochs/ Rounds', metric, title)
