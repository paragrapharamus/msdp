import torch.optim


class ExperimentConfig:
  def __init__(self):
    self.name = None

    ##########################
    # Client training
    ##########################
    self.batch_size = 256
    self.test_batch_size = 1000
    self.epochs = 25
    self.lr = 0.02
    self.gamma = 0.7
    self.weight_decay = 5e-4
    self.momentum = 0.9
    self.no_cuda = False
    self.dry_run = False
    self.seed = 42
    self.save_model = True
    self.save_model_path = 'model.pth'
    self.trained_model_path = './checkpoints/checkpoint.pth'

    ##########################
    # MSDP
    ##########################
    self.stage1 = False
    self.stage2 = False
    self.stage3 = False
    self.stage4 = False

    # Stage 1
    self.eps1 = 10

    # Stage 2
    self.noise_multiplier = 0.3
    self.max_grad_norm = 5
    self.virtual_batches = 1

    # Stage 3
    self.eps3 = 1
    self.max_weight_norm = 20

    # Stage 4
    self.eps4 = 1
    self.max_weight_norm_aggregated = 20

    ##########################
    # FL arguments
    ##########################
    self.num_rounds = 25
    self.num_clients = 5
    self.clients_per_round = 0
    self.client_optimizer_class = torch.optim.SGD
    self.partition_method = 'heterogeneous'
    self.alpha = 50
    self.client_local_test_split = 0.1

    ##########################
    # Enable Attacks
    ##########################
    self.membership_inference = True
    self.model_extraction = True
    self.knockoffnet_extraction = False

  def __repr__(self):
    s = '\n'
    for k, v in self.__dict__.items():
      s += f'{k}: {v}\n'
    return s

  def set_value(self, k, v):
    self.__dict__[k] = v
