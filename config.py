import torch.optim


class ExperimentConfig:
  def __init__(self):
    self.name = None

    ##########################
    # Client training
    ##########################
    self.batch_size = 128
    self.test_batch_size = 500
    self.epochs = 10
    self.lr = 0.02
    self.gamma = 0.7
    self.weight_decay = 5e-4
    self.momentum = 0.9
    self.no_cuda = False
    self.dry_run = False
    self.seed = 42
    self.save_model = True
    self.save_model_path = './model.pth'
    self.trained_model_path = './checkpoints/checkpoint.pth'

    ##########################
    # MSDP
    ##########################
    # Stage 1
    self.eps1 = 5

    # Stage 2
    self.noise_multiplier = 0.3
    self.max_grad_norm = 2
    self.virtual_batches = 1

    # Stage 3
    self.eps3 = 20
    self.max_weight_norm = 20

    # Stage 4
    self.eps4 = 1
    self.max_weight_norm_aggregated = 2

    ##########################
    # FL arguments
    ##########################
    self.num_rounds = 25
    self.num_clients = 10
    self.clients_per_round = 10
    self.client_optimizer_class = torch.optim.SGD
    self.partition_method = 'heterogeneous'
    self.alpha = 20
    self.client_local_test_split = 0.1

    ##########################
    # Enable Attacks
    ##########################
    self.membership_inference = False
    self.model_extraction = False
    self.knockoffnet_extraction = False