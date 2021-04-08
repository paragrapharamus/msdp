
class MSPDTrainer:
  def __init__(self, model, optimizer, data, stages, params, save_model=True):
    """

    :param model:
    :param optimizer:
    :param data:
    :param stages:
    :param params:
    :param save_model:
    """
    self.model = model
    self.optimizer = optimizer
    self.data = data
    self.stages = stages
    self.epochs = params['epochs']
    self.save_path = params['save_path'] if save_model else None

  def train(self):
    pass

  def _stage_1_noise(self):
    pass

  def _stage_3_noise(self):
    pass