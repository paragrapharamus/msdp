class Client:
  def __init__(self,
               id,
               data,
               epochs,
               batch_size,
               optimizer_class,
               learning_rate,
               weight_decay,
               momentum,
               device,
               eps1,
               noise_multiplier,
               max_grad_norm,
               eps3,
               max_weight_norm):
    pass

  def update_weights(self, weights):
    pass

  def get_weights(self):
    pass

  def train(self):
    pass

  def test(self):
    pass
