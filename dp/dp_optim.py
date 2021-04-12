import torch
from torch.optim import SGD, Adam


def generate_otimizer_class(cls):
  class DPOptimizerClass(cls):
    def __init__(self, params, l2_norm_clip, noise_multiplier, minibatch_size, microbatch_size, *args, **kwargs):
      kwargs['params'] = params
      super(DPOptimizerClass, self).__init__(*args, **kwargs)

      self.l2_norm_clip = l2_norm_clip
      self.noise_multiplier = noise_multiplier
      self.microbatch_size = microbatch_size
      self.minibatch_size = minibatch_size

      for group in self.param_groups:
        group['accum_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in group['params']]

    def zero_minibatch_grad(self):
      super(DPOptimizerClass, self).zero_grad()

    def minibatch_step(self):
      total_norm = 0.
      for group in self.param_groups:
        for param in group['params']:
          if param.requires_grad:
            total_norm += param.grad.data.norm(2).item() ** 2.
      total_norm = total_norm ** .5
      clip_coef = min(self.l2_norm_clip / (total_norm + 1e-6), 1.)

      for group in self.param_groups:
        for param, accum_grad in zip(group['params'], group['accum_grads']):
          if param.requires_grad:
            accum_grad.add_(param.grad.data.mul(clip_coef))

    def zero_grad(self):
      for group in self.param_groups:
        for accum_grad in group['accum_grads']:
          if accum_grad is not None:
            accum_grad.zero_()
        for p in group['params']:
          if p.grad is not None:
            if p.grad.grad_fn is not None:
              p.grad.detach_()
            else:
              p.grad.requires_grad_(False)
            p.grad.zero_()

    def step(self, *args, **kwargs):
      for group in self.param_groups:
        for param, accum_grad in zip(group['params'], group['accum_grads']):
          if param.requires_grad:
            param.grad.data = accum_grad.clone()
            param.grad.data.add_(self.l2_norm_clip * self.noise_multiplier * torch.randn_like(param.grad.data))
            param.grad.data.mul_(self.microbatch_size / self.minibatch_size)
      super(DPOptimizerClass, self).step(*args, **kwargs)

  return DPOptimizerClass


DPSGD = generate_otimizer_class(SGD)
DPAdam = generate_otimizer_class(Adam)