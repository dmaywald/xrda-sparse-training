# Author: Jonathan Siegel
#
# Implements the l1-prox operator in a form which can be called
# by the xRDA method. Applies group lasso to convolutional layers.

import torch
import math

class l1_kernel_prox:

  def __init__(self, lam, maximum_factor):
    self.lam = lam
    self.maximum_factor = maximum_factor
    return

  def apply(self, p, backward_step):
    if len(p.shape) == 2 or len(p.shape) == 1:
      p.data.copy_(torch.clamp(p - self.lam * backward_step, min=0) + torch.clamp(p + self.lam * backward_step, max=0))
    if len(p.shape) == 4:
      norms = torch.norm(p.data, dim=[2,3])
      new_norms = torch.clamp(norms - self.lam * backward_step, min=0)
      factor = new_norms / (norms + (norms == 0).float())
      p.data = factor[:, :, None, None] * p.data

  def get_zero_params(self, p):
    with torch.no_grad():
      if len(p.shape) == 4:
        return torch.zeros_like(torch.norm(p, dim=[2,3]))
      return torch.zeros_like(p.data)

  def get_param_increment(self, p):
    with torch.no_grad():
      if len(p.shape) == 4:
        norms = torch.norm(p.data, dim=[2,3])
        k_size = p.shape[2] * p.shape[3]
        maximum = torch.max(norms)
        if maximum > 0:
          return math.sqrt(k_size) * maximum * self.maximum_factor / (1.0 + (self.maximum_factor - 1.0)*(norms / maximum))
        else:
          return torch.zeros_like(norms)
      maximum = torch.max(torch.abs(p.data))
      if maximum > 0:
        return maximum * self.maximum_factor / (1.0 + (self.maximum_factor - 1.0)*(torch.abs(p.data) / maximum))
      else:
        return torch.zeros_like(p.data)

