# Author: Jonathan Siegel
#
# Implements the l1-prox operator in a form which can be called
# by the xRDA method. Contains both the regular mode and a mode for
# kernel sparsity.

import torch
import math

class l1_prox:

  def __init__(self, lam, maximum_factor, mode = 'normal'):
    self.lam = lam
    self.maximum_factor = maximum_factor
    self.mode = mode
    return

  def apply(self, p, backward_step):
    p.data.copy_(torch.clamp(p - self.lam * backward_step, min=0) + torch.clamp(p + self.lam * backward_step, max=0))

  def get_zero_params(self, p):
    return torch.zeros_like(p.data)

  def get_running_av(self, p):
    with torch.no_grad():
      if len(p.shape) == 4 and self.mode == 'kernel':
        norms = torch.norm(p, p=1, dim=[2,3])
        return (1.0 / (p.shape[2] * p.shape[3])) * norms[:,:,None,None] * torch.ones_like(p.data)
      if len(p.shape) == 4 and self.mode == 'channel':
        norms = torch.norm(p, p=1, dim=[1,2,3])
        return (1.0 / (p.shape[1] * p.shape[2] * p.shape[3])) * norms[:,None,None,None] * torch.ones_like(p.data)
      return torch.abs(p.data)

  def calculate_backward_v(self, running_av):
    maximum = torch.max(running_av)
    if maximum > 0 and len(running_av.shape) == 4 and self.mode == 'kernel':
      return math.sqrt(running_av.shape[2] * running_av.shape[3]) * self.maximum_factor / (1.0 + (self.maximum_factor - 1.0)*(running_av / maximum))
    elif maximum > 0 and len(running_av.shape) == 4 and self.mode == 'channel':
      return math.sqrt(running_av.shape[1] * running_av.shape[2] * running_av.shape[3]) \
             * self.maximum_factor / (1.0 + (self.maximum_factor - 1.0)*(running_av / maximum))
    elif maximum > 0:
      return self.maximum_factor / (1.0 + (self.maximum_factor - 1.0)*(running_av / maximum))
    else:
      return torch.zeros_like(running_av)

  def register_running_av(self, running_av):
    return

  def reset(self):
    return
