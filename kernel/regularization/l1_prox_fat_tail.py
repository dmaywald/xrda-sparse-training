# Author: Jonathan Siegel
#
# Implements the fat tailed l1-prox operator in a form which can be called
# by the xRDA method.

import torch
import math

class l1_prox_fat_tail:

  def __init__(self, lam, maximum_factor):
    self.lam = lam
    self.maximum_factor = maximum_factor
    return

  def apply(self, p, backward_step):
    p.data.copy_(torch.clamp(p - self.lam * backward_step, min=0) + torch.clamp(p + self.lam * backward_step, max=0))

  def get_zero_params(self, p):
    return torch.zeros_like(p.data)

  def get_param_increment(self, p):
    maximum = torch.max(torch.abs(p.data))
    if maximum > 0:
      return maximum * self.maximum_factor / (1.0 + (self.maximum_factor - 1.0)*(torch.abs(p.data) / maximum))
    else:
      return torch.zeros_like(p.data)

