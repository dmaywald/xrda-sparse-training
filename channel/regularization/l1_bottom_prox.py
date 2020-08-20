# Author: Jonathan Siegel
#
# Implements the l1-prox operator applied to the bottom values 
# in a form which can be called by the xRDA method.

import torch
import math

class l1_bottom_prox:

  def __init__(self, lam, c):
    self.lam = lam
    self.c = c
    return

  def apply(self, p, backward_step):
    if len(p.shape) == 2:
      p.data.copy_(torch.clamp(p - self.lam * backward_step, min=0) + torch.clamp(p + self.lam * backward_step, max=0))
    if len(p.shape) == 4:
      norms = torch.norm(p.data, dim=[2,3])
      new_norms = torch.clamp(norms - self.lam * backward_step, min=0)
      factor = new_norms / (norms + (norms == 0).float())
      p.data = factor[:, :, None, None] * p.data

  def get_zero_params(self, p):
    with torch.no_grad():
      if len(p.shape) == 2:
        return torch.zeros_like(p.data)
      if len(p.shape) == 4:
        return torch.zeros_like(torch.norm(p, dim=[2,3]))
      return torch.zeros_like(p.data)

  def get_param_increment(self, p):
    with torch.no_grad():
      if len(p.shape) == 2:
        nel = p.numel()
        nprun = math.floor(nel - self.c * math.sqrt(nel))
        if nprun <= 0:
          return torch.zeros_like(p.data)
        thresh = torch.kthvalue(torch.flatten(torch.abs(p)), nprun)[0]
        factor = torch.ones_like(p.data)
        if thresh > 0:
          factor = 1.0 - (torch.abs(p) / thresh)
        return (torch.abs(p) <= thresh).float() * factor
      if len(p.shape) == 4:
        norms = torch.norm(p.data, dim=[2,3])
        nel = norms.numel()
        nprun = math.floor(nel - self.c * math.sqrt(nel))
        if nprun <= 0:
          return torch.zeros_like(norms)
        thresh = torch.kthvalue(torch.flatten(norms), nprun)[0]
        factor = torch.ones_like(norms)
        if thresh > 0:
          factor = 1.0 - (norms / thresh)
        return (norms <= thresh).float() * factor
      return torch.zeros_like(p.data)
