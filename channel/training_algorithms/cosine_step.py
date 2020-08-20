# Author: Jonathan Siegel
#
# Implements iteration specs with the cosine annealing schedule for both step size
# and averaging parameter.

import math

class CosineSpecs:
  def __init__(self, max_iter, init_step_size=0.01, mom_ts=0.1, b_mom_ts=0.1, weight_decay=5e-4):
    self.max_iter = max_iter
    self.step = init_step_size
    self.mom_ts = mom_ts
    self.wd = weight_decay
    self.b_mom_ts = b_mom_ts

  def step_size(self, it):
    return 0.5 * self.step * (1.0 + math.cos(it * math.pi / self.max_iter))

  def backward_momentum_time_scale(self, it):
    return self.b_mom_ts

  def av_param(self, it):
    return 0.5 * (1.0 - math.cos(it * math.pi / self.max_iter))

  def momentum_time_scale(self, it):
    return self.mom_ts

  def weight_decay(self, it):
    return self.wd

