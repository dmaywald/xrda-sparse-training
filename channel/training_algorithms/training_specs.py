# Author: Jonathan Siegel
#
# Implements simple iteration specs with fixed values for each of the hyperparameters.

import math

class IterationSpecs:
  def __init__(self, max_iter=391000, init_step_size=0.01, init_av_param=1.0, mom_ts=0.1, b_mom_ts=0.1, weight_decay=5e-4):
    self.step = init_step_size
    self.av = init_av_param
    self.mom_ts = mom_ts
    self.wd = weight_decay
    self.b_mom_ts = b_mom_ts
    self.max_iter = max_iter

  def step_size(self, it):
    return 0.5 * self.step * (1.0 + math.cos((it * math.pi) / self.max_iter))

  def backward_momentum_time_scale(self, it):
    return self.b_mom_ts

  def av_param(self, it):
    return self.av + (1.0 - self.av) * 0.5 * (1.0 - math.cos((it * math.pi) / self.max_iter))

  def momentum_time_scale(self, it):
    return self.mom_ts

  def weight_decay(self, it):
    return self.wd

  def set_av_param(self, av_param):
    self.av = av_param

