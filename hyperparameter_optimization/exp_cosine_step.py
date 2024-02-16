# Author: Jonathan Siegel
#
# Implements iteration specs with an exponential cosine annealing schedule for the step size.
# The logarithm of the step size follows a cosine annealing between init_lr and init_lr / 100
# with warm-up.

import math

class ExpCosineSpecs:
  def __init__(self, max_iter, init_step_size=0.01, mom_ts=0.1, b_mom_ts=0.1, weight_decay=5e-4):
    self.max_iter = max_iter
    self.step = init_step_size
    self.mom_ts = mom_ts
    self.wd = weight_decay
    self.b_mom_ts = b_mom_ts

  def step_size(self, it):
    if 50 * it <= self.max_iter:
      return 50.0 * self.step * it / self.max_iter
    return self.step * math.exp(1.1*math.log(10) * (math.cos((50.0 / 49.0) * it * math.pi / self.max_iter - math.pi / 49.0) - 1.0))

  def backward_momentum_time_scale(self, it):
    return self.b_mom_ts

  def av_param(self, it):
    return 0.5 * (1.0 - math.cos(it * math.pi / self.max_iter))

  def momentum_time_scale(self, it):
    return self.mom_ts

  def weight_decay(self, it):
    return self.wd

