# Author: Jonathan Siegel
#
# Implements simple iteration specs with fixed values for each of the hyperparameters.

class IterationSpecs:
  def __init__(self, step_size=0.01, av_param=1.0, mom_ts=0.1, b_mom_ts=0.1, weight_decay=5e-4):
    self.step = step_size
    self.av = av_param
    self.mom_ts = mom_ts
    self.wd = weight_decay
    self.b_mom_ts = b_mom_ts

  def step_size(self, it):
    return self.step

  def backward_momentum_time_scale(self, it):
    return self.b_mom_ts

  def av_param(self, it):
    return self.av

  def momentum_time_scale(self, it):
    return self.mom_ts

  def weight_decay(self, it):
    return self.wd

  def set_step_size(self, step_size):
    self.step = step_size

  def set_av_param(self, av_param):
    self.av = av_param

