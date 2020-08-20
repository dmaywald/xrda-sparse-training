# Author: Jonathan Siegel
#
# Implements an advanced version of the dual averaging method
# which is augmented with momentum.
#
# The method is constructed with a class which provides the step size,
# averaging coefficient, weight decay parameter, and momentum, it_specs, and a class which 
# performs the proximal step of the regularizer, prox. 
#
# it_specs only definitely needs to provide the
# step size, if it fails to provide the other parameters a default value of 0 is used.
# In addition, the functions providing these parameters should take the iteration number
# as a parameter. This provides a way of defining adaptive iteartion parameters.
#
# It is assumed that this
# backward step can be performed on each of the parameter groups of the
# model independently. Additionally, the method prox.apply must modify
# its input (not return the result).

import math
import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer

class ConstStepFB:

  def __init__(self, step_size):
    self.step = step_size

  def step_size(self, it):
    return self.step

class xRDA(Optimizer):
  
  def __init__(self, params, it_specs=ConstStepFB(0.003), prox=None):
    defaults = dict()
    super(xRDA, self).__init__(params, defaults)
    self.it_specs = it_specs
    self.iteration = 1
    if prox is not None:
      self.prox = prox
      self.has_prox = True
    else:
      self.has_prox = False

  def set_it_specs(self, new_specs):
    self.it_specs = new_specs

  def step(self, closure=None):
    loss = None
    if closure is not None:
      loss = closure()
    # Gather Parameters
    step_size = self.it_specs.step_size(self.iteration)

    av = 0 # default value of the averaging parameter
    try: # if it_specs provides a different value, use it
      av = self.it_specs.av_param(self.iteration)
    except AttributeError:
      pass

    weight_decay = 0 # default value of weight_decay
    try:
      weight_decay = self.it_specs.weight_decay(self.iteration)
    except AttributeError:
      pass

    mom_ts = 0
    try:
      mom_ts = self.it_specs.momentum_time_scale(self.iteration)
    except AttributeError:
      pass

    b_mom_ts = 0
    try:
      b_mom_ts = self.it_specs.backward_momentum_time_scale(self.iteration)
    except AttributeError:
      pass

    if self.has_prox:
      self.prox.reset()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue

        state = self.state[p]

        if len(state) == 0:
          if self.has_prox:
            state['backward_step'] = self.prox.get_zero_params(p)
            state['running_av'] = self.prox.get_zero_params(p)
          state['p_temp'] = torch.clone(p).detach()
          state['v'] = torch.zeros_like(p.data)

        if self.has_prox:
          running_av = self.prox.get_running_av(p)
          state['running_av'].mul_(math.exp(-step_size / b_mom_ts)).add_(-math.expm1(-step_size / b_mom_ts), running_av)
          self.prox.register_running_av(state['running_av']) 

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue

        state = self.state[p]
        
        dp = p.grad.data
        dp.add_(weight_decay, p.data)

        if self.has_prox:
          backward_v = self.prox.calculate_backward_v(state['running_av'])
          state['backward_step'] = av * state['backward_step'] + step_size * backward_v

        # Average the pre and post proximal iterates
        state['p_temp'].mul_(av).add_((1.0 - av),  p.data)

        # Calculate the velocity.
        state['v'].mul_(math.exp(-step_size / mom_ts)).add_(-math.expm1(-step_size / mom_ts), dp)

        # Perform forward gradient step
        state['p_temp'].add_(-step_size, state['v'].data)

        # Copy the data in preparation for the backward step
        p.data.copy_(state['p_temp'].data)

        # Perform the backward step on each parameter group if it is available.
        if self.has_prox:
          self.prox.apply(p, state['backward_step'])

    self.iteration += 1
