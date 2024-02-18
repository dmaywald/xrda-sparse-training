from hyperopt import tpe
from hyperopt import STATUS_OK
from hyperopt import Trials
from hyperopt import hp
from hyperopt import fmin

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import math
import numpy as np
from scipy import stats
from functools import partial
# import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler

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
      return math.sqrt(running_av.shape[1] * running_av.shape[2] * running_av.shape[3])* self.maximum_factor / (1.0 + (self.maximum_factor - 1.0)*(running_av / maximum))
    elif maximum > 0 and len(running_av.shape) == 4 or len(running_av.shape) == 2:
      return self.maximum_factor / (1.0 + (self.maximum_factor - 1.0)*(running_av / maximum))
    else:
      return torch.zeros_like(running_av)

  def register_running_av(self, running_av):
    return

  def reset(self):
    return


class ConstStepFB:

  def __init__(self, step_size):
    self.step = step_size

  def step_size(self, it):
    return self.step

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

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue

        state = self.state[p]
        
        dp = p.grad.data
        dp.add_(weight_decay, p.data)

        if len(state) == 0:
          if self.has_prox:
            state['backward_step'] = self.prox.get_zero_params(p)
            state['running_av'] = self.prox.get_zero_params(p)
          state['p_temp'] = torch.clone(p).detach()
          state['v'] = torch.zeros_like(p.data)

        if self.has_prox:
          running_av = self.prox.get_running_av(p)
          state['running_av'].mul_(math.exp(-step_size / b_mom_ts)).add_(-math.expm1(-step_size / b_mom_ts), running_av)
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
    

def bayes_train(model, criterion, optimizer, train_loader, epoch, device):
    """
    

    Parameters
    ----------
    model : nn.Module object
        The Neural Network to be trained and tune hyperparameters
    criterion : function
        Loss function of Neural Network (nn.CrossEntropyLoss())
    optimizer : class object
        optimizer used to train model
    train_loader : torch.utils.data.dataloader object
        train loader used to get training data.
    epoch : int
        Tracks current epoch
    device : device
        device object of torch module

    Returns
    -------
    Steps through epoch of training. Returns None.

    """
    model.train()
    i = 0
    for batch_idx, (data, target) in enumerate(train_loader):
      
      
      inputs = Variable(data).to(device)
      labels = Variable(target).to(device)
      
    # zero the parameter gradients
      optimizer.zero_grad()

    # forward + backward + optimize
      outputs = model(inputs)
      loss = criterion(outputs, labels)

      loss.backward()
      optimizer.step()
      i += 1
      # print(i)


def bayes_objective_function(params, model, model_type, criterion, train_func, k_folds, num_epoch, trainset, batch_size, device, mode = "Normal"):
    """
    
    
    
    Runtime Warning:
        
        Runtime is O(len(trainset)/batch_size * k_folds * num_epochs)

    Parameters
    ----------
    params : dict
        Dictionary of Parameter space used in hyperopt. Instance of space object in bayes_optimizer
    model : nn.Module object
        The Neural Network to be trained and tune hyperparameters
    model_type : list
        list containing string. Options: 'resnet', 'vgg', 'densenet'
    criterion : function
        Loss function of Neural Network (nn.CrossEntropyLoss())
    train_func : function
        training function that trains model given criterion, optimizer, and train_loader (train())
    k_folds : int
        number of k-fold cross validation used to train model (k >= 2)
        set k_folds = 1 for no folds
    num_epoch : int
        number of epochs (iterations through trainset) to train model for evaluation of objective function given params
    trainset: torch.dataset
        Data set used to train model
    batch_size : int
        batch size used for training model
    device : device
        device object of torch module
    mode : string, optional
        Specify mode of model ('kernel', 'channel', or 'Normal'). The default is 'Normal' i.e. 'unstructured'.
    Returns
    -------
    dict
    
    Trains model with paramaters specified and returns 
        {loss: to be minimized by bayesian optimizer,
         params: hyperparameters calculated at current step of bayesian optimization,
         status: STATUS_OK object of hp module}
    """
    
    
    if k_folds >= 2:
        # Initialize the k-fold cross validation
        kf = KFold(n_splits= k_folds, shuffle=True)
        num_steps = (np.ceil((k_folds - 1)*len(trainset.indices) / (k_folds*batch_size)) * num_epoch)
        train_test_set = kf.split(trainset)
        
    else:
        # WORKING ON THIS RIGHT NOW!!!!!
        train_test_set = train_test_split(trainset.indices, shuffle=True)
        num_steps = np.ceil(len(trainset.indices)/batch_size)* num_epoch
        
    
    # DECISION: FOR OBJECTIVE FUNCTION, RETURN LOSS OR 1 - ACCURACY?
    test_loss_folds = []
    # accuracy_folds = []
    
    # Loop through each fold
    for fold, (train_idx, test_idx) in enumerate(train_test_set):
        # print(f"Fold {fold + 1}")
        # print("-------")

        # Define the data loaders for the current fold
        train_loader = DataLoader(
            dataset=trainset,
            batch_size= batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        )
        test_loader = DataLoader(
            dataset=trainset,
            batch_size= batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(test_idx),
        )
        
    
        # init_lr space has support [1,2] with heavyside 1. Change it to support [0,1] with heaviside 1
        init_lr = 1 - (params['init_lr'] - 1) 
        lam = params['lam'] 
        av_param = params['av_param']
             
        mom_ts = params['mom_ts']
        b_mom_ts = params['b_mom_ts']
        weight_decay = params['weight_decay']
        
        # Different models require different training specs and different prox        

        if model_type == 'resnet':
            training_specs = IterationSpecs(step_size=init_lr, mom_ts=mom_ts,
                                            b_mom_ts=b_mom_ts, weight_decay=weight_decay, av_param=av_param)
            
            
        if model_type == 'vgg' or model_type == 'densenet':

            training_specs = CosineSpecs(max_iter= num_steps,
                init_step_size= init_lr, mom_ts=mom_ts, b_mom_ts=b_mom_ts, weight_decay=weight_decay)
            
        # Initialize the optimizer
        optimizer = xRDA(model.parameters(), it_specs=training_specs,
                         prox= l1_prox(lam=lam, maximum_factor=500, mode= mode))
        
            
        
        # Train the model on the current fold
        for epoch in range(num_epoch):
            # print(epoch)
            train_func(model, criterion, optimizer, train_loader, epoch, device)
    
        # Evaluate the model on the test set
        model.eval()
        
        test_loss = 0
        # correct = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                inputs, labels = data.to(device), target.to(device)
                outputs = model(inputs)
                # implement sparsity measure into test_loss in Bayesian Optimization
                test_loss += criterion(outputs, labels)
                # _, predicted = torch.max(outputs.data, 1)
                # correct += append((predicted == labels).sum())
    
        test_loss /= len(test_loader)
        # accuracy = 100.0 * correct / len(test_loader.dataset)
        test_loss_folds.append(test_loss)
        # accuracy_folds.append(accuracy)
    loss_out = min(test_loss_folds)
    # loss_out = 1 - max(accuracy_folds)
    return {'loss': loss_out, 'params': params, 'status': STATUS_OK}


def bayes_optimizer(space, max_evals, model, model_type, criterion, k_folds, num_epoch, trainset, batch_size, device, mode = "Normal"):
    """
    
    
    
    Runtime Warning:
        
        Runtime is O(len(trainset)/batch_size * k_folds * num_epochs * max_evals)

    
    Parameters
    ----------
    space : dict
        Dictionary of hp function objects defining the parameter space for bayesian optimization
    max_evals : int
        Max evaluations in search of bayesian optimization.
    model : nn.Module object
        The Neural Network to be trained and tune hyperparameters
    model_type : list
        list containing string. Options: 'resnet', 'vgg', 'densenet'
    criterion : function
        Loss function of Neural Network (nn.CrossEntropyLoss())
    k_folds : int
        number of k-fold cross validation used to train model (k >= 2)
        set k_folds = 1 for no folds
    num_epoch : int
        number of epochs (iterations through trainset) to train model for evaluation of objective function given params
    trainset: torch.dataset
        Data set used to train model
    batch_size : int
        batch size used for training model
    device : device
        device object of torch module
    mode : string, optional
        Specify mode of model ('kernel', 'channel', or 'Normal'). The default is 'Normal' i.e. 'unstructured'.

    Returns
    -------
    best_params : dict
        dictionary of paramater values calculated by bayesian optimization

    """
    # wrapper function of objective function with given bayesian optimizer parameters
    obj_function = partial(bayes_objective_function, model = model, model_type = model_type, criterion = criterion,
                           train_func = bayes_train, k_folds = k_folds, num_epoch = num_epoch,
                           trainset = trainset, batch_size = batch_size, device = device, mode = mode)
    
    # call fmin on objective function to calculate best paramaters
    # return_argmin = False to return paramater values instead of list indices for parameters specified through list
    best_params = fmin(fn = obj_function, space = space, algo= tpe.suggest,
                       max_evals= max_evals, trials = Trials(), return_argmin=False)
    return best_params
            
