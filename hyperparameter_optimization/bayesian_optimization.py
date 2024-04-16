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
import os
import copy

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
    
  def get_type(self):
    return "IterationSpecs"
    

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

  def get_type(self):
    return "CosineSpecs"


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
        dp.add_(weight_decay, p.data) # originally: dp.add_(weight_decay, p.data)

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


def bayes_objective_function(params, model, it_specs, criterion, train_func, k_folds,
                             num_epoch, trainset, batch_size, device, subset_Data, mode = "normal",
                             sparse_scale = 1, check_point = None, max_iter = None, epoch_updates = None):
    """
    
    
    
    Runtime Warning:
        
        Runtime is O((len(trainset)/batch_size) * k_folds * num_epochs)

    Parameters
    ----------
    params : dict
        Dictionary of Parameter space used in hyperopt. Instance of space object in bayes_optimizer
    model : nn.Module object
        The Neural Network to be trained and tune hyperparameters
    it_specs: str
        The iteration specs for the optimizer of the model given. ("CosineSpecs"/"IterationSpecs")
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
    subset_Data : int
        size of random subset of training data. Set to None for no subsetting.
    device : device
        device object of torch module
        
    mode : string, optional
        Specify mode of model ('kernel', 'channel', or 'normal'). The default is 'normal' i.e. 'unstructured'.
        
    check_point : dict, optional
        Initial parameterization of model and optimizer (if desired).


    Returns
    -------
    dict
    
    Trains model with paramaters specified and returns 
        {loss: to be minimized by bayesian optimizer,
         params: hyperparameters calculated at current step of bayesian optimization,
         status: STATUS_OK object of hp module}
    """
    if check_point['model_state_dict'] is not None:
        model.load_state_dict(check_point['model_state_dict'])
        # print(model.state_dict()['classifier.bias'])
    
    
    num_params = sum([x.numel() for x in list(model.parameters())])
    
    if subset_Data is not None:

        # Define the desired subset size
        subset_train_size = subset_Data

        
        # Create a subset of the training dataset 
        subset_train_indices = torch.randperm(len(trainset))[:subset_train_size]

        
        
        trainset_sub = Subset(trainset, subset_train_indices)
        # print("Train indices: ")
        # print(subset_train_indices)
        del trainset # delete bloat (still carried by wrapper function, each subset is unique per bayes_opt iteration)
        
    if subset_Data is None:
        # rename trainset arbitrarily to trainset_sub so same code can be used
        trainset_sub = trainset
        del trainset # delete bloat (still carried by wrapper function)
         
    
    if k_folds >= 2:
        # Initialize the k-fold cross validation
        kf = KFold(n_splits= k_folds, shuffle=True)
        num_steps = (np.ceil((k_folds - 1)*len(trainset_sub) / (k_folds*batch_size)) * num_epoch)
        train_test_set = kf.split(trainset_sub)
        
    else:
        # If k_folds = 1, then do train_test_split of train size .8        
        train_test_set = [train_test_split(np.arange(len(trainset_sub)), test_size=.2, shuffle=True)] # List of tuple of indices to enumerate through (only 1)
        num_steps = np.ceil(.8*len(trainset_sub)/batch_size)* num_epoch
        
    
    # DECISION: FOR OBJECTIVE FUNCTION, RETURN LOSS OR 1 - ACCURACY?
    test_loss_folds = []
    sparsity_folds = []
    # accuracy_folds = []
    
    # Loop through each fold
    # Parallelize with Dask?
    for fold, (train_idx, test_idx) in enumerate(train_test_set):
        # print(f"Fold {fold + 1}")
        # print("-------")
        # print(train_idx)
        # print(test_idx)
        # Define the data loaders for the current fold
        train_loader = DataLoader(
            dataset=trainset_sub,
            batch_size= batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        )
        test_loader = DataLoader(
            dataset=trainset_sub,
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
        if it_specs == 'IterationSpecs':
            training_specs = IterationSpecs(step_size=init_lr, mom_ts=mom_ts,
                                            b_mom_ts=b_mom_ts, weight_decay=weight_decay, av_param=av_param)
            lr = init_lr
            if epoch_updates is None: # if epoch_updates is not specified, assume epoch_updates are similar to a full training scheme
                epoch_updates = list(dict.fromkeys(np.floor(np.linspace(1,num_epoch,5))))[2:]
            

        if it_specs == "CosineSpecs":
            if max_iter is None:
                training_specs = CosineSpecs(max_iter= num_steps,
                    init_step_size= init_lr, mom_ts=mom_ts, b_mom_ts=b_mom_ts, weight_decay=weight_decay)
            if max_iter is not None:
                training_specs = CosineSpecs(max_iter= max_iter,
                    init_step_size= init_lr, mom_ts=mom_ts, b_mom_ts=b_mom_ts, weight_decay=weight_decay)
        
        if check_point['optimizer_state_dict'] is None:
        # Initialize the optimizer
            optimizer = xRDA(model.parameters(), it_specs=training_specs,
                             prox= l1_prox(lam=lam, maximum_factor=500, mode= mode))
            
        if check_point['optimizer_state_dict'] is not None:     
            # Initialize the optimizer
            optimizer = xRDA(model.parameters(), it_specs=training_specs,
                             prox= l1_prox(lam=lam, maximum_factor=500, mode= mode))
            # load in check point
            optimizer.load_state_dict(check_point['optimizer_state_dict'])
            optimizer.set_it_specs(new_specs= training_specs)
                    
            
        
        # Train the model on the current fold
        for epoch in range(num_epoch):
            print('Epoch:', epoch+1)
            # print(optimizer.iteration)
            if epoch_updates is not None and it_specs == 'IterationSpecs':
                if epoch+1 in epoch_updates:
                    lr /= 2
                    training_specs.set_step_size(lr)
                    av_param = 1.0 - (1.0 - av_param) / 2.0
                    training_specs.set_av_param(av_param)
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
        
        ## count zero or zero-like? Without zero-like, torch.nonzero() and torch.count_nonzero() require
        ## ... a float of order 1e-46 to be considered 0
        sparsity_folds.append( sum(torch.count_nonzero(x) for x in list(model.parameters())))
        # print(test_loss)
        # print(sparsity_folds)
    sparsity = min(sparsity_folds)   
    print("Params: ")
    print(params)
    print("Percent Non-zero: "+str(100*sparsity.item()/num_params)+"%")
    print("Test Loss: "+str( min(test_loss_folds).item()))
    print("")
    # Alternatively
        # See if there is a difference between this and the other count zero method. 
        # tol = 1e-8
        # sparsity = sum([len(x[torch.logical_or(x<=-tol, x>=tol)]) for x in list(model.parameters())])
    
    # loss of random parameters is around 2, so sparity/num_params is of similar scale
    loss_out = min(test_loss_folds) + sparse_scale*(sparsity/num_params)
    # loss_out = 1 - max(accuracy_folds)
    return {'loss': loss_out, 'params': params,
            'status': STATUS_OK,
            'test_loss_val': min(test_loss_folds).item(),
            'perc_non_zero_val': sparsity.item()/num_params,
            'sparse_scale': sparse_scale}


def bayes_optimizer(space, max_evals, model, it_specs, criterion, k_folds, num_epoch,
                    trainset, batch_size, subset_Data, device, mode = "normal",
                    sparse_scale = 1, check_point = None, max_iter = None, epoch_updates = None):
    """
    
    
    
    Runtime Warning:
        
        Runtime is O((len(trainset)/batch_size) * k_folds * num_epochs * max_evals)

    
    Parameters
    ----------
    space : dict
        Dictionary of hp function objects defining the parameter space for bayesian optimization
    max_evals : int
        Max evaluations in search of bayesian optimization.
    model : nn.Module object
        The Neural Network to be trained and tune hyperparameters
    it_specs: str
        The iteration specs for the optimizer of the model given. ("CosineSpecs"/"IterationSpecs")
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
    subset_Data : int
        size of random subset of training data. Set to None for no subsetting.
    device : device
        device object of torch module
    mode : string, optional
        Specify mode of model ('kernel', 'channel', or 'normal'). The default is 'normal' i.e. 'unstructured'.

    Returns
    -------
    best_params : dict
        dictionary of paramater values calculated by bayesian optimization
        
    bayes_trials: dict
        dictionary of trials object providing optimization insight

    """
    # wrapper function of objective function with given bayesian optimizer parameters
    obj_function = partial(bayes_objective_function,
                           model = model,
                           it_specs = it_specs,
                           criterion = criterion,
                           train_func = bayes_train,
                           k_folds = k_folds,
                           num_epoch = num_epoch,
                           trainset = trainset,
                           batch_size = batch_size,
                           subset_Data = subset_Data,
                           device = device,
                           mode = mode,
                           sparse_scale = sparse_scale,
                           check_point = check_point, 
                           max_iter = max_iter, 
                           epoch_updates = epoch_updates)
    
    bayes_trials = Trials()
    
    # call fmin on objective function to calculate best paramaters
    # return_argmin = False to return paramater values instead of list indices for parameters specified through list
    best_params = fmin(fn = obj_function, space = space, algo= tpe.suggest,
                       max_evals= max_evals, trials = bayes_trials, return_argmin=False)
    
    return best_params, bayes_trials


# A tune parameters function will need to be a function of:
    # params_output_file
    # trials_output_file
    # transform_train
    # trainset (possibly just string of "mnist"/"cifar10"/"cifar100")
    # model
    # model_type ('resnet'/'vgg'/'densenet')
    # param_space
    # train_batch_size
    # subset_Data
    # k_folds
    # max_evals
    # num_epoch
            
def tune_parameters(model, it_specs, mode, space, params_output_file, trials_output_file, data, transform_train,
                    train_batch_size = 128, subset_Data = None, k_folds = 1, num_epoch = 10, max_evals = 40, sparse_scale = 1,
                    max_iter = None, epoch_updates = None, check_point = None):
    """
    
    
    Runtime Warning:
        
        Runtime is O((len(trainset)/train_batch_size) * k_folds * num_epochs * max_evals)
        
        
    Parameters
    ----------
    model : nn.Module object
        The Neural Network to be trained 
    it_specs: str
        The iteration specs for the optimizer of the model given. ("CosineSpecs"/"IterationSpecs")
    mode : string, optional
        Specify mode of model ('kernel', 'channel', or 'normal'). The default is 'normal' i.e. 'unstructured'.
    space : dict
        Dictionary of hp function objects defining the parameter space for bayesian optimization
    params_output_file : string
        Path and Name of tuned parameters. Set to None if parameters should not be saved
    trials_output_file : string
        path and name of bayes trials data. Set to None if trials should not be saved
    data : string
        Specify which dataset is being used. 
            Options: 'mnist', 'cifar10', 'cifar100' 
    transform_train : torchvision.transform object
        transformation made to training data
    train_batch_size : int, optional
        Batch size for training. The default is 128.
    subset_Data : int, optional
        Size of random subset of training data. Set to None for no subsetting. The default is None.
    k_folds : int, optional
        Number of cross-validation folds used to train model and evaluate bayes_objective function.
        Set to 1 to use an 80/20 train-test split instead. The default is 1.
    num_epoch : int, optional
        Number of epochs to train model for on each fold (or on 80% split of training data) when evaluating
        bayes_objective function. The default is 10.
    max_evals : int, optional
        Number of iterations of bayes optimization. The default is 40.


    Returns
    -------
    After tuning the tuning/hyperparameters by performing bayes optimization, function will save (optional) and return best_params
    and trials
    
    best_params : dict
        dictionary of paramater values calculated by bayesian optimization
        
    trials : dict
        dictionary of trials object providing optimization insight

    """
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if check_point is None: # if check_point not specified, initial model and optimizer is check_point
        check_point = {
            'model_state_dict': copy.deepcopy(model.state_dict()),
            'optimizer_state_dict': None
            }
        
    
    if data == "mnist":
        trainset = torchvision.datasets.MNIST(root='./', train=True, download=True, transform=transform_train)
        
    if data == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
        
    if data == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root='./', train=True, download=True, transform=transform_train)
    
    criterion = nn.CrossEntropyLoss()
    
    # If params_output_file has subdirectories, make sure those subdirectories exist  
    if params_output_file is not None:
        str_list = params_output_file.split("/")[:-1]
        if len(str_list) > 0:  
            temp_str = ''
            for idx in range(len(str_list)):
                temp_str = temp_str + str_list[idx]+'/'
                if not os.path.exists(temp_str):
                    os.mkdir(temp_str)
                    
    # If trials_output_file has subdirectories, make sure those subdirectories exist  
    if trials_output_file is not None:
        str_list = trials_output_file.split("/")[:-1]
        if len(str_list) > 0:  
            temp_str = ''
            for idx in range(len(str_list)):
                temp_str = temp_str + str_list[idx]+'/'
                if not os.path.exists(temp_str):
                    os.mkdir(temp_str)
        
    best_params, trials = bayes_optimizer(space=space, max_evals=max_evals, model=model, it_specs = it_specs,
                                          criterion=criterion,k_folds=k_folds, num_epoch=num_epoch, trainset=trainset,
                                          batch_size=train_batch_size, subset_Data=subset_Data, device=device, mode=mode, sparse_scale = sparse_scale,
                                          check_point= check_point, max_iter= max_iter, epoch_updates= epoch_updates)
    
    if params_output_file is not None:
        torch.save(best_params, params_output_file)   
    
    if trials_output_file is not None:        
        torch.save(trials, trials_output_file)
        
    return best_params, trials