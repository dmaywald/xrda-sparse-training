# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:42:37 2024

@author: Owner
"""

from hyperopt import tpe
from hyperopt import STATUS_OK
from hyperopt import Trials
from hyperopt import hp
from hyperopt import fmin

from sklearn.model_selection import KFold

import os
import time
import numpy as np
from scipy import stats
# import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from models import mnist_resnet18
from training_algorithms import xRDA
from regularization import l1_prox
from training_algorithms import IterationSpecs
from utils import test_accuracy
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler


K_FOLDS = 3
batch_size = 128
max_evals = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conv_net = mnist_resnet18(num_classes = 10).to(device)
  
criterion = nn.CrossEntropyLoss()

subset_Data = True

expected_lam = 1.3e-6
prob_max_lam = 5e-3
prob_max = .01

space = {
    'lam' : hp.lognormal('lam', np.log(expected_lam),
                             (np.log(prob_max_lam/expected_lam)/stats.norm.ppf(1-prob_max))),
    'init_lr' : hp.loguniform('init_lr', 0, np.log(2)), # this is on a scale of 1-2, remember to subtract 1 in parameter call
    'av_param' : hp.uniform('av_param', 0,1),
    'mom_ts' : hp.choice('mom_ts', [9.5]),
    'b_mom_ts' : hp.choice('b_mom_ts', [9.5]),
    'weight_decay' : hp.choice('weight_decay', [5e-4])
    
    }

if subset_Data:
    transform_train = transforms.Compose(
        [transforms.RandomCrop(28, padding=4),
         # transforms.RandomHorizontalFlip(), # I don't think a horizontal flip is appropriate for the MNIST dataset
         transforms.ToTensor()])
    mnist_train_dataset = torchvision.datasets.MNIST(root='./', train=True, download=True, transform=transform_train)
    
    # Define the desired subset size
    subset_train_size = 2048
    
    # Create a subset of the MNIST dataset for analysis purposes
    subset_train_indices = torch.randperm(len(mnist_train_dataset))[:subset_train_size]
    
    trainset = Subset(mnist_train_dataset, subset_train_indices)
         
    
if not subset_Data:
    transform_train = transforms.Compose(

        [transforms.RandomCrop(28, padding=4),
         # transforms.RandomHorizontalFlip(), # I don't think a horizontal flip is appropriate for the MNIST dataset
         transforms.ToTensor()])

    transform_val = transforms.Compose(

        [transforms.ToTensor()])
    
    trainset = torchvision.datasets.MNIST(root='./', train=True,
                                          download=True, transform=transform_train)



def train(model, device, train_loader, optimizer, epoch):
    model.train()
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

            
def obj_function(params, k_folds = K_FOLDS, num_epoch = 2):
    """
    Parameters
    ----------
    params : dict
        Dictionary of Parameter space used in hyperopt 
    k_folds : int, optional
        Number of cross-validation folds used to calculate objective function. The default is K_FOLDS.
    num_epoch : int, optional
        Number of epochs used to train NN on each k-fold
    Returns
    -------
    evaluation of objective function given params

    """
    # Initialize the k-fold cross validation
    kf = KFold(n_splits=k_folds, shuffle=True)
    
    # Loop through each fold
    for fold, (train_idx, test_idx) in enumerate(kf.split(trainset)):
    
        # Define the data loaders for the current fold
        train_loader = DataLoader(
            dataset=trainset,
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        )
        test_loader = DataLoader(
            dataset=trainset,
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(test_idx),
        )
        
        # init_lr space has support [1,2] with heavyside 1. Change it to support [0,1] with heaviside 1
        init_lr = 1 - (params['init_lr'] - 1) 
        lam = params['lam'] 
        av_param = params['av_param']
        
        training_specs = IterationSpecs(
            step_size=init_lr, mom_ts= params['mom_ts'], b_mom_ts= params['b_mom_ts'],
            weight_decay= params['weight_decay'], av_param=av_param)
        # Initialize the optimizer
        
        optimizer = xRDA(conv_net.parameters(), it_specs=training_specs,
                         prox=l1_prox(lam=lam, maximum_factor=500))
    
        # Train the model on the current fold
        for epoch in range(num_epoch):
            train(conv_net, device, train_loader, optimizer, epoch)
    
        # Evaluate the model on the test set
        conv_net.eval()
        
        # DECISION: FOR OBJECTIVE FUNCTION, RETURN LOSS OR 1 - ACCURACY?
        test_loss = []
        # correct = []
        
        with torch.no_grad():
            for data, target in test_loader:
                inputs, labels = data.to(device), target.to(device)
                outputs = conv_net(inputs)
                test_loss.append(criterion(outputs, labels))
                # _, predicted = torch.max(outputs.data, 1)
                
                # correct.append((predicted == labels).sum())
    
        test_loss = min(test_loss)
        # accuracy = 100.0 * max(correct) / len(test_loader.dataset)
    return {'loss': test_loss, 'params': params, 'weights': conv_net.parameters,'status': STATUS_OK}


def bayesian_optimizer():
    best_params = fmin(fn = obj_function, space = space, algo= tpe.suggest, max_evals= max_evals, trials = Trials())
    return best_params
        

if __name__ == '__main__':
    best_params = bayesian_optimizer()