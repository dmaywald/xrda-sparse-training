# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 20:29:51 2024

@author: Owner
"""

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

from torch.utils.data import DataLoader, Subset, SubsetRandomSampler


class Bayesian_Optimizer:
    
    def __init__(self, k_folds, max_evals, num_epoch, l1_prox, criterion):
        self.k_folds = k_folds
        self.max_evals = max_evals
        self.num_epoch = num_epoch
        self.criterion = criterion
        self.l1_prox = l1_prox
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        
    def train(model, self, train_loader, epoch, opt):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
          
          
          inputs = Variable(data).to(self.device)
          labels = Variable(target).to(self.device)
          
        # zero the parameter gradients
          opt.zero_grad()
    
        # forward + backward + optimize
          outputs = model(inputs)
          loss = self.criterion(outputs, labels)
    
          loss.backward()
          opt.step()
    
                
    def objective_function(params, self, train_func, model, params_specs):
        """
        Parameters
        ----------
        params : dict
            Dictionary of Parameter space used in hyperopt 
.
        Returns
        -------
        evaluation of objective function given params
    
        """
        # Initialize the k-fold cross validation
        kf = KFold(n_splits= self.k_folds, shuffle=True)
        
        # Loop through each fold
        for fold, (train_idx, test_idx) in enumerate(kf.split(self.trainset)):
        
            # Define the data loaders for the current fold
            train_loader = DataLoader(
                dataset=self.trainset,
                batch_size= self.batch_size,
                sampler=torch.utils.data.SubsetRandomSampler(train_idx),
            )
            test_loader = DataLoader(
                dataset=self.trainset,
                batch_size= self.batch_size,
                sampler=torch.utils.data.SubsetRandomSampler(test_idx),
            )
            
            # init_lr space has support [1,2] with heavyside 1. Change it to support [0,1] with heaviside 1
            init_lr = 1 - (params['init_lr'] - 1) 
            lam = params['lam'] 
            av_param = params['av_param']
            
            training_specs = params_specs(
                step_size=init_lr, mom_ts= params['mom_ts'], b_mom_ts= params['b_mom_ts'],
                weight_decay= params['weight_decay'], av_param=av_param)
            
            # Initialize the optimizer
            optimizer = xRDA(model.parameters(), it_specs=training_specs,
                             prox= self.l1_prox(lam=lam, maximum_factor=500))
        
            # Train the model on the current fold
            for epoch in range(self.num_epoch):
                train_func(model, self.device, train_loader, optimizer, epoch)
        
            # Evaluate the model on the test set
            model.eval()
            
            # DECISION: FOR OBJECTIVE FUNCTION, RETURN LOSS OR 1 - ACCURACY?
            test_loss = []
            # correct = []
            
            with torch.no_grad():
                for data, target in test_loader:
                    inputs, labels = data.to(self.device), target.to(self.device)
                    outputs = model(inputs)
                    test_loss.append(self.criterion(outputs, labels))
                    # _, predicted = torch.max(outputs.data, 1)
                    
                    # correct.append((predicted == labels).sum())
        
            test_loss = min(test_loss)
            # accuracy = 100.0 * max(correct) / len(test_loader.dataset)
        return {'loss': test_loss, 'params': params, 'status': STATUS_OK}
    
    
    def bayesian_optimizer(self, obj_function, space):
        best_params = fmin(fn = obj_function, space = space, algo= tpe.suggest, max_evals= self.max_evals, trials = Trials())
        return best_params
        

# if __name__ == '__main__':
#     K_FOLDS = 3
#     batch_size = 128
#     max_evals = 5
#     num_epoch = 2
#     subset_Data = 2048
    
#     if subset_Data is not None:
#         transform_train = transforms.Compose(
#             [transforms.RandomCrop(28, padding=4),
#              # transforms.RandomHorizontalFlip(), # I don't think a horizontal flip is appropriate for the MNIST dataset
#              transforms.ToTensor()])
#         mnist_train_dataset = torchvision.datasets.MNIST(root='./', train=True, download=True, transform=transform_train)
        
#         # Define the desired subset size
#         subset_train_size = subset_Data
        
#         # Create a subset of the MNIST dataset for analysis purposes
#         subset_train_indices = torch.randperm(len(mnist_train_dataset))[:subset_train_size]
        
#         trainset = Subset(mnist_train_dataset, subset_train_indices)
             
        
#     if subset_Data is None:
#         transform_train = transforms.Compose(
    
#             [transforms.RandomCrop(28, padding=4),
#              # transforms.RandomHorizontalFlip(), # I don't think a horizontal flip is appropriate for the MNIST dataset
#              transforms.ToTensor()])
        
#         trainset = torchvision.datasets.MNIST(root='./', train=True,
#                                               download=True, transform=transform_train)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     conv_net = mnist_resnet18(num_classes = 10).to(device)
      
#     criterion = nn.CrossEntropyLoss()
    
#     best_params = Bayesian_Optimizer.bayesian_optimizer(Bayesian_Optimizer.objective_function, ResNetParamSpace)