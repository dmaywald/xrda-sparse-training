# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 15:57:20 2024

@author: Devon
"""

import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from models import mnist_vgg16_bn
from training_algorithms import xRDA
from regularization import  l1_prox
from training_algorithms import CosineSpecs
from utils import test_accuracy, progress_dataframe
from hyperparameter_optimization import tune_parameters
from hyperparameter_optimization import VggParamSpace

if __name__ == '__main__':
    t0 = time.time()

    init_params = {
        'init_lr': 1.0, #1.0 by default
        'lam': 1e-6, #1e-6 by default
        'av_param': 0.0, # 0.0 by default
        'mom_ts': 9.5, # 9.5 by default
        'b_mom_ts': 9.5, # 9.5 by default
        'weight_decay': 5e-4 # 5e-4 by default
        }
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = mnist_vgg16_bn(num_classes=10).to(device)
    
    train_batch_size = 128
    num_epoch = 30
    subset_Data = None
    
    if subset_Data is not None:
        max_iter = math.ceil(subset_Data/train_batch_size) * num_epoch
        
    if subset_Data is None:
        max_iter = math.ceil(60000/train_batch_size) * num_epoch # 60000 is length of mnist data
        
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
          # transforms.RandomHorizontalFlip(), # I don't think a horizontal flip is appropriate for the MNIST dataset
          transforms.ToTensor()])
    
    transform_val = transforms.Compose(
        [transforms.Pad(2), # Padding to a 32 x 32 image so the output dimensions after convolution fits 
         transforms.ToTensor()])
    
    training_specs = CosineSpecs(max_iter=max_iter, 
                                 init_step_size= 2 - init_params['init_lr'],
                                 mom_ts=init_params['mom_ts'],
                                 b_mom_ts=init_params['b_mom_ts'],
                                 weight_decay=init_params['weight_decay'])
    
    optimizer = xRDA(model.parameters(), it_specs=training_specs, 
                     prox=l1_prox(lam=init_params['lam'], maximum_factor=500, mode='kernel'))
    
    
    model_output_file = 'results/model_data/vgg/mnist_vgg16_kernel_sparse_model_init_params_.dat'
    progress_data_output_file = 'results/progress_data/vgg/mnist_vgg16_kernel_sparse_training_progress_init_params.csv'
    
    # model_output_file = None
    # progress_data_output_file = None
    
    # Use the following below to visualize necessary 'num_epoch' for bayesian optimization
    progress_df = progress_dataframe(model=model,
                                     params=init_params,
                                     model_output_file=model_output_file,
                                     progress_data_output_file = progress_data_output_file,
                                     data = 'mnist',
                                     transform_train = transform_train,
                                     transform_val = transform_val,
                                     optimizer = optimizer,
                                     training_specs = training_specs,
                                     train_batch_size = train_batch_size,
                                     subset_Data = subset_Data,
                                     num_epoch = num_epoch)
    # From the dataframe above:
    #     Training accuracy reaches maximum 99% after 10 epochs on full 
    #     Testing accuracy reaches 99% after 15 epochs and 99.5% after 20 epochs
    #     Sparsity is 97% after 10 epochs and 99% after 20 epochs
    #     Cross Entropy Loss is minimized to [.004, .24] after 2 epochs
    
    
    params_output_file = 'results/bayes_opt_params/vgg/mnist_vgg16_kernel_sparse_model_bayes_params.dat'
    trials_output_file = 'results/bayes_opt_params/vgg/mnist_vgg16_kernel_sparse_model_bayes_trials.dat'
    
    # params_output_file = None
    # trials_output_file = None
    model = mnist_vgg16_bn(num_classes=10).to(device)
    # Experimentally determined 1e-4 is too large for lambda
    space = VggParamSpace(expected_lam = 1e-6, max_lam = 1e-4, prob_max_lam = .01,
                  init_lr_low = 0, init_lr_high = math.log(2), av_low = 0, av_high = 1,
                  mom_ts = 9.5, b_mom_ts = 9.5)
    
    best_params, trials = tune_parameters(model=model,
                                          model_type = 'vgg',
                                          mode = 'kernel',
                                          space = space,
                                          params_output_file = params_output_file,
                                          trials_output_file = trials_output_file,
                                          data = 'mnist',
                                          transform_train = transform_train,
                                          train_batch_size=128,
                                          subset_Data= 2**13,
                                          k_folds=1,
                                          num_epoch=15,
                                          max_evals=100)
    
    best_params = torch.load(params_output_file)
    trials = torch.load(trials_output_file)
    
    
    model = mnist_vgg16_bn(num_classes=10).to(device)
    
    
    model_output_file = 'results/model_data/vgg/mnist_vgg16_kernel_sparse_model_bayes_params.dat'
    progress_data_output_file = 'results/progress_data/vgg/mnist_vgg16_kernel_sparse_training_progress_bayes_params.csv'
    
    # model_output_file = None
    # progress_data_output_file = None
    
    training_specs = CosineSpecs(max_iter=max_iter, 
                                  init_step_size= 2 - best_params['init_lr'],
                                  mom_ts=best_params['mom_ts'],
                                  b_mom_ts=best_params['b_mom_ts'],
                                  weight_decay=best_params['weight_decay'])
    
    optimizer = xRDA(model.parameters(), it_specs=training_specs, 
                      prox=l1_prox(lam=best_params['lam'], maximum_factor=500))
    
    progress_df = progress_dataframe(model=model,
                                      params=best_params,
                                      model_output_file=model_output_file,
                                      progress_data_output_file = progress_data_output_file,
                                      data = 'mnist',
                                      transform_train = transform_train,
                                      transform_val = transform_val,
                                      optimizer = optimizer,
                                      training_specs = training_specs,
                                      train_batch_size=128,
                                      subset_Data=None,
                                      num_epoch=30)
    
    print(time.time() - t0)