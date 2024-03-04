# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 18:23:49 2024

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

def main():
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  output_file = 'results/model_data/vgg/mnist_vgg16_sparse_unstructured_model.dat'
  batch_size = 128
  epoch_count = 60

  # transform_train = transforms.Compose(

  #     [transforms.RandomCrop(32, padding=4),
  #      transforms.RandomHorizontalFlip(),
  #      transforms.ToTensor(),
  #      transforms.Normalize((0.4914, 0.4822, 0.4465),
  #                           (0.2023, 0.1994, 0.2010))])

  # transform_val = transforms.Compose(

  #     [transforms.ToTensor(),
  #      transforms.Normalize((0.4914, 0.4822, 0.4465),
  #                           (0.2023, 0.1994, 0.2010))])
  
  transform_train = transforms.Compose(

      [transforms.RandomCrop(32, padding=4), # Here we require 32 instead of 28, otherwise once we get to Avg pooling the output and input size do not match
       # transforms.RandomHorizontalFlip(), # I don't think a horizontal flip is appropriate for the MNIST data
       transforms.ToTensor()])

  transform_val = transforms.Compose(
      [transforms.Pad(2), # Padding to a 32 x 32 image so the output dimensions after convolution fits 
       transforms.ToTensor()])


  trainset = torchvision.datasets.MNIST(root='./', train=True,
                                          download=True, transform=transform_train)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=4)

  testset = torchvision.datasets.MNIST(root='./', train=False,
                                         download=True, transform=transform_val)
  testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                           shuffle=False, num_workers=2)

  if device.type == 'cpu':
    conv_net = mnist_vgg16_bn(num_classes=10).cpu()
  else:
    conv_net = mnist_vgg16_bn(num_classes=10).cuda()
    
  conv_net.train()
  criterion = nn.CrossEntropyLoss()

  init_lr = 1.0
  lam = 1e-6
  av_param = 0.0
  training_specs = CosineSpecs(max_iter=math.ceil(sum(1 for _ in trainloader.batch_sampler)) * epoch_count,
      init_step_size=init_lr, mom_ts=9.5, b_mom_ts=9.5, weight_decay=5e-4)
  optimizer = xRDA(conv_net.parameters(), it_specs=training_specs,
                   prox=l1_prox(lam=lam, maximum_factor=500))

  lr = init_lr
  prev_train_acc = 0
  prev_sparsity = 0
  for epoch in range(epoch_count):
    total = 0
    correct = 0
    for data in trainloader:
      # get the inputs
      inputs, labels = data
      if device.type == 'cpu':
        inputs = Variable(inputs).cpu()
        labels = Variable(labels).cpu()
      else:
          inputs = Variable(inputs).cuda()
          labels = Variable(labels).cuda()

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = conv_net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # Calculate train accuracy
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum()

    train_acc = correct
    sparsity = sum(torch.nonzero(x).size()[0]
                   for x in list(conv_net.parameters()))
    accuracy = 10000 * correct / total
    if device.type == 'cpu':
      t_accuracy = test_accuracy(testloader, conv_net, cuda=False)
    else:
      t_accuracy = test_accuracy(testloader, conv_net, cuda=True)
    print('Epoch:%d %% Training Accuracy: %d.%02d %% Test Accuracy: %d.%02d %% Sparsity: %d' % (epoch + 1,
                                                                                                accuracy / 100, accuracy % 100, t_accuracy / 100, t_accuracy % 100, sparsity))

  # Calculate accuracy and save output.
  final_accuracy = test_accuracy(testloader, conv_net, cuda=True)
  print('Accuracy of the network on the 10000 test images: %d.%02d %%' %
        (final_accuracy / 100, final_accuracy % 100))
  torch.save(conv_net, output_file)


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
    num_epoch = 2
    subset_Data = 2048
    
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
                     prox=l1_prox(lam=init_params['lam'], maximum_factor=500))
    
    
    # model_output_file = 'results/model_data/vgg/mnist_vgg16_sparse_model_init_params_.dat'
    # progress_data_output_file = 'results/progress_data/vgg/mnist_vgg16_sparse_training_progress_init_params_.csv'
    
    model_output_file = None
    progress_data_output_file = None
    
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
                                     train_batch_size=128,
                                     subset_Data=2048,
                                     num_epoch=2)

    # params_output_file = 'results/bayes_opt_params/vgg/mnist_vgg16_sparse_model_bayes_params_small_data.dat'
    # trials_output_file = 'results/bayes_opt_params/vgg/mnist_vgg16_sparse_model_bayes_trials_small_data.dat'
    
    params_output_file = None
    trials_output_file = None
    
    # Still need to figure out a prob_max_lam
    space = VggParamSpace(expected_lam = 1e-6, prob_max_lam = 5e-3, prob_max = .01,
                 init_lr_low = 0, init_lr_high = math.log(2), av_low = 0, av_high = 1,
                 mom_ts = 9.5, b_mom_ts = 9.5)
    
    best_params, trials = tune_parameters(model=model,
                                          model_type = 'vgg',
                                          mode = 'normal',
                                          space = space,
                                          params_output_file = params_output_file,
                                          trials_output_file = trials_output_file,
                                          data = 'mnist',
                                          transform_train = transform_train,
                                          train_batch_size=128,
                                          subset_Data= 2048,
                                          k_folds=1,
                                          num_epoch=2,
                                          max_evals=2)
    
    best_params = torch.load(params_output_file)
    trials = torch.load(trials_output_file)
    
    # model_output_file = 'results/model_data/vgg/mnist_vgg16_sparse_model_bayes_params_small_data.dat'
    # progress_data_output_file = 'results/progress_data/vgg/mnist_vgg16_sparse_training_progress_bayes_params_small_data.csv'
    
    model_output_file = None
    progress_data_output_file = None
    
    training_specs = CosineSpecs(max_iter=max_iter, 
                                 init_step_size= 2 - best_params['init_lr'],
                                 mom_ts=best_params['mom_ts'],
                                 b_mom_ts=best_params['b_mom_ts'],
                                 weight_decay=best_params['weight_decay'])
    
    optimizer = xRDA(model.parameters(), it_specs=training_specs, 
                     prox=l1_prox(lam=init_params['lam'], maximum_factor=500))
    
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
                                     num_epoch=3)
    
    print(time.time() - t0)
