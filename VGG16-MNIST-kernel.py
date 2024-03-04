# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 15:57:20 2024

@author: Devon
"""

# Author: Jonathan Siegel
#
# Tests xRDA training method (an advanced version of RDA augmented with momentum)
# on the VGG-16 model on the CIFAR-10 dataset.

import os
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from models import mnist_vgg16_bn
from training_algorithms import xRDA
from regularization import l1_prox
from training_algorithms import CosineSpecs
from utils import test_accuracy
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
import hyperparameter_optimization as hpo

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# A progress df function will need to be a function of:
    # tuning params
    # model_output_file
    # progress_data_output_file
    # transform_train
    # transform_val
    # model
    # training_specs (or xRDA optimizer with pre-allocated training specs)
    # data (possibly just string of "mnist"/"cifar10"/"cifar100")
    # train_batch_size
    # subset_Data
    # num_epoch
    # epoch_updates "[6, 10, 14, 18, 22, 26, 30, 34, 38, 42]"

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
    
def main():
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  output_file = 'results/model_data/vgg/mnist_vgg16_sparse_model.dat'
  batch_size = 128
  epoch_count = 60 # originally 600


  transform_train = transforms.Compose(

      [transforms.RandomCrop(32, padding=4), # Here we require 32 instead of 28, otherwise once we get to Avg pooling the output and input size do not match
       # transforms.RandomHorizontalFlip(),# I don't think a horizontal flip is appropriate for the MNIST data
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

    
    
  conv_net = mnist_vgg16_bn(num_classes = 10).to(device)
  conv_net.train()
  criterion = nn.CrossEntropyLoss()

  init_lr = 1.0
  lam = 8e-7
  av_param = 0.0
  training_specs = CosineSpecs(max_iter=math.ceil(sum(1 for _ in trainloader.batch_sampler)) * epoch_count,
                                  init_step_size=init_lr, mom_ts=9.5, b_mom_ts=9.5, weight_decay=5e-4)
  optimizer = xRDA(conv_net.parameters(), it_specs=training_specs,
                   prox=l1_prox(lam=lam, maximum_factor=500, mode='kernel'))

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
      # print(loss)
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
  if device.type == 'cpu':
    final_accuracy = test_accuracy(testloader, conv_net, cuda=False)
  else:
    final_accuracy = test_accuracy(testloader, conv_net, cuda=True)
  print('Accuracy of the network on the 10000 test images: %d.%02d %%' %
        (final_accuracy / 100, final_accuracy % 100))
  torch.save(conv_net, output_file)




def progress_dataframe(params, model_output_file, progress_data_output_file, train_batch_size = 128, subset_Data = None, num_epoch = 60):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # If model_output_file has subdirectories, make sure those subdirectories exist  
    if model_output_file is not None:
        str_list = model_output_file.split("/")[:-1]
        if len(str_list) > 0:  
            temp_str = ''
            for idx in range(len(str_list)):
                temp_str = temp_str + str_list[idx]+'/'
                if not os.path.exists(temp_str):
                    os.mkdir(temp_str)
                    
    # If progress_data_output_file has subdirectories, make sure those subdirectories exist            
    if progress_data_output_file is not None:
        str_list = progress_data_output_file.split("/")[:-1]
        if len(str_list) > 0: 
            temp_str = ''
            for idx in range(len(str_list)):
                temp_str = temp_str + str_list[idx]+'/'
                if not os.path.exists(temp_str):
                    os.mkdir(temp_str)
    
    # if subset data is used, load full data and get data subset sizes
    if subset_Data is not None:
        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             # transforms.RandomHorizontalFlip(), # I don't think a horizontal flip is appropriate for the MNIST dataset
             transforms.ToTensor()])
        
        
        transform_val = transforms.Compose(
            [transforms.Pad(2), # Padding to a 32 x 32 image so the output dimensions after convolution fits 
             transforms.ToTensor()])
        
        mnist_train_dataset = torchvision.datasets.MNIST(root='./', train=True, download=True, transform=transform_train)
        mnist_val_dataset = torchvision.datasets.MNIST(root='./', train=False, download=True, transform=transform_train)
        
        # Define the desired subset size
        subset_train_size = subset_Data
        subset_val_size = int(np.ceil(len(mnist_val_dataset) / (len(mnist_train_dataset)/subset_train_size)))
    
        # Create a subset of the MNIST dataset for analysis purposes
        subset_train_indices = torch.randperm(len(mnist_train_dataset))[:subset_train_size]
        subset_val_indices = torch.randperm(len(mnist_val_dataset))[:subset_val_size]
        
        trainset = Subset(mnist_train_dataset, subset_train_indices)
        testset = Subset(mnist_val_dataset, subset_val_indices)    
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                                      shuffle=True, num_workers=4)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                                     shuffle=False, num_workers=2)
        
        
    
    # if subset data is not used, load full data
    if subset_Data is None:
        transform_train = transforms.Compose(
    
            [transforms.RandomCrop(28, padding=4),
             # transforms.RandomHorizontalFlip(), # I don't think a horizontal flip is appropriate for the MNIST dataset
             transforms.ToTensor()])
    
        transform_val = transforms.Compose(
    
            [transforms.ToTensor()])
        
        trainset = torchvision.datasets.MNIST(root='./', train=True,
                                              download=True, transform=transform_train)
    
        testset = torchvision.datasets.MNIST(root='./', train=False,
                                             download=True, transform=transform_val)

        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                                      shuffle=True, num_workers=4)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                                     shuffle=False, num_workers=2)
        
        
        
    conv_net = mnist_vgg16_bn(num_classes = 10).to(device)
      
    conv_net.train()
    criterion = nn.CrossEntropyLoss()
    
    init_lr = 2.0 - params['init_lr'] # Originally 1
    lam = params['lam'] # Originally 8e-7
    av_param = params['av_param']# Originally 0.0
    training_specs = CosineSpecs(max_iter=math.ceil(sum(1 for _ in trainloader.batch_sampler)) * num_epoch,
                                    init_step_size=init_lr, mom_ts=9.5, b_mom_ts=9.5, weight_decay=5e-4)
    optimizer = xRDA(conv_net.parameters(), it_specs=training_specs,
                     prox=l1_prox(lam=lam, maximum_factor=500, mode='kernel'))

    
    lr = init_lr
    prev_train_acc = 0
    prev_sparsity = 0
    num_epoch = num_epoch # originally 500
    init_sparsity = sum([x.numel() for x in list(conv_net.parameters())])
    len_per_epoch = sum(1 for _ in trainloader.batch_sampler)
    
    # DataFrame to keep track of progress, will contain:
    # Epoch, loss per step, Training Accuracy per step, Testing Accuracy per end of Epoch,
    # Sparsity per step, step_size per epoch, mom_ts per epoch, b_mom_ts per epoch,
    # weight_decay per epoch, av_param per epoch

    df_len = num_epoch*len_per_epoch
    df_col_names = ['Epoch', 'loss', 'Train_acc', 'Epoch_Final_Test_acc', 'Sparsity',
                    'step_size', 'mom_ts', 'b_mom_ts', 'weight_decay', 'av_param']
    progress_df = pd.DataFrame(np.nan, index = [i for i in range(df_len)], columns= df_col_names)
    
    # Fix the epoch column of dataframe since it will not change through training
    for i in range(num_epoch):
        progress_df.Epoch[i*(len_per_epoch):(i+1)*len_per_epoch] = i+1
    
    progress_df_idx = 0
    
    for epoch in range(num_epoch):
        # For each epoch, if subset of data is used, make training/testing data a subset of full data
        if subset_Data is not None:
            # Create a subset of the MNIST dataset for analysis purposes
            subset_train_indices = torch.randperm(len(mnist_train_dataset))[:subset_train_size]
            subset_val_indices = torch.randperm(len(mnist_val_dataset))[:subset_val_size]
            
            trainset = Subset(mnist_train_dataset, subset_train_indices)
            testset = Subset(mnist_val_dataset, subset_val_indices)    
            
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                                          shuffle=True, num_workers=4)
            testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                                         shuffle=False, num_workers=2)
            
        total = 0
        correct = 0
        epoch_prog = 0
        for data in trainloader:
            # get the inputs
            inputs, labels = data
            
            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)
            
          # zero the parameter gradients
            optimizer.zero_grad()

          # forward + backward + optimize
            outputs = conv_net(inputs)
            loss = criterion(outputs, labels)
            progress_df.loss[progress_df_idx] = loss.item()
            
            loss.backward()
            optimizer.step()
            epoch_prog += 1
          # Calculate train accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            print('Epoch: ', epoch+1)
            print('Epoch progress: ', epoch_prog, '/', len_per_epoch, sep = '')
            correct += (predicted == labels).sum()
            progress_df.Train_acc[progress_df_idx] = 100*correct.item()/total
            sparsity= sum(torch.nonzero(x).size()[0] for x in list(conv_net.parameters()))
            progress_df.Sparsity[progress_df_idx] = 1 - sparsity/init_sparsity
            progress_df.step_size[progress_df_idx] = training_specs.step
            progress_df.mom_ts[progress_df_idx] = training_specs.mom_ts
            progress_df.b_mom_ts[progress_df_idx] = training_specs.b_mom_ts
            progress_df.weight_decay[progress_df_idx] = training_specs.wd
            progress_df.av_param[progress_df_idx] = training_specs.av
            print('Current Loss: ', loss.item())
            print('Current Train Accuracy:', 100*correct.item()/total)
            print('Current Sparsity:', 1 - sparsity/init_sparsity)
            
            progress_df_idx += 1
            print('')
            
        train_acc = correct
        
        accuracy = 10000 * correct / total
        sparsity = 10000 * (1- sparsity/init_sparsity)
        if device.type == "cpu":
            t_accuracy = test_accuracy(testloader, conv_net, cuda=False)
        else:
            t_accuracy = test_accuracy(testloader, conv_net, cuda=True)
        
        progress_df.Epoch_Final_Test_acc[(epoch)*len_per_epoch:(epoch+1)*len_per_epoch] = t_accuracy.item()/100
        print('Epoch:%d %% Training Accuracy: %d.%02d %% Test Accuracy: %d.%02d %% \nSparsity Percentage: %d.%02d %%' % (epoch+1,
                                accuracy / 100, accuracy % 100, t_accuracy / 100, t_accuracy % 100, sparsity/100, sparsity % 100))
        
        print('')
        # At about every 4 epochs, halve step size and double averaging.
        if epoch in [6, 10, 14, 18, 22, 26, 30, 34, 38, 42]:
            lr /= 2
            training_specs.set_step_size(lr)
            av_param = 1.0 - (1.0 - av_param) / 2.0
            training_specs.set_av_param(av_param)

    # Calculate accuracy and save output.
    if device.type == "cpu":
      final_accuracy = test_accuracy(testloader, conv_net, cuda=False)
    else:
      final_accuracy = test_accuracy(testloader, conv_net, cuda=True)
    print('Accuracy of the network on the %d test images: %d.%02d %%' %
          (len(testset), final_accuracy / 100, final_accuracy % 100))
    
    if model_output_file is not None:
        torch.save(conv_net, model_output_file)
    
    if progress_data_output_file is not None:
        progress_df.to_csv(progress_data_output_file, index = False)

    return progress_df



def tune_parameters(params_output_file, trials_output_file, train_batch_size = 128, subset_Data = None, k_folds = 1, max_evals = 40, num_epoch = 10):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
         
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         # transforms.RandomHorizontalFlip(), # I don't think a horizontal flip is appropriate for the MNIST dataset
         transforms.ToTensor()])

    trainset = torchvision.datasets.MNIST(root='./', train=True,
                                          download=True, transform=transform_train)


    model = mnist_resnet18(num_classes = 10).to(device)
      
    criterion = nn.CrossEntropyLoss()
    k_folds = k_folds # 5 fold
    max_evals = max_evals # 40 evals
    num_epoch = num_epoch # 5 epoch
    
    space = hpo.parameter_spaces.ResNetParamSpace()
    
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
        
    best_params, trials = hpo.bayesian_optimization.bayes_optimizer(space, max_evals, model, "resnet", criterion,
                                                               k_folds, num_epoch, trainset, train_batch_size, subset_Data, device)
    
    if params_output_file is not None:
        torch.save(best_params, params_output_file)   
    
    if trials_output_file is not None:        
        torch.save(trials, trials_output_file)
        
    return best_params, trials


if __name__ == '__main__':
    t0 = time.time()

    init_params = {
        'init_lr': 1.0, #1.0 by default
        'lam': 1.3e-6, #1.3e-6 by default
        'av_param': 0.0, # 0.0 by default
        'mom_ts': 9.5, # 9.5 by default
        'b_mom_ts': 9.5, # 9.5 by default
        'weight_decay': 5e-4 # 5e-4 by default
        }
    
    
    # model_output_file = 'results/model_data/resnet/mnist_resnet18_sparse_model_init_params_.dat'
    # progress_data_output_file = 'results/progress_data/resnet/mnist_resnet18_sparse_training_progress_init_params_.csv'
    
    model_output_file = None
    progress_data_output_file = None
    
    # # Use the following below to visualize necessary 'num_epoch' for bayesian optimization
    # progress_df = progress_dataframe(init_params, model_output_file= model_output_file, progress_data_output_file=progress_data_output_file,
    #                                   train_batch_size = 128, subset_Data = None, num_epoch=30)
    
    # # From the dataframe above:
    # #     Training accuracy reaches maximum 99% after 5 epochs on half of the dataset
    # #     Testing accuracy reaches 99% after 8 epochs and 99.5% after 16 epochs
    # #     Sparsity is 77% after 10 epochs and 86% after 20 epochs
    # #     Cross Entropy Loss is minimized to [.003, .15] after 5 epochs
    # # So 10 epochs on the full data set should be used for 'num_epoch' in bayesian optimization

    params_output_file = 'results/bayes_opt_params/resnet/mnist_resnet18_sparse_model_bayes_params_small_data.dat'
    trials_output_file = 'results/bayes_opt_params/resnet/mnist_resnet18_sparse_model_bayes_trials_small_data.dat'
    
    # params_output_file = None
    # trials_output_file = None
    
    # best_params, trials = tune_parameters(params_output_file, trials_output_file,
    #                                       train_batch_size=128, subset_Data=2**11, k_folds=1, max_evals=5, num_epoch=5)
    
    best_params = torch.load(params_output_file)
    trials = torch.load(trials_output_file)
    
    # model_output_file = 'results/model_data/resnet/mnist_resnet18_sparse_model_bayes_params_small_data.dat'
    # progress_data_output_file = 'results/progress_data/resnet/mnist_resnet18_sparse_training_progress_bayes_params_small_data.csv'
    
    model_output_file = None
    progress_data_output_file = None
    
    progress_df = progress_dataframe(best_params, model_output_file= model_output_file, progress_data_output_file=progress_data_output_file,
                                      train_batch_size = 128, subset_Data = 2**10, num_epoch= 3)
    
    # print(os.getcwd())
    print(time.time() - t0)
