# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 18:14:38 2024

@author: Devon
"""

import os
import time
import numpy as np
import pandas as pd
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
import hyperparameter_optimization as hpo


def progress_dataframe(train_batch_size = 128, subset_Data = None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_file = 'results/model_data/resnet/mnist_resnet18_sparse_model.dat'

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

    
    if subset_Data is not None:
        transform_train = transforms.Compose(
            [transforms.RandomCrop(28, padding=4),
             # transforms.RandomHorizontalFlip(), # I don't think a horizontal flip is appropriate for the MNIST dataset
             transforms.ToTensor()])
        
        transform_val = transforms.Compose(
            [transforms.ToTensor()])
        
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
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                                 shuffle=False, num_workers=2)
        
    conv_net = mnist_resnet18(num_classes = 10).to(device)
      
    conv_net.train()
    criterion = nn.CrossEntropyLoss()

    init_lr = 1.0
    lam = 5e-4 # orignially 1e-6
    av_param = 0.0
    training_specs = IterationSpecs(
        step_size=init_lr, mom_ts=9.5, b_mom_ts=9.5, weight_decay=5e-4, av_param=av_param)
    
    optimizer = xRDA(conv_net.parameters(), it_specs=training_specs,
                     prox=l1_prox(lam=lam, maximum_factor=500))
    
    lr = init_lr
    prev_train_acc = 0
    prev_sparsity = 0
    num_epoch = 50 # originally 500
    init_sparsity = sum([torch.nonzero(x).size()[0] for x in list(conv_net.parameters())])
    
    # DataFrame to keep track of progress, will contain:
    # Epoch, loss per step, Training Accuracy per step, Testing Accuracy per end of Epoch,
    # Sparsity per step, step_size per epoch, mom_ts per epoch, b_mom_ts per epoch,
    # weight_decay per epoch, av_param per epoch
    
    len_per_epoch = sum(1 for _ in trainloader.batch_sampler)
    df_len = num_epoch*len_per_epoch
    df_col_names = ['Epoch', 'loss', 'Train_acc', 'Epoch_Final_Test_acc', 'Sparsity',
                    'step_size', 'mom_ts', 'b_mom_ts', 'weight_decay', 'av_param']
    progress_df = pd.DataFrame(np.nan, index = [i for i in range(df_len)], columns= df_col_names)
    
    # Fix the epoch column of dataframe since it will not change through training
    for i in range(num_epoch):
        progress_df.Epoch[i*(len_per_epoch):(i+1)*len_per_epoch] = i+1
    
    progress_df_idx = 0
    
    for epoch in range(num_epoch):
        total = 0
        correct = 0
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

          # Calculate train accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            print(total)
            correct += (predicted == labels).sum()
            progress_df.Train_acc[progress_df_idx] = 100*correct.item()/total
            sparsity= sum(torch.nonzero(x).size()[0] for x in list(conv_net.parameters()))
            progress_df.Sparsity[progress_df_idx] = 1 - sparsity/init_sparsity
            progress_df.step_size[progress_df_idx] = training_specs.step
            progress_df.mom_ts[progress_df_idx] = training_specs.mom_ts
            progress_df.b_mom_ts[progress_df_idx] = training_specs.b_mom_ts
            progress_df.weight_decay[progress_df_idx] = training_specs.wd
            progress_df.av_param[progress_df_idx] = training_specs.av
            print(1 - sparsity/init_sparsity)
            
            progress_df_idx += 1
            
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
    print('Accuracy of the network on the 10000 test images: %d.%02d %%' %
          (final_accuracy / 100, final_accuracy % 100))
    torch.save(conv_net, output_file)

    return progress_df


# def bayes_opt_test(batch_size = 128, subset_Data = None):
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
output_file = 'results/model_data/resnet/mnist_resnet18_sparse_model.dat'

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

subset_Data = 1024
batch_size = 128

if subset_Data is not None:
    transform_train = transforms.Compose(
        [transforms.RandomCrop(28, padding=4),
         # transforms.RandomHorizontalFlip(), # I don't think a horizontal flip is appropriate for the MNIST dataset
         transforms.ToTensor()])
    
    transform_val = transforms.Compose(
        [transforms.ToTensor()])
    
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

    
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    
model = mnist_resnet18(num_classes = 10).to(device)
  
criterion = nn.CrossEntropyLoss()
k_folds = 3
max_evals = 1
num_epoch = 5

space = hpo.parameter_spaces.ResNetParamSpace()




best_params = hpo.bayesian_optimization.bayes_optimizer(space, max_evals, model, "resnet", criterion,
                                                           k_folds, num_epoch, trainset, batch_size, device)


# if __name__ == '__main__':
#     t0 = time.time()
#     progress_df = ()
#     # print(os.getcwd())
#     print(time.time() - t0)
