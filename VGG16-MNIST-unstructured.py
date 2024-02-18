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
from utils import test_accuracy

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
  main()
  print(time.time() - t0)
