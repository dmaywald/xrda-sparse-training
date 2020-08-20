# Author: Jonathan Siegel
#
# Tests xRDA training method (an advanced version of RDA augmented with momentum)
# on the resnet-18 model on the CIFAR-10 dataset.
import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from models import vgg16_bn
from training_algorithms import xRDA
from regularization import l1_prox
from training_algorithms import IterationSpecs
from utils import test_accuracy

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
  output_file = 'vgg16_sparse_model.dat'
  batch_size = 128
  epoch_count = 450

  transform_train = transforms.Compose(

      [transforms.RandomCrop(32, padding=4),
       transforms.RandomHorizontalFlip(),
       transforms.ToTensor(),
       transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010))])

  transform_val = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010))])

  trainset = torchvision.datasets.CIFAR10(root='../../data', train=True,
                                          download=False, transform=transform_train)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=4)

  testset = torchvision.datasets.CIFAR10(root='../../data', train=False,
                                         download=False, transform=transform_val)
  testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                           shuffle=False, num_workers=2)

  conv_net = vgg16_bn(num_classes=10).cuda()
  conv_net.train()
  criterion = nn.CrossEntropyLoss()

  init_lr = 1.0
  lam = 8e-7
  av_param = 0.0
  training_specs = IterationSpecs(max_iter=math.ceil(50000 / batch_size) * epoch_count,
                                  init_step_size=init_lr, mom_ts=9.5, b_mom_ts=9.5, weight_decay=5e-4, init_av_param=av_param)
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
