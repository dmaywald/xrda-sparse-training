# Author: Jonathan Siegel
#
# Simple utility code which test a network on a dataset.

import torch
from torch.autograd import Variable

def test_accuracy(test_loader, net, cuda=False):
  # Test the network on the test set.
  net.eval()
  correct = 0
  total = 0
  for data in test_loader:
    images, labels = data
    images = Variable(images)
    labels = labels
    if cuda:
      images = images.cuda()
      labels = labels.cuda()
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
  net.train()
  return (10000 * correct / total)
