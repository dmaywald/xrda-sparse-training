# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 20:28:46 2024

@author: Owner
"""

import argparse
import os
import math
import time
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models

import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from training_algorithms import xRDA
from regularization import l1_prox
from training_algorithms import CosineSpecs
from utils import test_accuracy



model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__") and name.startswith("mnist_")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--data', metavar='DIR', default='../../data/',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='mnist_resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: mnist_resnet20)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=9.5, type=float, metavar='M',
                    help='momentum times timescale (around 10)')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--lam', default=2.8e-6, type=float, metavar='M',
                    help='lambda')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--save', default='./results/', type=str,
                    help='folder to save the results.')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.gpu is not None:
        if device.type == 'cpu':
            model = model.cpu()
        else:
            model = model.cuda(args.gpu)
    elif args.distributed:
        if device.type == 'cpu':
            model.cpu()
        else:
            model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            if device.type == 'cpu':
                model.cpu()
            else:
                model.cuda()
        else:
            if device.type == 'cpu':
                model = torch.nn.DataParallel(model).cpu()
            else:
                model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # # Data loading code
    # transform_train = transforms.Compose(

    #   [transforms.RandomCrop(28, padding=4),
    #    transforms.RandomHorizontalFlip(),
    #    transforms.ToTensor(),
    #    transforms.Normalize((0.4914, 0.4822, 0.4465),
    #                         (0.2023, 0.1994, 0.2010)),
    #    transforms.RandomErasing(p= 0.5, scale=(0,0.4), ratio=(0.3, 3.3), ),])

    # transform_val = transforms.Compose(
    #   [transforms.ToTensor(),
    #    transforms.Normalize((0.4914, 0.4822, 0.4465),
    #                         (0.2023, 0.1994, 0.2010)),])
    
    # Data loading code
    transform_train = transforms.Compose(

      [transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(), # I don't think a horizontal flip is appropriate for the MNIST data
        transforms.ToTensor(),
        transforms.RandomErasing(p= 0.5, scale=(0,0.4), ratio=(0.3, 3.3), ),])

    transform_val = transforms.Compose(
      [transforms.Pad(2), # Padding to a 32 x 32 image so the output dimensions after convolution fits
       transforms.ToTensor(),])

    trainset = torchvision.datasets.MNIST(root='./', train=True,
                                          download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=4)

    valset = torchvision.datasets.MNIST(root='./', train=False,
                                         download=True, transform=transform_val)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=128,
                                           shuffle=False, num_workers=2)

    # define loss function (criterion) and optimizer
    if device.type == 'cpu':
        criterion = nn.CrossEntropyLoss().cpu()
    else:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    training_specs = CosineSpecs(max_iter=math.ceil(len(trainset) / args.batch_size) * args.epochs,
                                 init_step_size=args.lr, mom_ts=args.momentum, b_mom_ts=args.momentum, weight_decay=args.weight_decay)
    optimizer = xRDA(model.parameters(), it_specs=training_specs,
                     prox=l1_prox(lam=args.lam, maximum_factor=500, mode='normal'))

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        #####################################################################################################
        num_zero_parameters = get_conv_zero_param(model)
        print('Zero parameters: {}'.format(num_zero_parameters))
        num_parameters = sum([param.nelement()
                              for param in model.parameters()])
        print('Parameters: {}'.format(num_parameters))
        #####################################################################################################

        # train for one epoch
        loss, prec1_train = train(
            train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=args.save, args=args)

        with open(os.path.join(args.save, 'model_data/resnet/resnet20_mnist_results_lr%.4f_lam%.8f_mom%.6f.txt' % (args.lr, args.lam, args.momentum)), "a+") as text_file:
            text_file.write(str(epoch + 1) + ' ' + '%.3f' % (loss.detach().cpu().numpy()) +
                            ' ' + '%.2f' % (prec1_train.detach().cpu().numpy()) +
                            ' ' + '%.2f' % (prec1.detach().cpu().numpy()) +
                            ' ' + '%d' % (num_zero_parameters) + '\n')
    return


def get_conv_zero_param(model):
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            total += torch.sum(m.weight.data.eq(0))
    return total


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
            
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type == 'cpu':
            target = target.cpu()
        else:
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5))
    return loss, prec1


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            if device.type == 'cpu':
                target = target.cpu
            else:
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, checkpoint, args):
    filename = 'model_data/resnet/resnet20_mnist_checkpoint_lr%.4f_lam%.8f_mom%.6f.pth.tar' % (
        args.lr, args.lam, args.momentum)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(
            checkpoint, 'model_data/resnet/resnet20_mnist_model_best_lr%.4f_lam%.8f_mom%.6f.pth.tar' % (args.lr, args.lam, args.momentum)))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            # original code: 
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            # Produced error:  view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
            
            # revised code: 
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(time.time() - t0)