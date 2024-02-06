# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 16:02:14 2024

@author: Devon
"""

import torch.nn as nn
import torch.quantization as q
import torch.utils.model_zoo as model_zoo
from torch.quantization import QuantStub, DeQuantStub
import math


# __all__ = [
#     'VGG', 'mnist_vgg11', 'mnist_vgg11_bn', 'mnist_vgg13', 'mnist_vgg13_bn', 'mnist_vgg16', 'mnist_vgg16_bn',
#     'mnist_vgg19_bn', 'mnist_vgg19',
# ]


__all__ = [
    'mnist_VGG','mnist_vgg11', 'mnist_vgg11_bn', 'mnist_vgg13', 'mnist_vgg13_bn', 'mnist_vgg16', 'mnist_vgg16_bn',
    'mnist_vgg19_bn', 'mnist_vgg19',
]


model_urls = {
    'mnist_vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'mnist_vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'mnist_vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'mnist_vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class mnist_VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(mnist_VGG, self).__init__()
        self.features = features
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = nn.AvgPool2d(2)(x) # Originally x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels)
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class mnist_VGG_quant(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(mnist_VGG_quant, self).__init__()
        self.features = features
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels)
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def fuse_model(self):
        modules_list = []
        for name, m in self.features.named_modules():
            if isinstance(m, nn.Sequential):
              continue
            if isinstance(m, nn.Conv2d):
                modules_list.append(name)
            elif isinstance(m, nn.ReLU):
                modules_list.append(name)
                q.fuse_modules(self.features, modules_list, inplace=True)
                modules_list = []
            elif len(modules_list) > 0:
                modules_list.append(name)

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1) # Kernel size originally 3
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
    # 'E': [64, 128, 'M', 128, 256, 'M', 64, 128, 256, 512, 1024, 'M', 64, 128, 256, 512, 1024, 2048,'M',256, 512, 1024, 512,'M']
}


def mnist_vgg11(**kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = mnist_VGG(make_layers(cfg['A']), **kwargs)
    return model


def mnist_vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = mnist_VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model


def mnist_vgg13(**kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = mnist_VGG(make_layers(cfg['B']), **kwargs)
    return model


def mnist_vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = mnist_VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model


def mnist_vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = mnist_VGG(make_layers(cfg['D']), **kwargs)
    return model


def mnist_vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = mnist_VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model

def mnist_vgg16_bn_quant(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization and supporting quantization"""
    model = mnist_VGG_quant(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model

def mnist_vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = mnist_VGG(make_layers(cfg['E']), **kwargs)
    return model


def mnist_vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = mnist_VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model

def mnist_vgg19_bn_quant(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization and supporting quantization"""
    model = mnist_VGG_quant(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model


