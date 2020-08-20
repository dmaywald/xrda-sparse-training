# Code from https://github.com/simochen/model-tools.
import numpy as np
import os

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from models import vgg16_bn, resnet56


def print_model_param_nums(model=None):
    if model == None:
        model = torchvision.models.alexnet()
    total = sum([len(param.nonzero())
                 if param.requires_grad else 0 for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))
    return total

def count_model_param_flops(model=None, input_res=224, multiply_adds=True):

    prods = {}

    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)
        return hook_per

    list_1 = []

    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))
    list_2 = {}

    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)

    list_conv = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * \
            self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        # flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size

        num_weight_params = (self.weight.data != 0).float().sum()
        assert self.weight.numel() == kernel_ops * output_channels, "Not match"
        flops = (num_weight_params * (2 if multiply_adds else 1) + bias_ops *
                 output_channels) * output_height * output_width * batch_size

        list_conv.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = len(self.weight.nonzero()) * (2 if multiply_adds else 1)
        bias_ops = len(self.bias.nonzero())

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn = []

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu = []

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling = []

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * \
            output_height * output_width * batch_size

        list_pooling.append(flops)

    list_upsample = []
    # For bilinear upsample

    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            if isinstance(net, torch.nn.Upsample):
                net.register_forward_hook(upsample_hook)
            return
        for c in childrens:
            foo(c)

    if model == None:
        model = torchvision.models.alexnet()
    foo(model)
    input = Variable(torch.rand(
        3, input_res, input_res).unsqueeze(0), requires_grad=True).cuda()
    out = model(input)

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) +
                   sum(list_relu) + sum(list_pooling) + sum(list_upsample))

    print('  + Number of FLOPs: %.2fG' % (total_flops / 1e9))

    return total_flops

# checkpoint = torch.load('./model/vgg19_sparse.dat')

# state_dict = checkpoint['state_dict']
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     if 'module.' in k:
#         name = k[:9] + k[16:]  # remove `module.`
#     else:
#         name = k
#     new_state_dict[name] = v
# # load params
# model.load_state_dict(new_state_dict)
# # model.load_state_dict(checkpoint['state_dict'])


model = resnet56().cuda()
print(count_model_param_flops(model, input_res=32).cpu().numpy())
model.load_state_dict(torch.load(
    './results/resnet56_sparse_model_cosine_7en7.dat'))
print(count_model_param_flops(model, input_res=32).cpu().numpy())

# model = vgg16_bn().cuda()
# print(count_model_param_flops(model, input_res=32).cpu().numpy())
# model = torch.load('./results/vgg16_8en7_sparse_model.dat')
# print(count_model_param_flops(model, input_res=32).cpu().numpy())
