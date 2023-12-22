import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np

class NormLayer(nn.Module):
    """Normalization Layers.
    ------------
    # Arguments
        - channels: input channels, for batch norm and instance norm.
        - input_size: input shape without batch size, for layer norm.
    """
    def __init__(self, channels, normalize_shape=None, norm_type='bn'):
        super(NormLayer, self).__init__()
        norm_type = norm_type.lower()
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(channels)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(channels, affine=True)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(32, channels, affine=True)
        elif norm_type == 'pixel':
            self.norm = lambda x: F.normalize(x, p=2, dim=1)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(normalize_shape)
        elif norm_type == 'none':
            self.norm = lambda x: x
        else:
            assert 1==0, 'Norm type {} not support.'.format(norm_type)

    def forward(self, x):
        return self.norm(x)


class ReluLayer(nn.Module):
    """Relu Layer.
    ------------
    # Arguments
        - relu type: type of relu layer, candidates are
            - ReLU
            - LeakyReLU: default relu slope 0.2
            - PRelu 
            - SELU
            - none: direct pass
    """
    def __init__(self, channels, relu_type='relu'):
        super(ReluLayer, self).__init__()
        relu_type = relu_type.lower()
        if relu_type == 'relu':
            self.func = nn.ReLU(True)
        elif relu_type == 'leakyrelu':
            self.func = nn.LeakyReLU(0.2, inplace=True)
        elif relu_type == 'prelu':
            self.func = nn.PReLU(channels)
        elif relu_type == 'selu':
            self.func = nn.SELU(True)
        elif relu_type == 'none':
            self.func = lambda x: x
        else:
            assert 1==0, 'Relu type {} not support.'.format(relu_type)

    def forward(self, x):
        return self.func(x)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale='none', norm_type='none', relu_type='none', use_pad=True):
        super(ConvLayer, self).__init__()
        self.use_pad = use_pad
        
        bias = True if norm_type in ['pixel', 'none'] else False 
        stride = 2 if scale == 'down' else 1

        self.scale_func = lambda x: x
        if scale == 'up':
            self.scale_func = lambda x: nn.functional.interpolate(x, scale_factor=2, mode='nearest')

        self.reflection_pad = nn.ReflectionPad2d(kernel_size // 2) 
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)

        self.relu = ReluLayer(out_channels, relu_type)
        self.norm = NormLayer(out_channels, norm_type=norm_type)

    def forward(self, x):
        out = self.scale_func(x)
        if self.use_pad:
            out = self.reflection_pad(out)
        out = self.conv2d(out)
        out = self.norm(out)
        out = self.relu(out)
        return out


class ResidualBlock(nn.Module):
    """
    Residual block recommended in: http://torch.ch/blog/2016/02/04/resnets.html
    ------------------
    # Args
        - hg_depth: depth of HourGlassBlock. 0: don't use attention map.
        - use_pmask: whether use previous mask as HourGlassBlock input.
    """
    def __init__(self, c_in, c_out, relu_type='prelu', norm_type='bn', scale='none', hg_depth=2, att_name='spar'):
        super(ResidualBlock, self).__init__()
        self.c_in      = c_in
        self.c_out     = c_out
        self.norm_type = norm_type
        self.relu_type = relu_type
        self.hg_depth  = hg_depth

        kwargs = {'norm_type': norm_type, 'relu_type': relu_type}

        if scale == 'none' and c_in == c_out:
            self.shortcut_func = lambda x: x
        else:
            self.shortcut_func = ConvLayer(c_in, c_out, 3, scale)

        self.preact_func = nn.Sequential(
                    NormLayer(c_in, norm_type=self.norm_type),
                    ReluLayer(c_in, self.relu_type),
                    )

        if scale == 'down':
            scales = ['none', 'down']
        elif scale == 'up':
            scales = ['up', 'none']
        elif scale == 'none':
            scales = ['none', 'none']

        self.conv1 = ConvLayer(c_in, c_out, 3, scales[0], **kwargs) 
        self.conv2 = ConvLayer(c_out, c_out, 3, scales[1], norm_type=norm_type, relu_type='none')

        if att_name.lower() == 'spar':
            c_attn = 1
        elif att_name.lower() == 'spar3d':
            c_attn = c_out
        else:
            raise Exception("Attention type {} not implemented".format(att_name))

        # self.att_func = HourGlassBlock(self.hg_depth, c_out, c_attn, **kwargs)
        
    def forward(self, x):
        identity = self.shortcut_func(x)
        out = self.preact_func(x)
        out = self.conv1(out)
        out1 = self.conv2(out)
        out = identity + out1
        return out


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class PALayer(nn.Module):
    def __init__(self):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            ConvLayer(128, 128, 3, use_pad=True),
            nn.LeakyReLU(inplace=True),
            ConvLayer(128, 1, 3, use_pad=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            ConvLayer(128, 128, 1, use_pad=True),
            nn.LeakyReLU(inplace=True),
            ConvLayer(128, 128, 1, use_pad=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


# class Block(nn.Module):
#     def __init__(self):
#         super(Block, self).__init__()
#         self.conv1 = ConvLayer(128, 128, 3, use_pad=True)
#         self.act1 = nn.LeakyReLU(inplace=True)
#         self.conv2 = ConvLayer(128, 128, 3, use_pad=True)
#         self.calayer = CALayer()
#         self.palayer = PALayer()
#
#     def forward(self, x):
#         res = self.act1(self.conv1(x))
#         res = res + x
#         res = self.conv2(res)
#         res = self.calayer(res)
#         res = self.palayer(res)
#         res += x
#         return res




