# -*- coding:utf-8 -*-
from models.blocks import *
import torch
from torch import nn
import numpy as np

from models.common import *
import torch
import torch.nn as nn

import numpy as np
import scipy.io as sio


class UFSRNet(nn.Module):
    def __init__(self, conv=default_conv,res_depth = 10,
        relu_type = 'leakyrelu',
        norm_type = 'bn',
        att_name = 'spar',
        bottleneck_size = 4,):
        super(UFSRNet, self).__init__()
        n_feats = 32
        kernel_size = 3
        self.scale_idx = 0
        n_colors = 3

        bias=True

        self.DWT = DWT()
        self.IWT = IWT()
        self.head = nn.Conv2d(n_colors, n_feats, kernel_size, padding=(kernel_size // 2), bias=bias)

        self.d_l0 = nn.Conv2d(n_feats*4, n_feats*2, kernel_size=1)

        self.d_l0_Block1 = nn.Conv2d(in_channels=n_feats*2, out_channels=n_feats*2, kernel_size=3, padding=1, stride=1,
                               groups=n_feats*2,
                               bias=True)
        self.d_l0_Block2 = nn.Conv2d(in_channels=n_feats*2, out_channels=n_feats*2, kernel_size=3, padding=1, stride=1,
                               groups=n_feats*2,
                               bias=True)

        self.zishiying1 = nn.Sequential(
            nn.Conv2d(n_feats*2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.d_l1 = nn.Conv2d(n_feats*8, n_feats*4, kernel_size=1)

        self.d_l1_Block1 = nn.Conv2d(in_channels=n_feats*4, out_channels=n_feats*4, kernel_size=3, padding=1, stride=1,
                               groups=n_feats*4,
                               bias=True)
        self.d_l1_Block2 = nn.Conv2d(in_channels=n_feats*4, out_channels=n_feats*4, kernel_size=3, padding=1, stride=1,
                               groups=n_feats*4,
                               bias=True)

        self.zishiying2 = nn.Sequential(
            nn.Conv2d(n_feats*4, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.d_l2 = nn.Conv2d(n_feats*16, n_feats*4, kernel_size=1)

        self.d_l2_Block1 = nn.Conv2d(in_channels=n_feats*4, out_channels=n_feats*4, kernel_size=3, padding=1, stride=1,
                               groups=n_feats*4,
                               bias=True)
        self.d_l2_Block2 = nn.Conv2d(in_channels=n_feats*4, out_channels=n_feats*4, kernel_size=3, padding=1, stride=1,
                               groups=n_feats*4,
                               bias=True)

        self.zishiying3 = nn.Sequential(
            nn.Conv2d(n_feats*4, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.i_l2_Block2 = NAFBlock2(conv, n_feats*4, kernel_size)
        self.i_l2_Block1 = NAFBlock2(conv, n_feats*4, kernel_size)

        self.i_l2 = nn.Conv2d(n_feats*4, n_feats*2*8, kernel_size=1)

        self.i_l1_Block2 = NAFBlock2(conv, n_feats*4, kernel_size)
        self.i_l1_Block1 = NAFBlock2(conv, n_feats*4, kernel_size)

        self.i_l1 = nn.Conv2d(n_feats * 4, n_feats*8 , kernel_size=1)

        self.i_l0_Block2 = NAFBlock1(conv, n_feats*2 , kernel_size)
        self.i_l0_Block1 = NAFBlock1(conv, n_feats*2 , kernel_size)

        self.i_l0 = nn.Conv2d(n_feats*2, n_feats*4, kernel_size=1)

        self.act = nn.LeakyReLU(0.2, inplace=True)

        self.tail1 = nn.Conv2d(n_feats, n_colors, kernel_size, padding=(kernel_size // 2), bias=bias)

        self.tanh = nn.Tanh()

    def forward(self, x):
        c1 = self.head(x)

        d1 = self.DWT(c1)
        d1 = self.d_l0(d1)
        d1 = self.act(d1)
        b1 = self.act(self.d_l0_Block1(d1))
        b2 = self.d_l0_Block2(b1) * self.zishiying1(d1) + d1

        d2 = self.DWT(b2)
        d2 = self.d_l1(d2)
        d2 = self.act(d2)
        b3 = self.act(self.d_l1_Block1(d2))
        b4 = self.d_l1_Block2(b3) * self.zishiying2(d2) + d2

        d3 = self.DWT(b4)
        d3 = self.d_l2(d3)
        d3 = self.act(d3)
        b5 = self.d_l2_Block1(d3)
        b6 = self.i_l2_Block2(b5) * self.zishiying3(d3) + d3


        b7 = self.i_l2_Block1(b6)
        b8 = self.i_l2_Block2(b7)

        i1 = self.i_l2(b8)
        i1 = self.act(i1)
        i1 = self.IWT(i1) + b4
        b9 = self.i_l1_Block2(i1)
        b10 = self.i_l1_Block1(b9)
        i2 = self.i_l1(b10)

        i2 = self.IWT(i2) + b2
        b11 = self.i_l0_Block2(i2)
        b12 = self.i_l0_Block1(b11)
        i3 = self.i_l0(b12)
        i3 = self.act(i3)

        i3 = self.IWT(i3) + c1

        c2 = self.tail1(i3)
        c2 = self.act(c2)
        c2 = self.tanh(c2) + x
        return c2


