import math
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.utils.data
from torch import nn

class EqualizedWeight(nn.Module):
    def __init__(self, shape: List[int]):
        super(EqualizedWeight, self).__init__()
        self.shape = shape
        self.c = 1 / math.sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.nn.init.normal_(torch.empty(shape), mean=0, std=1))

    def forward(self):
        return self.weight * self.c


class EqualizedLinear(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, bias: float = 0.):
        super(EqualizedLinear, self).__init__()
        self.weight = EqualizedWeight([out_planes, in_planes])
        self.bias = nn.Parameter(torch.nn.init.normal_(torch.empty(out_planes), mean=0, std=1) * bias)

    def forward(self, x: torch.Tensor):
        return nn.functional.linear(x, self.weight(), bias=self.bias)


class EqualizedConv2d(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 kernel_size: int, padding: int = 0, stride: int = 1):
        super().__init__()
        self.padding = padding
        self.stride = stride
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.bias = nn.Parameter(torch.nn.init.normal_(torch.empty(out_features), mean=0, std=1))

    def forward(self, x: torch.Tensor):
        return nn.functional.conv2d(x, self.weight(), bias=self.bias, padding=self.padding, stride=self.stride)

class MiniBatchStdDev(nn.Module):
    def __init__(self, group_size: int = 4):
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor):
        assert x.shape[0] % self.group_size == 0
        grouped = x.view(self.group_size, -1)
        std = torch.sqrt(grouped.var(dim=0) + 1e-8)
        std = std.mean().view(1, 1, 1, 1)
        b, _, h, w = x.shape
        std = std.expand(b, -1, h, w)
        return torch.cat([x, std], dim=1)

class Smooth(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = [[1, 2, 1],
                  [2, 4, 2],
                  [1, 2, 1]]
        kernel = torch.tensor([[kernel]], dtype=torch.float)
        kernel = torch.div(kernel, kernel.sum())
        self.kernel = nn.Parameter(kernel, requires_grad=False)
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        x = x.view(-1, 1, h, w)
        x = self.pad(x)
        x = nn.functional.conv2d(x, self.kernel)
        return x.view(b, c, h, w)

class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = Smooth()

    def forward(self, x: torch.Tensor):
        x = self.smooth(x)
        return nn.functional.interpolate(x, (x.shape[2] // 2, x.shape[3] // 2), mode='bicubic', align_corners=False)

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.residual = nn.Sequential(DownSample(),
                                      EqualizedConv2d(in_features, out_features, kernel_size=1))
        self.block = nn.Sequential(
            EqualizedConv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.PReLU(in_features),
            EqualizedConv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.PReLU(in_features),
            EqualizedConv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.PReLU(in_features),
            EqualizedConv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.PReLU(out_features),
        )
        self.down_sample = nn.Sequential(
            EqualizedConv2d(out_features, out_features, kernel_size=3, padding=1, stride=2),
            nn.PReLU(out_features),
        )
        self.scale = 1 / math.sqrt(2)

    def forward(self, x):
        residual = self.residual(x)
        x = self.block(x)
        x = self.down_sample(x)
        return (x + residual) * self.scale

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        features = 64

        self.conv = nn.Sequential(
            EqualizedConv2d(3, features, 1),  # 64
            nn.PReLU(features),
            DiscriminatorBlock(features, 2 * features),  # 32
            DiscriminatorBlock(2 * features, 4 * features),  # 16
            DiscriminatorBlock(4 * features, 8 * features),  # 8
            DiscriminatorBlock(8 * features, 16 * features),  # 4
            MiniBatchStdDev(),
            DiscriminatorBlock(16 * features + 1, 16 * features + 1),  # 2
        )
        self.fc = nn.Sequential(
            EqualizedLinear(2 * 2 * (16 * features + 1), 2 * 2 * (16 * features + 1)),
            nn.PReLU(2 * 2 * (16 * features + 1)),
            EqualizedLinear(2 * 2 * (16 * features + 1), 2 * 2 * (16 * features + 1)),
            nn.PReLU(2 * 2 * (16 * features + 1)),
            EqualizedLinear(2 * 2 * (16 * features + 1), 2 * 2 * (16 * features + 1)),
            nn.PReLU(2 * 2 * (16 * features + 1)),
            EqualizedLinear(2 * 2 * (16 * features + 1), 1),
        )


    def forward(self, input):
        out = self.conv(input)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out


def test():
    net = Discriminator()
    #net = Stem_block(in_planes=4,planes=16)
    #net = Tree(BasicBlock, 32, 16, 3, 3)
    print(net)
    from torchsummary import summary
    summary(net, input_size=(8, 3, 64, 64))
