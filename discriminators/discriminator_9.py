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
    def __init__(self, in_features: int, out_features: int, kernel_size: int, padding: int = 0, stride: int = 1):
        super().__init__()
        self.padding = padding
        self.stride = stride
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.bias = nn.Parameter(torch.nn.init.normal_(torch.empty(out_features), mean=0, std=1))

    def forward(self, x: torch.Tensor):
        x = nn.functional.pad(input=x, pad=(self.padding, self.padding, self.padding, self.padding), mode="replicate")
        return nn.functional.conv2d(x, self.weight(), bias=self.bias, stride=self.stride)

class SEBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = EqualizedConv2d(in_planes, in_planes, kernel_size=1)
        self.activation1 = nn.PReLU(in_planes)
        self.conv2 = EqualizedConv2d(in_planes, out_planes, kernel_size=1)
        self.activation2 = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        return self.activation2(x)

class SelfAttention(nn.Module):
    def __init__(self, in_planes: int, embedding_channels: int):
        super(SelfAttention, self).__init__()
        self.query = EqualizedConv2d(in_planes, embedding_channels, 1)
        self.key = EqualizedConv2d(in_planes, embedding_channels, 1)
        self.value = EqualizedConv2d(in_planes, embedding_channels, 1)
        self.self_att = EqualizedConv2d(embedding_channels, in_planes, 1)
        self.gamma = SEBlock(in_planes, in_planes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.size()
        N = H * W
        f_x = self.query(x).view(B, -1, N)
        g_x = self.key(x).view(B, -1, N)
        h_x = self.value(x).view(B, -1, N)
        s = torch.bmm(f_x.permute(0, 2, 1), g_x)
        beta = self.softmax(s)
        v = torch.bmm(h_x, beta)
        v = v.view(B, -1, H, W)
        o = self.self_att(v)
        y = self.gamma(o) * o + x
        return y

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
    def __init__(self, in_features, out_features, downsample):
        super().__init__()
        self.residual = nn.Sequential()
        self.block = nn.Sequential(
            EqualizedConv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            EqualizedConv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )
        self.se = SEBlock(out_features, out_features)
        self.down_sample = nn.Sequential()
        if downsample:
            self.residual = nn.Sequential(DownSample(),
                                          EqualizedConv2d(in_features, out_features, kernel_size=1))
            self.down_sample = nn.Sequential(
                Smooth(),
                EqualizedConv2d(out_features, out_features, kernel_size=3, padding=1, stride=2),
                nn.LeakyReLU(0.2, True),
            )

    def forward(self, x):
        residual = self.residual(x)
        x = self.block(x)
        x = self.down_sample(x)
        x = x * self.se(x)
        return x + residual

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        features = 64

        self.conv = nn.Sequential(
            EqualizedConv2d(3, features, 3, 1),  # 64
            nn.LeakyReLU(0.2, True),
            SelfAttention(features, features),
            DiscriminatorBlock(features, features, False),
            DiscriminatorBlock(features, features, False),
            DiscriminatorBlock(features, 2 * features, True),  # 32
            SelfAttention(2 * features, 2 * features),
            DiscriminatorBlock(2 * features, 2 * features, False),
            DiscriminatorBlock(2 * features, 2 * features, False),
            DiscriminatorBlock(2 * features, 4 * features, True),  # 16
            SelfAttention(4 * features, 4 * features),
            DiscriminatorBlock(4 * features, 4 * features, False),
            DiscriminatorBlock(4 * features, 4 * features, False),
            DiscriminatorBlock(4 * features, 8 * features, True),  # 8
            SelfAttention(8 * features, 8 * features),
            DiscriminatorBlock(8 * features, 8 * features, False),
            DiscriminatorBlock(8 * features, 8 * features, False),
            DiscriminatorBlock(8 * features, 16 * features, True),  # 4
            SelfAttention(16 * features, 16 * features),
            MiniBatchStdDev(),
            DiscriminatorBlock(16 * features + 1, 16 * features + 1, False),
            DiscriminatorBlock(16 * features + 1, 16 * features + 1, False),
            DiscriminatorBlock(16 * features + 1, 16 * features + 1, True),  # 2
        )
        self.fc = nn.Sequential(
            EqualizedLinear(2 * 2 * (16 * features + 1), 2 * 2 * (16 * features + 1)),
            nn.LeakyReLU(0.2, True),
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
