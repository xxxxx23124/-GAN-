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
        self.stride = stride
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.bias = nn.Parameter(torch.nn.init.normal_(torch.empty(out_features), mean=0, std=1))
        self.pad = nn.ReplicationPad2d(padding)

    def forward(self, x: torch.Tensor):
        x = self.pad(x)
        return nn.functional.conv2d(x, self.weight(), bias=self.bias, stride=self.stride)

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

class SKAttention_conv(nn.Module):
    def __init__(self, planes: int, m: int):
        super(SKAttention_conv, self).__init__()
        self.gap_conv = nn.AdaptiveAvgPool2d(5)
        layers_conv = []
        for i in range(3):
            layers_conv.append(EqualizedConv2d(planes, planes, 3, 1))
            layers_conv.append(nn.PReLU(planes))
        self.conv_main = nn.Sequential(*layers_conv)

        self.gap_fc = nn.AdaptiveAvgPool2d(1)
        layers_fc = []
        for i in range(2):
            layers_fc.append(EqualizedLinear(planes, planes))
            layers_fc.append(nn.PReLU(planes))
        self.fc_main = nn.Sequential(*layers_fc)

        self.M = m
        for i in range(m):
            layers_sub = []
            for _ in range(1):
                layers_sub.append(EqualizedLinear(planes, planes))
                layers_sub.append(nn.PReLU(planes))
            layers_sub.append(EqualizedLinear(planes, planes))
            fc_sub = nn.Sequential(*layers_sub)
            self.__setattr__('fc_sub_%d' % i, fc_sub)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, feas: torch.Tensor):
        b, s, c, _, _ = feas.shape
        assert (_ >= 8)
        fea_u = torch.sum(feas, dim=1)
        fea_s = self.conv_main(self.gap_conv(fea_u))
        fea_z = self.fc_main(self.gap_fc(fea_s).view(b, c))

        for i in range(self.M):
            fc_sub = self.__getattr__('fc_sub_%d' % i)
            vector = fc_sub(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)

        attention_vectors = self.softmax(attention_vectors)
        return attention_vectors.view(b, s, c, 1, 1)


class SKAttention_fc(nn.Module):
    def __init__(self, planes: int, m: int):
        super(SKAttention_fc, self).__init__()
        self.gap_fc = nn.AdaptiveAvgPool2d(1)
        layers_fc = []
        for i in range(4):
            layers_fc.append(EqualizedLinear(planes, planes))
            layers_fc.append(nn.PReLU(planes))
        self.fc_main = nn.Sequential(*layers_fc)

        self.M = m
        for i in range(m):
            layers_sub = []
            for _ in range(1):
                layers_sub.append(EqualizedLinear(planes, planes))
                layers_sub.append(nn.PReLU(planes))
            layers_sub.append(EqualizedLinear(planes, planes))
            fc_sub = nn.Sequential(*layers_sub)
            self.__setattr__('fc_sub_%d' % i, fc_sub)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, feas: torch.Tensor):
        b, s, c, _, _ = feas.shape
        fea_u = torch.sum(feas, dim=1)
        fea_s = self.gap_fc(fea_u).view(b, c)
        fea_z = self.fc_main(fea_s)

        for i in range(self.M):
            fc_sub = self.__getattr__('fc_sub_%d' % i)
            vector = fc_sub(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)

        attention_vectors = self.softmax(attention_vectors)
        return attention_vectors.view(b, s, c, 1, 1)


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_features, out_features, downsample, image_size):
        super().__init__()
        self.residual = nn.Sequential()
        self.block = nn.Sequential(
            EqualizedConv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            EqualizedConv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )
        if image_size > 4:
            self.sk_attention = SKAttention_conv(out_features, 2)
        else:
            self.sk_attention = SKAttention_fc(out_features, 2)

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
        residual = self.residual(x).unsqueeze(dim=1)
        x = self.block(x)
        x = self.down_sample(x).unsqueeze(dim=1)
        feas = torch.cat([residual, x], dim=1)
        return (feas * self.sk_attention(feas)).sum(dim=1)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        features = 32

        self.conv = nn.Sequential(
            EqualizedConv2d(3, features, 3, 1),  # 64
            nn.LeakyReLU(0.2, True),
            #DiscriminatorBlock(features, features, False, 64),
            #DiscriminatorBlock(features, features, False, 64),
            DiscriminatorBlock(features, 2 * features, True, 32),  # 32
            #DiscriminatorBlock(2 * features, 2 * features, False, 32),
            #DiscriminatorBlock(2 * features, 2 * features, False, 32),
            DiscriminatorBlock(2 * features, 4 * features, True, 16),  # 16
            #DiscriminatorBlock(4 * features, 4 * features, False, 16),
            #DiscriminatorBlock(4 * features, 4 * features, False, 16),
            DiscriminatorBlock(4 * features, 8 * features, True, 8),  # 8
            #DiscriminatorBlock(8 * features, 8 * features, False, 8),
            #DiscriminatorBlock(8 * features, 8 * features, False, 8),
            DiscriminatorBlock(8 * features, 16 * features, True, 4),  # 4
            MiniBatchStdDev(),
            #DiscriminatorBlock(16 * features + 1, 16 * features + 1, False, 4),
            #DiscriminatorBlock(16 * features + 1, 16 * features + 1, False, 4),
            DiscriminatorBlock(16 * features + 1, 16 * features + 1, True, 2),  # 2
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
