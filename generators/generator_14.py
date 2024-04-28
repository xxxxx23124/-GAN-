import torch
import torch.nn as nn
import math
from typing import Tuple, List
import numpy as np


class EqualizedWeight(nn.Module):
    def __init__(self, shape: List[int]):
        super(EqualizedWeight, self).__init__()
        self.shape = shape
        self.scale = 1 / math.sqrt(np.prod(shape[1:]))
        self.weights = nn.Parameter(torch.nn.init.normal_(torch.empty(shape), mean=0, std=1))

    def forward(self):
        return self.weights * self.scale


class EqualizedLinear(nn.Module):
    def __init__(self, in_planes: int, out_planes: int):
        super(EqualizedLinear, self).__init__()
        self.weight = EqualizedWeight([out_planes, in_planes])
        self.bias = nn.Parameter(torch.nn.init.normal_(torch.empty(out_planes), mean=0, std=1))

    def forward(self, x: torch.Tensor):
        return nn.functional.linear(x, self.weight(), bias=self.bias)


class EqualizedConv2d(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int):
        super(EqualizedConv2d, self).__init__()
        self.weight = EqualizedWeight([out_planes, in_planes, kernel_size, kernel_size])
        self.bias = nn.Parameter(torch.nn.init.normal_(torch.empty(out_planes), mean=0, std=1))
        self.pad = nn.ReplicationPad2d((kernel_size - 1) // 2)

    def forward(self, x: torch.Tensor):
        x = self.pad(x)
        return nn.functional.conv2d(x, self.weight(), bias=self.bias)


class SKAttention_conv(nn.Module):
    def __init__(self, planes: int, m: int):
        super(SKAttention_conv, self).__init__()
        self.gap_conv = nn.AdaptiveAvgPool2d(5)
        layers_conv = []
        for i in range(3):
            layers_conv.append(EqualizedConv2d(planes, planes, 3))
            layers_conv.append(nn.BatchNorm2d(planes))
            layers_conv.append(nn.PReLU(planes))
        self.conv_main = nn.Sequential(*layers_conv)

        self.gap_fc = nn.AdaptiveAvgPool2d(1)
        layers_fc = []
        for i in range(2):
            layers_fc.append(EqualizedLinear(planes, planes))
            layers_fc.append(nn.BatchNorm1d(planes))
            layers_fc.append(nn.PReLU(planes))
        self.fc_main = nn.Sequential(*layers_fc)

        self.M = m
        for i in range(m):
            layers_sub = []
            for _ in range(1):
                layers_sub.append(EqualizedLinear(planes, planes))
                layers_sub.append(nn.BatchNorm1d(planes))
                layers_sub.append(nn.PReLU(planes))
            layers_sub.append(EqualizedLinear(planes, planes))
            layers_sub.append(nn.BatchNorm1d(planes))
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
            layers_fc.append(nn.BatchNorm1d(planes))
            layers_fc.append(nn.PReLU(planes))
        self.fc_main = nn.Sequential(*layers_fc)

        self.M = m
        for i in range(m):
            layers_sub = []
            for _ in range(1):
                layers_sub.append(EqualizedLinear(planes, planes))
                layers_sub.append(nn.BatchNorm1d(planes))
                layers_sub.append(nn.PReLU(planes))
            layers_sub.append(EqualizedLinear(planes, planes))
            layers_sub.append(nn.BatchNorm1d(planes))
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


class Smooth(nn.Module):
    def __init__(self):
        super(Smooth, self).__init__()
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


class SKConvT(nn.Module):
    def __init__(self, planes: int):
        super(SKConvT, self).__init__()
        self.convT = nn.ConvTranspose2d(planes, planes, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(planes)
        self.activation_convT = nn.PReLU(planes)

        self.up_sample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
        self.smooth = Smooth()

        self.sk_attention = SKAttention_conv(planes, 2)

    def forward(self, x: torch.Tensor):
        fea_convT = self.activation_convT(self.bn(self.convT(x))).unsqueeze_(dim=1)
        fea_bic = self.smooth(self.up_sample(x)).unsqueeze_(dim=1)
        feas = torch.cat([fea_convT, fea_bic], dim=1)
        fea_v = (feas * self.sk_attention(feas)).sum(dim=1)
        return fea_v


class SKConv(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, m: int, image_size: int):
        super(SKConv, self).__init__()
        assert (m > 0)
        self.M = m
        for i in range(m):
            conv = EqualizedConv2d(in_planes, out_planes, kernel_size=3 + i * 2)
            self.__setattr__('conv_%d' % i, conv)
            bn = nn.BatchNorm2d(out_planes)
            self.__setattr__('BatchNorm_%d' % i, bn)
            nonlinear = nn.PReLU(out_planes)
            self.__setattr__('nonlinear_%d' % i, nonlinear)

        if image_size > 4:
            self.sk_attention = SKAttention_conv(out_planes, m)
        else:
            self.sk_attention = SKAttention_fc(out_planes, m)

    def forward(self, x: torch.Tensor):
        for i in range(self.M):
            conv = self.__getattr__('conv_%d' % i)
            bn = self.__getattr__('BatchNorm_%d' % i)
            nonlinear = self.__getattr__('nonlinear_%d' % i)
            fea = nonlinear(bn(conv(x))).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_v = (feas * self.sk_attention(feas)).sum(dim=1)
        return fea_v


class SEBlock_conv(nn.Module):
    def __init__(self, in_planes: int):
        super(SEBlock_conv, self).__init__()
        self.gap_conv = nn.AdaptiveAvgPool2d(5)
        layers_conv = []
        for i in range(3):
            layers_conv.append(EqualizedConv2d(in_planes, in_planes, 3))
            layers_conv.append(nn.BatchNorm2d(in_planes))
            layers_conv.append(nn.PReLU(in_planes))
        self.convs = nn.Sequential(*layers_conv)
        self.gap_fc = nn.AdaptiveAvgPool2d(1)
        layers_fc = []
        for i in range(2):
            layers_fc.append(EqualizedLinear(in_planes, in_planes))
            layers_fc.append(nn.BatchNorm1d(in_planes))
            layers_fc.append(nn.PReLU(in_planes))
        self.fcs = nn.Sequential(*layers_fc)
        self.fc_out = EqualizedLinear(in_planes, in_planes)
        self.fc_bn = nn.BatchNorm1d(in_planes)
        self.activation2 = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        b, c, _, _ = x.shape
        assert (_ >= 8)
        x = self.gap_conv(x)
        x = self.convs(x)
        x = self.gap_fc(x).view(b, c)
        x = self.fcs(x)
        x = self.fc_out(x)
        x = self.fc_bn(x)
        return self.activation2(x).view(b, c, 1, 1)


class SEBlock_fc(nn.Module):
    def __init__(self, in_planes: int):
        super(SEBlock_fc, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        layers_fc = []
        for i in range(4):
            layers_fc.append(EqualizedLinear(in_planes, in_planes))
            layers_fc.append(nn.BatchNorm1d(in_planes))
            layers_fc.append(nn.PReLU(in_planes))
        self.fcs = nn.Sequential(*layers_fc)
        self.fc_out = EqualizedLinear(in_planes, in_planes)
        self.fc_bn = nn.BatchNorm1d(in_planes)
        self.activation2 = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        b, c, _, _ = x.shape
        x = self.gap(x).view(b, c)
        x = self.fcs(x)
        x = self.fc_out(x)
        x = self.fc_bn(x)
        return self.activation2(x).view(b, c, 1, 1)


class GeneratorBlock(nn.Module):
    def get_out_planes(self):
        return self.tree.get_out_planes()

    def __init__(self, in_planes: int, out_planes: int, m: int, image_size: int):
        super(GeneratorBlock, self).__init__()

        self.upsample = SKConvT(in_planes)
        self.convs_1 = SKConv(in_planes, in_planes, m, image_size)
        self.convs_2 = SKConv(in_planes, out_planes, m, image_size)

    def forward(self, x: torch.Tensor):
        x = self.upsample(x)
        x = self.convs_1(x)
        x = self.convs_2(x)
        return x


class GeneratorStart(nn.Module):
    def get_out_planes(self):
        return self.tree.get_out_planes()

    def __init__(self, z_dim: int, out_planes: int):
        super(GeneratorStart, self).__init__()
        self.convT = nn.ConvTranspose2d(z_dim, out_planes, kernel_size=4, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(out_planes)
        self.activation = nn.PReLU(out_planes)
        self.convs_1 = nn.Sequential(
            EqualizedConv2d(out_planes, out_planes, 3),
            nn.BatchNorm2d(out_planes),
            nn.PReLU(out_planes)
        )
        self.convs_2 = nn.Sequential(
            EqualizedConv2d(out_planes, out_planes, 3),
            nn.BatchNorm2d(out_planes),
            nn.PReLU(out_planes)
        )

    def forward(self, x: torch.Tensor):
        x = self.convT(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.convs_1(x)
        x = self.convs_2(x)
        return x


class Generator(nn.Module):
    def __init__(self, z_dim, planes=32):
        super(Generator, self).__init__()

        self.block0 = GeneratorStart(z_dim, planes*8)
        self.block1 = GeneratorBlock(planes * 8, planes * 4, 2, 8)
        self.block2 = GeneratorBlock(planes * 4, planes * 2, 2, 16)
        self.block3 = GeneratorBlock(planes * 2, planes * 2, 2, 32)
        self.block4 = GeneratorBlock(planes * 2, planes * 1, 2, 64)
        self.to_rgb = EqualizedConv2d(planes * 1, 3, 5)

    def forward(self, x: torch.Tensor):
        x= self.block0(x)
        x= self.block1(x)
        x= self.block2(x)
        x= self.block3(x)
        x= self.block4(x)
        x = self.to_rgb(x)
        return x


def test():
    # net = BasicBlock(last_planes=16, in_planes=32, out_planes=12, dense_depth=8, root=False, feature_size=64)

    # net = Stem_block(in_planes=4,planes=16)
    # net = Tree(last_planes=16, in_planes=8, out_planes=8, dense_depth=8, level=5, block_num=6, feature_size=64)
    net = Generator(z_dim=256)
    # print(net)
    from torchsummary import summary
    summary(net, input_size=(4, 256, 1, 1))
    from torchviz import make_dot

    x = torch.randn(4, 256, 1, 1)
    y = net(x)
    # dot = make_dot(y, params=dict(net.named_parameters()))
    # dot.view()
    # dot.render("G13_model_graph")
    print(y.size())
    # print(net.get_out_planes())
    # print("successed")
