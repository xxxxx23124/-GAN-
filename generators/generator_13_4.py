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


class StyleBlock(nn.Module):
    def __init__(self, last_planes: int, in_planes: int, out_planes: int, dense_depth: int,
                 kernel_size: int, m: int, image_size: int):
        super(StyleBlock, self).__init__()
        assert (m > 0)
        self.conv1 = EqualizedConv2d(last_planes, in_planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.activation1 = nn.PReLU(in_planes)
        self.m = m
        if m == 1:
            self.conv2 = EqualizedConv2d(in_planes, out_planes + dense_depth, kernel_size=kernel_size)
            self.bn2 = nn.BatchNorm2d(out_planes + dense_depth)
            self.activation2 = nn.PReLU(out_planes + dense_depth)
        else:
            self.skconv = SKConv(in_planes, out_planes + dense_depth, m, image_size)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        if self.m == 1:
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.activation2(x)
        else:
            x = self.skconv(x)
        return x


class ResnetInit(nn.Module):
    def __init__(self, last_planes: int, in_planes: int, out_planes: int, dense_depth: int,
                 kernel_size: int, m: int, image_size: int):
        super(ResnetInit, self).__init__()
        self.residual = StyleBlock(last_planes, in_planes, out_planes, dense_depth, kernel_size, m, image_size)
        self.transient = StyleBlock(last_planes, in_planes, out_planes, 0, kernel_size, m, image_size)
        self.residual_across = StyleBlock(last_planes, in_planes, out_planes, 0, kernel_size, m, image_size)
        self.transient_across = StyleBlock(last_planes, in_planes, out_planes, dense_depth, kernel_size, m, image_size)
        if image_size > 4:
            self.sk_attention_residual = SKAttention_conv(out_planes + dense_depth, 2)
            self.sk_attention_transient = SKAttention_conv(out_planes, 2)
        else:
            self.sk_attention_residual = SKAttention_fc(out_planes + dense_depth, 2)
            self.sk_attention_transient = SKAttention_fc(out_planes, 2)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        x_residual, x_transient = x

        residual_r_r = self.residual(x_residual).unsqueeze_(dim=1)
        residual_r_t = self.residual_across(x_residual).unsqueeze_(dim=1)
        transient_t_t = self.transient(x_transient).unsqueeze_(dim=1)
        transient_t_r = self.transient_across(x_transient).unsqueeze_(dim=1)

        feas_residual = torch.cat([residual_r_r, transient_t_r], dim=1)
        feas_transient = torch.cat([residual_r_t, transient_t_t], dim=1)

        fea_residual_v = (feas_residual * self.sk_attention_residual(feas_residual)).sum(dim=1)
        fea_transient_v = (feas_transient * self.sk_attention_transient(feas_transient)).sum(dim=1)
        return fea_residual_v, fea_transient_v


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


class SelfAttention(nn.Module):
    def __init__(self, in_planes: int, embedding_channels: int, image_size: int):
        super(SelfAttention, self).__init__()
        self.query = EqualizedConv2d(in_planes, embedding_channels, 3)
        self.key = EqualizedConv2d(in_planes, embedding_channels, 3)
        self.value = EqualizedConv2d(in_planes, embedding_channels, 3)
        self.self_att = EqualizedConv2d(embedding_channels, in_planes, 3)

        if image_size > 4:
            self.gamma = SEBlock_conv(in_planes)
        else:
            self.gamma = SEBlock_fc(in_planes)

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


class BasicBlock(nn.Module):

    def get_out_planes(self):
        if self.is_unify:
            return 2 * self.out_planes + 2 * self.dense_depth
        else:
            if self.root:
                return 2 * self.out_planes + 2 * self.dense_depth
            else:
                return self.last_planes + 1 * self.dense_depth

    def __init__(self, last_planes: int, in_planes: int, out_planes: int, dense_depth: int, root: bool,
                 is_unify: bool, m: int, image_size: int):
        super(BasicBlock, self).__init__()
        self.root = root
        self.last_planes = last_planes
        self.out_planes = out_planes
        self.dense_depth = dense_depth

        self.is_unify = is_unify
        self.root = root

        if is_unify:
            self.unify = EqualizedConv2d(last_planes, 2 * out_planes + dense_depth, kernel_size=1)
            self.bn_unify = nn.BatchNorm2d(2 * out_planes + dense_depth)
            self.rir_3 = ResnetInit(out_planes + dense_depth, in_planes, out_planes, dense_depth, 3, m, image_size)
        else:
            self.rir_3 = ResnetInit(last_planes - out_planes, in_planes, out_planes, dense_depth, 3, m, image_size)

        if root:
            self.shortcut = EqualizedConv2d(last_planes, 2 * out_planes + dense_depth, kernel_size=1)
            self.bn_shortcut = nn.BatchNorm2d(2 * out_planes + dense_depth)

        if image_size > 4:
            self.sk_attention_residual = SKAttention_conv(out_planes, 2)
        else:
            self.sk_attention_residual = SKAttention_fc(out_planes, 2)

        self.attention_residual = SelfAttention(out_planes + dense_depth, out_planes + dense_depth, image_size)
        self.attention_transient = SelfAttention(out_planes, out_planes, image_size)

    def forward(self, x: torch.Tensor):
        d = self.out_planes
        if self.is_unify:
            x = self.unify(x)
            x = self.bn_unify(x)
        x_residual = torch.cat([x[:, :d, :, :], x[:, 2 * d:, :, :]], 1)
        x_transient = x[:, d:, :, :]
        x_residual_3, x_transient_3 = self.rir_3((x_residual, x_transient))

        if self.root:
            x = self.shortcut(x)
            x = self.bn_shortcut(x)

        feas_residual = torch.cat([x[:, :d, :, :].unsqueeze(dim=1), x_residual_3[:, :d, :, :].unsqueeze(dim=1)], dim=1)
        feas_residual = (feas_residual * self.sk_attention_residual(feas_residual)).sum(dim=1)

        x_residual_3 = self.attention_residual(torch.cat([feas_residual, x_residual_3[:, d:, :, :]], 1))
        x_transient_3 = self.attention_transient(x_transient_3)

        out = torch.cat([x_residual_3[:, :d, :, :], x_transient_3, x[:, 2 * d:, :, :], x_residual_3[:, d:, :, :]], 1)
        return out


class ToRGB(nn.Module):
    def __init__(self, planes: int, m: int, image_size: int):
        super().__init__()
        assert (m > 0)
        self.m = m
        if m == 1:
            self.pre_conv = EqualizedConv2d(planes, planes, 3)
            self.pre_bn = nn.BatchNorm2d(planes)
            self.pre_activation = nn.PReLU(planes)
        else:
            self.skconv = SKConv(planes, planes, m, image_size)
        self.conv = EqualizedConv2d(planes, 3, kernel_size=1)
        self.bn = nn.BatchNorm2d(3)
        self.activation = nn.PReLU(3)

    def forward(self, x: torch.Tensor):
        if self.m == 1:
            x = self.pre_conv(x)
            x = self.pre_bn(x)
            x = self.pre_activation(x)
        else:
            x = self.skconv(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.activation(x)


class Tree(nn.Module):
    def get_out_planes(self):
        return self.root.get_out_planes()

    def __init__(self, last_planes: int, in_planes: int, out_planes: int, dense_depth: int, level: int,
                 block_num: int, m: int, image_size: int):
        super(Tree, self).__init__()
        assert (block_num > 0)
        self.level = level
        self.block_num = block_num
        self.out_planes = out_planes
        self.dense_depth = dense_depth

        if level == 1:
            self.root_last_planes = 2 * out_planes * (block_num - 1)
            sub_block = BasicBlock(last_planes, in_planes, out_planes, dense_depth, False,
                                   last_planes < 2 * out_planes, m, image_size)
            last_planes = sub_block.get_out_planes()
            self.__setattr__('block_%d' % 0, sub_block)
            for i in range(1, block_num):
                sub_block = BasicBlock(last_planes, in_planes, out_planes, dense_depth, False,
                                       False, m, image_size)
                last_planes = sub_block.get_out_planes()
                self.__setattr__('block_%d' % i, sub_block)
            self.root_last_planes += sub_block.get_out_planes()
            self.root = BasicBlock(self.root_last_planes, in_planes * block_num, out_planes, dense_depth,
                                   True, False, m, image_size)

        else:
            self.root_last_planes = 2 * out_planes * (block_num - 1)
            self.prev_root = BasicBlock(last_planes, in_planes, out_planes, dense_depth, False,
                                        last_planes < 2 * out_planes, m, image_size)
            self.root_last_planes += self.prev_root.get_out_planes()

            for i in reversed(range(1, level)):
                subtree = Tree(last_planes, in_planes, out_planes, dense_depth, i, block_num, m, image_size)
                last_planes = subtree.get_out_planes()
                self.root_last_planes += last_planes
                self.__setattr__('level_%d' % i, subtree)

            for i in range(block_num):
                sub_block = BasicBlock(last_planes, in_planes, out_planes, dense_depth, False, False, m,
                                       image_size)
                last_planes = sub_block.get_out_planes()
                self.__setattr__('block_%d' % i, sub_block)
            self.root_last_planes += sub_block.get_out_planes()
            self.root = BasicBlock(self.root_last_planes, in_planes * block_num, out_planes, dense_depth,
                                   True, False, m, image_size)

        self.to_rgb = ToRGB(self.get_out_planes(), m, image_size)
        if image_size > 4:
            self.sk_attention = SKAttention_conv(3, 2)
        else:
            self.sk_attention = SKAttention_fc(3, 2)

    def forward(self, x: torch.Tensor, rgb: torch.Tensor):
        d = self.out_planes
        xs = [self.prev_root(x)] if self.level > 1 else []
        for i in reversed(range(1, self.level)):
            level_i = self.__getattr__('level_%d' % i)
            x, rgb = level_i(x, rgb)
            xs.append(x)
        for i in range(self.block_num):
            block_i = self.__getattr__('block_%d' % i)
            x = block_i(x)
            xs.append(x[:, :2 * d, :, :])
        xs.append(x[:, 2 * d:, :, :])
        xs = torch.cat(xs, 1)
        out = self.root(xs)
        rgb_new = self.to_rgb(out)
        rgb.unsqueeze_(dim=1)
        rgb_new.unsqueeze_(dim=1)
        feas = torch.cat([rgb, rgb_new], dim=1)
        rgb = (feas * self.sk_attention(feas)).sum(dim=1)
        return out, rgb


class GeneratorBlock(nn.Module):
    def get_out_planes(self):
        return self.tree.get_out_planes()

    def __init__(self, last_planes: int, in_planes: int, out_planes: int, dense_depth: int, level: int,
                 block_num: int, m: int, image_size: int):
        super(GeneratorBlock, self).__init__()

        self.upsample = SKConvT(last_planes)
        self.tree = Tree(last_planes, in_planes, out_planes, dense_depth, level, block_num, m, image_size)
        self.upsample_rgb = SKConvT(3)

    def forward(self, x: torch.Tensor, rgb: torch.Tensor):
        rgb = self.upsample_rgb(rgb)
        x = self.upsample(x)
        x, rgb = self.tree(x, rgb)
        return x, rgb


class GeneratorStart(nn.Module):
    def get_out_planes(self):
        return self.tree.get_out_planes()

    def __init__(self, z_dim: int, in_planes: int, out_planes: int, dense_depth: int, level: int,
                 block_num: int, m: int):
        super(GeneratorStart, self).__init__()
        self.convT = nn.ConvTranspose2d(z_dim, out_planes, kernel_size=4, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(out_planes)
        self.activation = nn.PReLU(out_planes)
        self.to_rgb = ToRGB(out_planes, m, 4)
        self.tree = Tree(out_planes, in_planes, out_planes // 2, dense_depth, level, block_num, m, 4)

    def forward(self, x: torch.Tensor):
        x = self.convT(x)
        x = self.bn(x)
        x = self.activation(x)
        rgb = self.to_rgb(x)
        x, rgb = self.tree(x, rgb)
        return x, rgb


class Generator(nn.Module):
    def __init__(self, z_dim, planes=64):
        super(Generator, self).__init__()

        self.block0 = GeneratorStart(z_dim, planes * 8, planes * 8, planes // 8,
                                     1, 2, 1)
        self.block1 = GeneratorBlock(self.block0.get_out_planes(), planes * 4, planes * 4, planes // 8,
                                     1, 2, 2, 8)
        self.block2 = GeneratorBlock(self.block1.get_out_planes(), planes * 2, planes * 2, planes // 8,
                                     1, 2, 2, 16)
        self.block3 = GeneratorBlock(self.block2.get_out_planes(), planes * 1, planes * 1, planes // 8,
                                     2, 2, 2, 32)
        self.block4 = GeneratorBlock(self.block3.get_out_planes(), planes * 1, planes * 1, planes // 8,
                                     2, 2, 2, 64)

    def forward(self, x: torch.Tensor):
        x, rgb = self.block0(x)
        x, rgb = self.block1(x, rgb)
        x, rgb = self.block2(x, rgb)
        x, rgb = self.block3(x, rgb)
        x, rgb = self.block4(x, rgb)
        return rgb


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
