import torch
import torch.nn as nn
import math
from typing import Tuple, List
import numpy as np


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


class EqualizedWeight(nn.Module):
    def __init__(self, shape: List[int]):
        super(EqualizedWeight, self).__init__()
        self.shape = shape
        self.c = 1 / math.sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.nn.init.normal_(torch.empty(shape), mean=0, std=1))

    def forward(self):
        return self.weight * self.c


class EqualizedLinear(nn.Module):
    def __init__(self, in_planes: int, out_planes: int):
        super(EqualizedLinear, self).__init__()
        self.weight = EqualizedWeight([out_planes, in_planes])
        self.bias = nn.Parameter(torch.nn.init.normal_(torch.empty(out_planes), mean=0, std=1))

    def forward(self, x: torch.Tensor):
        return nn.functional.linear(x, self.weight(), bias=self.bias)


class EqualizedConv2d(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int):
        super().__init__()
        self.weight = EqualizedWeight([out_planes, in_planes, kernel_size, kernel_size])
        self.bias = nn.Parameter(torch.nn.init.normal_(torch.empty(out_planes), mean=0, std=1))
        self.pad = nn.ReplicationPad2d((kernel_size - 1) // 2)

    def forward(self, x: torch.Tensor):
        x = self.pad(x)
        return nn.functional.conv2d(x, self.weight(), bias=self.bias)


class EqualizedWeightConvTranspose2D(nn.Module):
    def __init__(self, shape: List[int]):
        super(EqualizedWeightConvTranspose2D, self).__init__()
        self.shape = shape
        self.c = 1 / math.sqrt(shape[0])
        self.weight = nn.Parameter(torch.nn.init.normal_(torch.empty(shape), mean=0, std=1))

    def forward(self):
        return self.weight * self.c


class EqualizedConvTranspose2D(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(EqualizedConvTranspose2D, self).__init__()
        self.padding = padding
        self.stride = stride
        self.weight = EqualizedWeightConvTranspose2D([in_planes, out_planes, kernel_size, kernel_size])
        self.bias = nn.Parameter(torch.nn.init.normal_(torch.empty(out_planes), mean=0, std=1))

    def forward(self, x):
        return nn.functional.conv_transpose2d(x, self.weight(), bias=self.bias, stride=self.stride,
                                              padding=self.padding)


class MappingNetwork(nn.Module):
    def __init__(self, planes: int, n_layers: int):
        super(MappingNetwork, self).__init__()
        layers = []
        for i in range(n_layers):
            layers.append(EqualizedLinear(planes, planes))
            layers.append(nn.PReLU(planes))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor):
        return self.net(z)


class SKConvT(nn.Module):
    def __init__(self, planes: int):
        super(SKConvT, self).__init__()
        self.convT = EqualizedConvTranspose2D(planes, planes, kernel_size=4, stride=2, padding=1)
        self.activation_convT = nn.PReLU(planes)

        self.up_sample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
        self.smooth = Smooth()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_main = MappingNetwork(planes, 4)

        self.fc_convT = nn.Sequential(
            MappingNetwork(planes, 2),
            EqualizedLinear(planes, planes)
        )

        self.fc_bic = nn.Sequential(
            MappingNetwork(planes, 2),
            EqualizedLinear(planes, planes)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        fea_convT = self.activation_convT(self.convT(x)).unsqueeze_(dim=1)
        fea_bic = self.smooth(self.up_sample(x)).unsqueeze_(dim=1)
        feas = torch.cat([fea_convT, fea_bic], dim=1)

        b, s, c, _, _ = feas.shape
        fea_u = torch.sum(feas, dim=1)
        fea_s = self.gap(fea_u).view(b, c)
        fea_z = self.fc_main(fea_s)

        fc_cpnvT = self.fc_convT(fea_z).unsqueeze_(dim=1)
        fc_bic = self.fc_bic(fea_z).unsqueeze_(dim=1)
        attention_vectors = torch.cat([fc_cpnvT, fc_bic], dim=1)

        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.view(b, s, c, 1, 1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


class Conv2dWeightModulate(nn.Module):
    def __init__(self, d_latent: int, in_planes: int, out_planes: int, kernel_size: int, demodulate: bool = True,
                 eps: float = 1e-8):
        super(Conv2dWeightModulate, self).__init__()
        self.to_style = nn.Sequential(
            MappingNetwork(d_latent, 2),
            EqualizedLinear(d_latent, in_planes)
        )
        self.out_planes = out_planes
        self.demodulate = demodulate
        self.pad = nn.ReplicationPad2d((kernel_size - 1) // 2)
        self.weight = EqualizedWeight([out_planes, in_planes, kernel_size, kernel_size])
        self.eps = eps

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        s = self.to_style(s)
        b, _, h, w = x.shape
        s = s[:, None, :, None, None]
        weights = self.weight()[None, :, :, :, :]
        weights = weights * s
        if self.demodulate:
            sigma_inv = torch.rsqrt(torch.pow(weights, 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * sigma_inv
        x = x.reshape(1, -1, h, w)
        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_planes, *ws)
        x = self.pad(x)
        x = nn.functional.conv2d(x, weights, groups=b)
        return x.reshape(-1, self.out_planes, h, w)


class StyleConv(nn.Module):
    def __init__(self, d_latent: int, in_planes: int, out_planes: int, kernel_size: int, use_noise: bool = False):
        super(StyleConv, self).__init__()
        self.conv = Conv2dWeightModulate(d_latent, in_planes, out_planes, kernel_size=kernel_size)
        self.use_noise = use_noise
        if use_noise:
            self.scale_noise = nn.Parameter(torch.nn.init.uniform_(torch.empty(out_planes), a=0.2, b=0.3))
        self.bias = nn.Parameter(torch.nn.init.normal_(torch.empty(out_planes), mean=0, std=1))

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        x = self.conv(x, w)
        b, c, hight, width = x.shape
        if self.use_noise:
            x = x + self.scale_noise[None, :, None, None] * torch.randn(b, c, hight, width, device=x.device)
        return x + self.bias[None, :, None, None]


class SKConv(nn.Module):
    def __init__(self, d_latent: int, in_planes: int, out_planes: int, m: int):
        super(SKConv, self).__init__()
        assert (m > 0)
        self.M = m
        for i in range(m):
            conv = StyleConv(d_latent, in_planes, out_planes, kernel_size=3 + i * 2)
            self.__setattr__('conv_%d' % i, conv)
            nonlinear = nn.PReLU(out_planes)
            self.__setattr__('nonlinear_%d' % i, nonlinear)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_main = MappingNetwork(out_planes, 4)
        for i in range(m):
            fc_sub = nn.Sequential(
                MappingNetwork(out_planes, 2),
                EqualizedLinear(out_planes, out_planes)
            )
            self.__setattr__('fc_sub_%d' % i, fc_sub)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        for i in range(self.M):
            conv = self.__getattr__('conv_%d' % i)
            nonlinear = self.__getattr__('nonlinear_%d' % i)
            fea = nonlinear(conv(x, w)).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)

        b, s, c, _, _ = feas.shape
        fea_u = torch.sum(feas, dim=1)
        fea_s = self.gap(fea_u).view(b, c)
        fea_z = self.fc_main(fea_s)

        for i in range(self.M):
            fc_sub = self.__getattr__('fc_sub_%d' % i)
            vector = fc_sub(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)

        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.view(b, s, c, 1, 1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


class StyleBlock(nn.Module):
    def __init__(self, d_latent: int, last_planes: int, in_planes: int, out_planes: int, dense_depth: int,
                 kernel_size: int, m: int):
        super(StyleBlock, self).__init__()
        assert (m > 0)
        self.conv1 = StyleConv(d_latent, last_planes, in_planes, kernel_size=1)
        self.activation1 = nn.PReLU(in_planes)
        self.m = m
        if m == 1:
            self.conv2 = StyleConv(d_latent, in_planes, in_planes, kernel_size=kernel_size)
            self.activation2 = nn.PReLU(in_planes)
        else:
            self.skconv = SKConv(d_latent, in_planes, in_planes, m)
        self.conv3 = StyleConv(d_latent, in_planes, out_planes + dense_depth, kernel_size=kernel_size, use_noise=False)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        x = self.conv1(x, w)
        x = self.activation1(x)
        if self.m == 1:
            x = self.conv2(x, w)
            x = self.activation2(x)
        else:
            x = self.skconv(x, w)
        x = self.conv3(x, w)
        return x


class SEBlock(nn.Module):
    def __init__(self, in_planes: int):
        super(SEBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fcs = MappingNetwork(in_planes, 2)
        self.fc_out = EqualizedLinear(in_planes, in_planes)
        self.activation2 = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        b, c, _, _ = x.shape
        x = self.gap(x).view(b, c)
        x = self.fcs(x)
        x = self.fc_out(x)
        return self.activation2(x).view(b, c, 1, 1)


class SelfAttention(nn.Module):
    def __init__(self, in_planes: int, embedding_channels: int):
        super(SelfAttention, self).__init__()
        self.query = EqualizedConv2d(in_planes, embedding_channels, 3)
        self.key = EqualizedConv2d(in_planes, embedding_channels, 3)
        self.value = EqualizedConv2d(in_planes, embedding_channels, 3)
        self.self_att = EqualizedConv2d(embedding_channels, in_planes, 3)
        self.gamma = SEBlock(in_planes)
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


class ResnetInit(nn.Module):
    def __init__(self, d_latent: int, last_planes: int, in_planes: int, out_planes: int, dense_depth: int,
                 kernel_size: int, m: int):
        super(ResnetInit, self).__init__()
        self.residual = StyleBlock(d_latent, last_planes, in_planes, out_planes, dense_depth, kernel_size, m)
        self.transient = StyleBlock(d_latent, last_planes, in_planes, out_planes, 0, kernel_size, m)
        self.residual_across = StyleBlock(d_latent, last_planes, in_planes, out_planes, 0, kernel_size, m)
        self.transient_across = StyleBlock(d_latent, last_planes, in_planes, out_planes, dense_depth, kernel_size, m)

        self.se_residual = SEBlock(out_planes + dense_depth)
        self.se_transient = SEBlock(out_planes)
        self.se_residual_across = SEBlock(out_planes)
        self.se_transient_across = SEBlock(out_planes + dense_depth)

        self.activation_residual = nn.PReLU(out_planes + dense_depth)
        self.activation_transient = nn.PReLU(out_planes)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor], w: torch.Tensor):
        x_residual, x_transient = x
        residual_r_r = self.residual(x_residual, w)
        residual_r_r = residual_r_r * self.se_residual(residual_r_r)
        residual_r_t = self.residual_across(x_residual, w)
        residual_r_t = residual_r_t * self.se_residual_across(residual_r_t)

        transient_t_t = self.transient(x_transient, w)
        transient_t_t = transient_t_t * self.se_transient(transient_t_t)
        transient_t_r = self.transient_across(x_transient, w)
        transient_t_r = transient_t_r * self.se_transient_across(transient_t_r)

        x_residual = residual_r_r + transient_t_r
        x_transient = residual_r_t + transient_t_t

        x_residual = self.activation_residual(x_residual)
        x_transient = self.activation_transient(x_transient)

        return x_residual, x_transient


class BasicBlock(nn.Module):

    def get_out_planes(self):
        if self.is_unify:
            return 2 * self.out_planes + 2 * self.dense_depth
        else:
            if self.root:
                return 2 * self.out_planes + 2 * self.dense_depth
            else:
                return self.last_planes + 1 * self.dense_depth

    def __init__(self, d_latent: int, last_planes: int, in_planes: int, out_planes: int, dense_depth: int, root: bool,
                 is_unify: bool, m: int):
        super(BasicBlock, self).__init__()
        self.root = root
        self.last_planes = last_planes
        self.out_planes = out_planes
        self.dense_depth = dense_depth
        self.scale = nn.Parameter(torch.nn.init.uniform_(torch.empty(out_planes), a=0.25, b=0.35))

        self.is_unify = is_unify
        self.root = root

        if is_unify:
            self.unify = StyleConv(d_latent, last_planes, 2 * out_planes + dense_depth, kernel_size=1)
            self.rir_3 = ResnetInit(d_latent, out_planes + dense_depth, in_planes, out_planes, dense_depth, 3, m)
        else:
            self.rir_3 = ResnetInit(d_latent, last_planes - out_planes, in_planes, out_planes, dense_depth, 3, m)

        if root:
            self.shortcut = StyleConv(d_latent, last_planes, 2 * out_planes + dense_depth, kernel_size=1)

        self.attention_residual = SelfAttention(out_planes + dense_depth, out_planes + dense_depth)
        self.attention_transient = SelfAttention(out_planes, out_planes)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        d = self.out_planes
        if self.is_unify:
            x = self.unify(x, w)
        x_residual = torch.cat([x[:, :d, :, :], x[:, 2 * d:, :, :]], 1)
        x_transient = x[:, d:, :, :]
        x_residual_3, x_transient_3 = self.rir_3((x_residual, x_transient), w)

        if self.root:
            x = self.shortcut(x, w)

        res = x[:, :d, :, :] + x_residual_3[:, :d, :, :] * self.scale[None, :, None, None]

        x_residual_3 = self.attention_residual(torch.cat([res, x_residual_3[:, d:, :, :]], 1))
        x_transient_3 = self.attention_transient(x_transient_3)

        out = torch.cat([x_residual_3[:, :d, :, :], x_transient_3, x[:, 2 * d:, :, :], x_residual_3[:, d:, :, :]], 1)
        return out


class ToRGB(nn.Module):
    def __init__(self, d_latent: int, planes: int, m: int):
        super().__init__()
        assert (m > 0)
        self.m = m
        if m == 1:
            self.pre_conv = StyleConv(d_latent, planes, planes, 3)
            self.pre_activation = nn.PReLU(planes)
        else:
            self.skconv = SKConv(d_latent, planes, planes, m)
        self.conv = Conv2dWeightModulate(d_latent, planes, 3, kernel_size=1, demodulate=False)
        self.bias = nn.Parameter(torch.nn.init.normal_(torch.empty(3), mean=0, std=1))
        self.activation = nn.PReLU(3)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        if self.m == 1:
            x = self.pre_conv(x, w)
            x = self.pre_activation(x)
        else:
            x = self.skconv(x, w)
        x = self.conv(x, w)
        return self.activation(x + self.bias[None, :, None, None])


class Tree(nn.Module):
    def get_out_planes(self):
        return self.root.get_out_planes()

    def __init__(self, d_latent: int, last_planes: int, in_planes: int, out_planes: int, dense_depth: int, level: int,
                 block_num: int, m: int):
        super(Tree, self).__init__()
        assert (block_num > 0)
        self.level = level
        self.block_num = block_num
        self.out_planes = out_planes
        self.dense_depth = dense_depth

        if level == 1:
            self.root_last_planes = 2 * out_planes * (block_num - 1)
            sub_block = BasicBlock(d_latent, last_planes, in_planes, out_planes, dense_depth, False,
                                   last_planes < 2 * out_planes, m)
            last_planes = sub_block.get_out_planes()
            self.__setattr__('block_%d' % 0, sub_block)
            for i in range(1, block_num):
                sub_block = BasicBlock(d_latent, last_planes, in_planes, out_planes, dense_depth, False,
                                       False, m)
                last_planes = sub_block.get_out_planes()
                self.__setattr__('block_%d' % i, sub_block)
            self.root_last_planes += sub_block.get_out_planes()
            self.root = BasicBlock(d_latent, self.root_last_planes, in_planes * block_num, out_planes, dense_depth,
                                   True, False, m)

        else:
            self.root_last_planes = 2 * out_planes * (block_num - 1)
            self.prev_root = BasicBlock(d_latent, last_planes, in_planes, out_planes, dense_depth, False,
                                        last_planes < 2 * out_planes, m)
            self.root_last_planes += self.prev_root.get_out_planes()

            for i in reversed(range(1, level)):
                subtree = Tree(d_latent, last_planes, in_planes, out_planes, dense_depth, i, block_num, m)
                last_planes = subtree.get_out_planes()
                self.root_last_planes += last_planes
                self.__setattr__('level_%d' % i, subtree)

            for i in range(block_num):
                sub_block = BasicBlock(d_latent, last_planes, in_planes, out_planes, dense_depth, False, False, m)
                last_planes = sub_block.get_out_planes()
                self.__setattr__('block_%d' % i, sub_block)
            self.root_last_planes += sub_block.get_out_planes()
            self.root = BasicBlock(d_latent, self.root_last_planes, in_planes * block_num, out_planes, dense_depth,
                                   True, False, m)
        self.to_rgb = ToRGB(d_latent, self.get_out_planes(), m)

    def forward(self, x: torch.Tensor, w: torch.Tensor, rgb: torch.Tensor):
        d = self.out_planes
        xs = [self.prev_root(x, w)] if self.level > 1 else []
        for i in reversed(range(1, self.level)):
            level_i = self.__getattr__('level_%d' % i)
            x, rgb = level_i(x, w, rgb)
            xs.append(x)
        for i in range(self.block_num):
            block_i = self.__getattr__('block_%d' % i)
            x = block_i(x, w)
            xs.append(x[:, :2 * d, :, :])
        xs.append(x[:, 2 * d:, :, :])
        xs = torch.cat(xs, 1)
        out = self.root(xs, w)
        rgb_new = self.to_rgb(out, w)
        rgb = rgb + rgb_new
        return out, rgb


class GeneratorBlock(nn.Module):
    def get_out_planes(self):
        return self.tree.get_out_planes()

    def __init__(self, d_latent: int, last_planes: int, in_planes: int, out_planes: int, dense_depth: int, level: int,
                 block_num: int, m: int):
        super(GeneratorBlock, self).__init__()

        self.upsample = SKConvT(last_planes)
        self.tree = Tree(d_latent, last_planes, in_planes, out_planes, dense_depth, level, block_num, m)
        self.upsample_rgb = SKConvT(3)

    def forward(self, x: torch.Tensor, w: torch.Tensor, rgb: torch.Tensor):
        rgb = self.upsample_rgb(rgb)
        x = self.upsample(x)
        x, rgb = self.tree(x, w, rgb)
        return x, rgb


class GeneratorStart(nn.Module):
    def get_out_planes(self):
        return self.tree.get_out_planes()

    def __init__(self, z_dim: int, mapping_layer: int, in_planes: int, out_planes: int, dense_depth: int, level: int,
                 block_num: int, m: int):
        super(GeneratorStart, self).__init__()
        self.mapping_network = MappingNetwork(z_dim, mapping_layer)
        self.convT = EqualizedConvTranspose2D(z_dim, out_planes, kernel_size=4, stride=1, padding=0)
        self.activation = nn.PReLU(out_planes)
        self.to_rgb = ToRGB(z_dim, out_planes, m)
        self.tree = Tree(z_dim, out_planes, in_planes, out_planes // 2, dense_depth, level, block_num, m)

    def forward(self, x: torch.Tensor):
        w = self.mapping_network(torch.squeeze(x))
        x = self.convT(x)
        x = self.activation(x)
        rgb = self.to_rgb(x, w)
        x, rgb = self.tree(x, w, rgb)
        return x, w, rgb


class Generator(nn.Module):
    def __init__(self, z_dim, planes=32):
        super(Generator, self).__init__()

        self.block0 = GeneratorStart(z_dim, 8, planes * 8, planes * 8, planes // 8, 1,
                                     1, 1)
        self.block1 = GeneratorBlock(z_dim, self.block0.get_out_planes(), planes * 4, planes * 4, planes // 8, 1,
                                     1, 2)  # 8
        self.block2 = GeneratorBlock(z_dim, self.block1.get_out_planes(), planes * 2, planes * 2, planes // 8, 1,
                                     1, 2)  # 16
        self.block3 = GeneratorBlock(z_dim, self.block2.get_out_planes(), planes * 1, planes * 1, planes // 8, 1,
                                     1, 2)  # 32
        self.block4 = GeneratorBlock(z_dim, self.block3.get_out_planes(), planes * 1, planes * 1, planes // 8, 1,
                                     1, 2)  # 64

    def forward(self, x: torch.Tensor):
        x, w, rgb = self.block0(x)
        x, rgb = self.block1(x, w, rgb)
        x, rgb = self.block2(x, w, rgb)
        x, rgb = self.block3(x, w, rgb)
        x, rgb = self.block4(x, w, rgb)
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
