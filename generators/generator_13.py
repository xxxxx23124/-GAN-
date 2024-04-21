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


class UpSample_rgb(nn.Module):
    def __init__(self):
        super(UpSample_rgb, self).__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
        self.smooth = Smooth()

    def forward(self, x: torch.Tensor):
        return self.smooth(self.up_sample(x))


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


class MappingNetwork(nn.Module):
    def __init__(self, planes: int, n_layers: int):
        super(MappingNetwork, self).__init__()
        layers = []
        for i in range(n_layers):
            layers.append(EqualizedLinear(planes, planes))
            layers.append(nn.LeakyReLU(0.2, True))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor):
        return self.net(z)


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

class SelfAttention(nn.Module):
    def __init__(self, in_planes: int, embedding_channels: int):
        super(SelfAttention, self).__init__()
        self.key = EqualizedConv2d(in_planes, embedding_channels, 1)
        self.query = EqualizedConv2d(in_planes, embedding_channels, 1)
        self.value = EqualizedConv2d(in_planes, embedding_channels, 1)
        self.self_att = EqualizedConv2d(embedding_channels, in_planes, 1)
        self.gamma = nn.Parameter(torch.nn.init.uniform_(torch.empty(in_planes), a=0.2, b=0.3))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.size()
        N = H * W
        f_x = self.key(x).view(B, -1, N)
        g_x = self.query(x).view(B, -1, N)
        h_x = self.value(x).view(B, -1, N)
        s = torch.bmm(f_x.permute(0, 2, 1), g_x)
        beta = self.softmax(s)
        v = torch.bmm(h_x, beta)
        v = v.view(B, -1, H, W)
        o = self.self_att(v)
        y = self.gamma[None, :, None, None] * o + x
        return y


class Conv2dWeightModulate(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int, demodulate: bool = True, eps: float = 1e-8):
        super(Conv2dWeightModulate, self).__init__()
        self.out_planes = out_planes
        self.demodulate = demodulate
        self.padding = (kernel_size - 1) // 2
        self.weight = EqualizedWeight([out_planes, in_planes, kernel_size, kernel_size])
        self.eps = eps

    def forward(self, x: torch.Tensor, s: torch.Tensor):
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
        x = nn.functional.pad(input=x, pad=(self.padding, self.padding, self.padding, self.padding), mode="replicate")
        x = nn.functional.conv2d(x, weights, groups=b)
        return x.reshape(-1, self.out_planes, h, w)


class StyleConv(nn.Module):
    def __init__(self, d_latent: int, in_planes: int, out_planes: int, kernel_size: int):
        super(StyleConv, self).__init__()
        self.to_style = nn.Sequential(
            MappingNetwork(d_latent, 2),
            EqualizedLinear(d_latent, in_planes)
        )
        self.conv = Conv2dWeightModulate(in_planes, out_planes, kernel_size=kernel_size)
        self.scale_noise = nn.Parameter(torch.nn.init.normal_(torch.empty(out_planes), mean=0, std=1))
        self.bias = nn.Parameter(torch.nn.init.normal_(torch.empty(out_planes), mean=0, std=1))

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        b, _, hight, width = x.shape
        s = self.to_style(w)
        x = self.conv(x, s)
        x = x + self.scale_noise[None, :, None, None] * torch.randn(b, 1, hight, width, device=x.device)
        return x + self.bias[None, :, None, None]


class StyleBlock(nn.Module):
    def __init__(self, d_latent: int, last_planes: int, in_planes: int, out_planes: int, dense_depth: int,
                 kernel_size: int):
        super(StyleBlock, self).__init__()
        self.conv1 = StyleConv(d_latent, last_planes, in_planes, kernel_size=1)
        self.activation1 = nn.LeakyReLU(0.2, True)
        self.conv2 = StyleConv(d_latent, in_planes, in_planes, kernel_size=kernel_size)
        self.activation2 = nn.LeakyReLU(0.2, True)
        self.conv3 = StyleConv(d_latent, in_planes, out_planes + dense_depth, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        x = self.conv1(x, w)
        x = self.activation1(x)
        x = self.conv2(x, w)
        x = self.activation2(x)
        x = self.conv3(x, w)
        return x


class SEBlock(nn.Module):
    def __init__(self, d_latent: int, in_planes: int, out_planes: int, dense_depth: int):
        super(SEBlock, self).__init__()
        self.conv1 = StyleConv(d_latent, out_planes + dense_depth, in_planes, kernel_size=1)
        self.activation1 = nn.LeakyReLU(0.2, True)
        self.conv2 = StyleConv(d_latent, in_planes, out_planes + dense_depth, kernel_size=1)
        self.activation2 = nn.Sigmoid()

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        b, _, hight, width = x.shape
        x = nn.functional.avg_pool2d(x, kernel_size=(hight, width))
        x = self.conv1(x, w)
        x = self.activation1(x)
        x = self.conv2(x, w)
        return self.activation2(x)


class SEStyleBlock(nn.Module):
    def __init__(self, d_latent: int, last_planes: int, in_planes: int, out_planes: int, dense_depth: int,
                 kernel_size: int):
        super(SEStyleBlock, self).__init__()
        self.out_planes = out_planes
        self.dense_depth = dense_depth
        self.convs = StyleBlock(d_latent, last_planes, in_planes, out_planes, dense_depth, kernel_size)
        self.se = SEBlock(d_latent, in_planes, out_planes, dense_depth)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        x = self.convs(x, w)
        se = self.se(x, w)
        return x * se


class ResnetInit(nn.Module):
    def __init__(self, d_latent, last_planes, in_planes, out_planes, dense_depth, kernel_size):
        super(ResnetInit, self).__init__()
        self.residual = SEStyleBlock(d_latent, last_planes, in_planes, out_planes, dense_depth, kernel_size)
        self.transient = SEStyleBlock(d_latent, last_planes, in_planes, out_planes, 0, kernel_size)
        self.residual_across = SEStyleBlock(d_latent, last_planes, in_planes, out_planes, 0, kernel_size)
        self.transient_across = SEStyleBlock(d_latent, last_planes, in_planes, out_planes, dense_depth, kernel_size)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor], w: torch.Tensor):
        x_residual, x_transient = x
        residual_r_r = self.residual(x_residual, w)
        residual_r_t = self.residual_across(x_residual, w)

        transient_t_t = self.transient(x_transient, w)
        transient_t_r = self.transient_across(x_transient, w)

        x_residual = residual_r_r + transient_t_r
        x_transient = residual_r_t + transient_t_t

        return x_residual, x_transient


class ToRGB(nn.Module):
    def __init__(self, d_latent: int, planes: int):
        super().__init__()
        self.to_style = nn.Sequential(
            MappingNetwork(d_latent, 2),
            EqualizedLinear(d_latent, planes)
        )
        self.attention = SelfAttention(planes, planes)
        self.conv = Conv2dWeightModulate(planes, 3, kernel_size=1, demodulate=False)
        self.bias = nn.Parameter(torch.nn.init.normal_(torch.empty(3), mean=0, std=1))
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        x = self.attention(x)
        s = self.to_style(w)
        x = self.conv(x, s)
        return self.activation(x + self.bias[None, :, None, None])

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
                 is_unify: bool):
        super(BasicBlock, self).__init__()
        self.root = root
        self.last_planes = last_planes
        self.out_planes = out_planes
        self.dense_depth = dense_depth
        self.scale = nn.Parameter(torch.nn.init.uniform_(torch.empty(out_planes), a=0.1, b=0.3))
        self.activation = nn.LeakyReLU(0.2, True)

        self.is_unify = is_unify
        self.root = root

        if is_unify:
            self.unify = StyleConv(d_latent, last_planes, 2 * out_planes + dense_depth, kernel_size=1)
            self.attention = SelfAttention(2 * out_planes + dense_depth, 2 * out_planes + dense_depth)
            self.rir_3 = ResnetInit(d_latent, out_planes + dense_depth, in_planes, out_planes, dense_depth, 3)
        else:
            self.attention = SelfAttention(last_planes, last_planes)
            self.rir_3 = ResnetInit(d_latent, last_planes - out_planes, in_planes, out_planes, dense_depth, 3)

        if root:
            self.shortcut = StyleConv(d_latent, last_planes, 2 * out_planes + dense_depth, kernel_size=1)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        d = self.out_planes
        if self.is_unify:
            x = self.unify(x, w)
        x_attention = self.attention(x)
        x_residual = torch.cat([x_attention[:, :d, :, :], x_attention[:, 2 * d:, :, :]], 1)
        x_transient = x_attention[:, d:, :, :]
        x_residual_3, x_transient_3 = self.rir_3((x_residual, x_transient), w)

        if self.root:
            x = self.shortcut(x, w)
        res = x[:, :d, :, :] + x_residual_3[:, :d, :, :] * self.scale[None, :, None, None]
        dense = x_residual_3[:, d:, :, :]
        out = torch.cat([res, x_transient_3, x[:, 2 * d:, :, :], dense], 1)
        out = self.activation(out)
        return out


class Tree(nn.Module):
    def get_out_planes(self):
        return self.root.get_out_planes()

    def __init__(self, d_latent: int, last_planes: int, in_planes: int, out_planes: int, dense_depth: int, level: int,
                 block_num: int):
        super(Tree, self).__init__()
        assert (block_num > 0)
        self.level = level
        self.block_num = block_num
        self.out_planes = out_planes
        self.dense_depth = dense_depth

        if level == 1:
            self.root_last_planes = 2 * out_planes * (block_num - 1)
            sub_block = BasicBlock(d_latent, last_planes, in_planes, out_planes, dense_depth, False,
                                   last_planes < 2 * out_planes)
            last_planes = sub_block.get_out_planes()
            self.__setattr__('block_%d' % 0, sub_block)
            for i in range(1, block_num):
                sub_block = BasicBlock(d_latent, last_planes, in_planes, out_planes, dense_depth, False,
                                       False)
                last_planes = sub_block.get_out_planes()
                self.__setattr__('block_%d' % i, sub_block)
            self.root_last_planes += sub_block.get_out_planes()
            self.root = BasicBlock(d_latent, self.root_last_planes, in_planes * block_num, out_planes, dense_depth,
                                   True, False)

        else:
            self.root_last_planes = 2 * out_planes * (block_num - 1)
            self.prev_root = BasicBlock(d_latent, last_planes, in_planes, out_planes, dense_depth, False,
                                        last_planes < 2 * out_planes)
            self.root_last_planes += self.prev_root.get_out_planes()

            for i in reversed(range(1, level)):
                subtree = Tree(d_latent, last_planes, in_planes, out_planes, dense_depth, i, block_num)
                last_planes = subtree.get_out_planes()
                self.root_last_planes += last_planes
                self.__setattr__('level_%d' % i, subtree)

            for i in range(block_num):
                sub_block = BasicBlock(d_latent, last_planes, in_planes, out_planes, dense_depth, False,
                                       False)
                last_planes = sub_block.get_out_planes()
                self.__setattr__('block_%d' % i, sub_block)
            self.root_last_planes += sub_block.get_out_planes()
            self.root = BasicBlock(d_latent, self.root_last_planes, in_planes * block_num, out_planes, dense_depth,
                                   True, False)
            self.mix_rgb = ToRGB(d_latent, 6)
            self.scale_old_mix = nn.Parameter(torch.nn.init.uniform_(torch.empty(3), a=0.4, b=0.6))
            self.scale_new_mix = nn.Parameter(torch.nn.init.uniform_(torch.empty(3), a=0.4, b=0.6))
        self.to_rgb = ToRGB(d_latent, self.get_out_planes())
        self.scale_old = nn.Parameter(torch.nn.init.uniform_(torch.empty(3), a=0.4, b=0.6))
        self.scale_new = nn.Parameter(torch.nn.init.uniform_(torch.empty(3), a=0.4, b=0.6))

    def forward(self, x: torch.Tensor, w: torch.Tensor, rgb: torch.Tensor):
        d = self.out_planes
        xs = [self.prev_root(x, w)] if self.level > 1 else []
        for i in reversed(range(1, self.level)):
            level_i = self.__getattr__('level_%d' % i)
            x, rgb_new = level_i(x, w, rgb)
            rgb = (rgb * self.scale_old_mix[None, :, None, None] +
                   self.mix_rgb(torch.cat([rgb, rgb_new], 1), w) * self.scale_new_mix[None, :, None, None])
            xs.append(x)
        for i in range(self.block_num):
            block_i = self.__getattr__('block_%d' % i)
            x = block_i(x, w)
            xs.append(x[:, :2 * d, :, :])
        xs.append(x[:, 2 * d:, :, :])
        xs = torch.cat(xs, 1)
        out = self.root(xs, w)
        rgb_new = self.to_rgb(out, w)
        rgb = rgb * self.scale_old[None, :, None, None] + rgb_new * self.scale_new[None, :, None, None]
        return out, rgb

class UpSample(nn.Module):
    def __init__(self, planes: int, out_planes: int, kernel_size: int, stride: int, padding: int,
                 use_attention: bool = True):
        super(UpSample, self).__init__()
        if use_attention:
            self.attention = SelfAttention(planes, planes)
        self.use_attention = use_attention
        self.convT = nn.ConvTranspose2d(planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.activation = nn.LeakyReLU(0.2, True)
        self.smooth = Smooth()

    def forward(self, x: torch.Tensor):
        if self.use_attention:
            x = self.attention(x)
        x = self.convT(x)
        x = self.activation(x)
        return self.smooth(x)

class GeneratorBlock(nn.Module):
    def get_out_planes(self):
        return self.tree.get_out_planes()

    def __init__(self, d_latent: int, last_planes: int, in_planes: int, out_planes: int, dense_depth: int, level: int,
                 block_num: int, use_attention: bool = True, kernel_size: int = 4, stride: int = 2, padding: int = 1):
        super(GeneratorBlock, self).__init__()

        self.upsample = UpSample(planes=last_planes, out_planes=last_planes,
                                 kernel_size=kernel_size, stride=stride, padding=padding, use_attention=use_attention)
        self.tree = Tree(d_latent, last_planes, in_planes, out_planes, dense_depth, level, block_num)
        self.scale_old = nn.Parameter(torch.nn.init.uniform_(torch.empty(3), a=0.4, b=0.6))
        self.scale_new = nn.Parameter(torch.nn.init.uniform_(torch.empty(3), a=0.4, b=0.6))
        self.upsample_rgb = UpSample_rgb()

    def forward(self, x: torch.Tensor, w: torch.Tensor, rgb: torch.Tensor):
        rgb = self.upsample_rgb(rgb)
        x = self.upsample(x)
        x, rgb_new = self.tree(x, w, rgb)
        rgb = rgb * self.scale_old[None, :, None, None] + rgb_new * self.scale_new[None, :, None, None]
        return x, rgb


class Generator(nn.Module):
    def __init__(self, z_dim, planes=64):
        super(Generator, self).__init__()
        self.mapping_network = MappingNetwork(z_dim, 8)
        self.upsample1 = UpSample(planes=z_dim, out_planes=planes * 8, kernel_size=4, stride=1, padding=0,
                                  use_attention=False)
        self.initial_constant = nn.Parameter(torch.nn.init.normal_(torch.empty((1, planes * 8, 4, 4)), mean=0, std=1))
        self.scale_old = nn.Parameter(torch.nn.init.uniform_(torch.empty(planes * 8), a=0.4, b=0.6))
        self.scale_new = nn.Parameter(torch.nn.init.uniform_(torch.empty(planes * 8), a=0.4, b=0.6))
        self.style1 = SEStyleBlock(z_dim, planes * 8, planes * 4, planes * 8, 0, 3)
        self.activation1 = nn.LeakyReLU(0.2, True)
        self.to_rgb1 = ToRGB(z_dim, planes * 8)
        self.tree = Tree(z_dim, planes * 8, planes * 4, planes * 4,
                         planes // 16, 2, 2)

        self.block1 = GeneratorBlock(z_dim, self.tree.get_out_planes(), planes * 4, planes * 4,
                                     planes // 8, 2, 2)  # 8
        self.block2 = GeneratorBlock(z_dim, self.block1.get_out_planes(), planes * 2, planes * 2,
                                     planes // 8, 2, 2)  # 16
        self.block3 = GeneratorBlock(z_dim, self.block2.get_out_planes(), planes * 2, planes * 2,
                                     planes // 8, 2, 3)  # 32
        self.block4 = GeneratorBlock(z_dim, self.block3.get_out_planes(), planes * 1, planes * 1,
                                     planes // 8, 2, 4)  # 64

    def forward(self, x: torch.Tensor):
        w = self.mapping_network(torch.squeeze(x))
        g = self.initial_constant.expand(x.shape[0], -1, -1, -1)
        x = self.upsample1(x)
        x = self.style1(x * self.scale_new[None, :, None, None] + g * self.scale_old[None, :, None, None], w)
        x = self.activation1(x)
        rgb = self.to_rgb1(x, w)
        x, rgb = self.tree(x, w, rgb)
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
    #dot = make_dot(y, params=dict(net.named_parameters()))
    # dot.view()
    #dot.render("G13_model_graph")
    print(y.size())
    # print(net.get_out_planes())
    #print("successed")
