import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

# geralized
class ResnetInit(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(ResnetInit, self).__init__()
        self.residual_stream_conv = spectral_norm(nn.Conv2d(in_channel, out_channel, 3, padding=1, stride=stride))
        self.transient_stream_conv = spectral_norm(nn.Conv2d(in_channel, out_channel, 3, padding=1, stride=stride))
        self.residual_stream_conv_across = spectral_norm(nn.Conv2d(in_channel, out_channel, 3, padding=1, stride=stride))
        self.transient_stream_conv_across = spectral_norm(nn.Conv2d(in_channel, out_channel, 3, padding=1, stride=stride))

        self.residual_LeakyReLU = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.transient_LeakyReLU = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.short_cut = nn.Sequential()
        if in_channel != out_channel or stride != 1:
            self.short_cut = nn.Sequential(
                spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride))
            )

    def forward(self, x):
        x_residual, x_transient = x
        residual_r_r = self.residual_stream_conv(x_residual)
        residual_r_t = self.residual_stream_conv_across(x_residual)
        residual_shortcut = self.short_cut(x_residual)

        transient_t_t = self.transient_stream_conv(x_transient)
        transient_t_r = self.transient_stream_conv_across(x_transient)

        x_residual = self.residual_LeakyReLU(residual_r_r + transient_t_r + residual_shortcut)
        x_transient = self.transient_LeakyReLU(residual_r_t + transient_t_t)

        return x_residual, x_transient


class RiRBlock(nn.Module):
    def __init__(self, in_channel, out_channel, layer_num, stride, layer=ResnetInit):
        super(RiRBlock, self).__init__()
        self.resnetinit = self._make_layers(in_channel, out_channel, layer_num, stride)

    def forward(self, x):
        x_residual, x_transient = self.resnetinit(x)

        return (x_residual, x_transient)

    def _make_layers(self, in_channel, out_channel, layer_num, stride, layer=ResnetInit):
        strides = [stride] + [1] * (layer_num - 1)
        layers = nn.Sequential()
        for index, s in enumerate(strides):
            layers.add_module("generalized layers{}".format(index), layer(in_channel, out_channel, s))
            in_channel = out_channel

        return layers


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        base = 64
        self.residual_pre_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(3, base, 3, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.transient_pre_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(3, base, 3, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.rir1 = RiRBlock(base, base, 3, 1)
        self.rir2 = RiRBlock(base, base, 3, 1)  # 64
        self.rir3 = RiRBlock(base, base * 2, 3, 2)  # 32
        self.rir4 = RiRBlock(base * 2, base * 2, 3, 2)  # 16
        self.rir5 = RiRBlock(base * 2, base * 2, 3, 1)
        self.rir6 = RiRBlock(base * 2, base * 4, 3, 2)  # 8
        self.rir7 = RiRBlock(base * 4, base * 4, 3, 2)  # 4
        self.rir8 = RiRBlock(base * 4, base * 4, 3, 1)

        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)),  # without this convolution, loss will soon be nan
        )

    def forward(self, x):
        x_residual = self.residual_pre_conv(x)
        x_transient = self.transient_pre_conv(x)

        x_residual, x_transient = self.rir1((x_residual, x_transient))
        x_residual, x_transient = self.rir2((x_residual, x_transient))
        x_residual, x_transient = self.rir3((x_residual, x_transient))
        x_residual, x_transient = self.rir4((x_residual, x_transient))
        x_residual, x_transient = self.rir5((x_residual, x_transient))
        x_residual, x_transient = self.rir6((x_residual, x_transient))
        x_residual, x_transient = self.rir7((x_residual, x_transient))
        x_residual, x_transient = self.rir8((x_residual, x_transient))
        h = torch.cat([x_residual, x_transient], 1)
        h = self.conv1(h)
        h = h.view(h.size(0), -1)

        return h