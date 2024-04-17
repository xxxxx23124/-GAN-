import torch
import torch.nn as nn
import numpy

class EqualizedConv2d(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, padding_mode='reflect'):
        super(EqualizedConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups,
                              padding_mode=padding_mode)
        self.scale = numpy.sqrt(2) / numpy.sqrt(kernel_size * kernel_size * in_planes)
        self.bias = self.conv.bias
        self.conv.bias = None
        nn.init.normal_(self.conv.weight)
        nn.init.normal_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class EqualizedConvTranspose2D(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(EqualizedConvTranspose2D, self).__init__()
        self.convT = nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding)
        self.scale = numpy.sqrt(2) / numpy.sqrt(in_planes)
        self.bias = self.convT.bias
        self.convT.bias = None
        nn.init.normal_(self.convT.weight)
        nn.init.normal_(self.bias)

    def forward(self, x):
        return self.convT(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)

class Generator(nn.Module):
    def get_upsample(self, planes, out_planes, kernel_size, stride, padding):
        return nn.Sequential(
            #EqualizedConvTranspose2D(in_planes=planes, out_planes=out_planes, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ConvTranspose2d(planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_planes),
            nn.PReLU(),
        )

    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            self.get_upsample(nz, ngf * 8, 4, 1, 0),
            self.get_upsample(ngf * 8, ngf * 4, 4, 2, 1),
            self.get_upsample(ngf * 4, ngf * 2, 4, 2, 1),
            self.get_upsample(ngf * 2, ngf * 1, 4, 2, 1),
            self.get_upsample(ngf * 1, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)