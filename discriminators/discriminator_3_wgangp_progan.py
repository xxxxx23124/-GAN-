import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

# StandardDeviation from PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION's 3 INCREASING VARIATION USING MINIBATCH STANDARD DEVIATION
class StandardDeviation(torch.nn.Module):

    def forward(self, in_features):
        batch_size, _, height, width = in_features.shape
        output = in_features - in_features.mean(dim=0, keepdim=True)
        output = torch.sqrt_(output.pow_(2.0).mean(dim=0, keepdim=False) + 10e-8)
        output = output.mean().view(1, 1, 1, 1)
        output = output.repeat(batch_size, 1, height, width)
        output = torch.cat([in_features, output], 1)
        return output

class EqualizedConv2d(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1):
        super(EqualizedConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups)
        self.scale = numpy.sqrt(2) / numpy.sqrt(kernel_size * kernel_size * in_planes)
        self.bias = self.conv.bias
        self.conv.bias = None
        nn.init.normal_(self.conv.weight)
        nn.init.normal_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)

class Discriminator(nn.Module):
    def __init__(self, ngpu, ndf, nc):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(

            EqualizedConv2d(nc, ndf, 1, 1, 0),
            nn.PReLU(),
            EqualizedConv2d(ndf, ndf, 3, 1, 1),
            nn.PReLU(),
            EqualizedConv2d(ndf, ndf, 3, 2, 1),
            nn.PReLU(),

            EqualizedConv2d(ndf, ndf*2, 3, 1, 1),
            nn.PReLU(),
            EqualizedConv2d(ndf*2, ndf*2, 3, 2, 1),
            nn.PReLU(),

            EqualizedConv2d(ndf*2, ndf * 4, 3, 1, 1),
            nn.PReLU(),
            EqualizedConv2d(ndf * 4, ndf * 4, 3, 2, 1),
            nn.PReLU(),

            EqualizedConv2d(ndf * 4, ndf * 8, 3, 1, 1),
            nn.PReLU(),
            EqualizedConv2d(ndf * 8, ndf * 8, 3, 2, 1),
            nn.PReLU(),

            StandardDeviation(),
            EqualizedConv2d(ndf * 8 + 1, ndf * 8, 3, 1, 1),
            nn.PReLU(),
            EqualizedConv2d(ndf * 8, ndf * 8, 4, 1, 0),
            nn.PReLU(),
            EqualizedConv2d(ndf * 8, 1, 1, 1, 0),
        )

    def forward(self, input):
        out = self.main(input)
        out = out.view(out.shape[0], -1)
        return out


def test():
    net = Discriminator(ngpu=1, ndf=256, nc=3)
    #net = Stem_block(in_planes=4,planes=16)
    #net = Tree(BasicBlock, 32, 16, 3, 3)
    print(net)
    from torchsummary import summary
    summary(net, input_size=(8, 3, 64, 64))
