import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, ngpu, ndf, nc):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(

            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )
        self.active = nn.Sigmoid()

    def forward(self, input):
        out = self.main(input)
        out = out.view(out.shape[0], -1)
        out = self.active(out)
        return out


def test():
    net = Discriminator(ngpu=1, ndf=256, nc=3)
    #net = Stem_block(in_planes=4,planes=16)
    #net = Tree(BasicBlock, 32, 16, 3, 3)
    print(net)
    from torchsummary import summary
    summary(net, input_size=(8, 3, 64, 64))