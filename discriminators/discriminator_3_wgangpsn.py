import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm



class Discriminator(nn.Module):
    def __init__(self, ngpu, ndf, nc):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(

            spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False)),

            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False)),

            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),

            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)),
        )

    def forward(self, input):
        out = self.main(input)
        out = out.view(out.shape[0], -1)
        return out
