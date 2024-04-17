import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, image_size):
        """
        initialize

        :param image_size: tuple (3, h, w)
        """
        super().__init__()
        self.in_image_size = image_size[0] * image_size[1] * image_size[2]
        self.discriminator = nn.Sequential()
        self.discriminator.add_module(name="0", module=nn.Linear(in_features=self.in_image_size, out_features=256))
        self.discriminator.add_module(name="1", module=nn.LeakyReLU(0.2))
        self.discriminator.add_module(name="2", module=nn.Linear(in_features=256, out_features=64))
        self.discriminator.add_module(name="3", module=nn.LeakyReLU(0.2))
        self.discriminator.add_module(name="4", module=nn.Linear(in_features=64, out_features=1))
        self.discriminator.add_module(name="5", module=nn.Sigmoid())

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.discriminator(x)
        return out