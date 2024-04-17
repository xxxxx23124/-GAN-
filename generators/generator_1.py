import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, z_dim, target_image_size):
        """
        initialize

        :param z_dim: latent z dim, like z=100
        :param target_image_size:  tuple (3, h, w)
        """
        super().__init__()
        self.view_image_size = target_image_size[0] * target_image_size[1] * target_image_size[2]
        self.out_image_size = target_image_size
        self.z_dim = z_dim
        self.generator = nn.Sequential()
        self.generator.add_module(name="0", module=nn.Linear(in_features=self.z_dim, out_features=256))
        self.generator.add_module(name="1", module=nn.LeakyReLU(0.2))
        self.generator.add_module(name="2", module=nn.Linear(in_features=256, out_features=512))
        self.generator.add_module(name="3", module=nn.LeakyReLU(0.2))
        self.generator.add_module(name="4", module=nn.Linear(in_features=512, out_features=self.view_image_size))
        self.generator.add_module(name="5", module=nn.Tanh())

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.generator(x)
        out = out.view(x.size(0), *self.out_image_size)
        return out