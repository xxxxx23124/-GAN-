import math
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn

class Smooth(nn.Module):
    """
    <a id="smooth"></a>

    ### Smoothing Layer

    This layer blurs each channel
    """

    def __init__(self):
        super().__init__()
        # Blurring kernel
        kernel = [[1, 2, 1],
                  [2, 4, 2],
                  [1, 2, 1]]
        # Convert the kernel to a PyTorch tensor
        kernel = torch.tensor([[kernel]], dtype=torch.float)
        # Normalize the kernel
        kernel /= kernel.sum()
        # Save kernel as a fixed parameter (no gradient updates)
        self.kernel = nn.Parameter(kernel, requires_grad=False)
        # Padding layer
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, x: torch.Tensor):
        # Get shape of the input feature map
        b, c, h, w = x.shape
        # Reshape for smoothening
        x = x.view(-1, 1, h, w)

        # Add padding
        x = self.pad(x)

        # Smoothen (blur) with the kernel
        x = F.conv2d(x, self.kernel)

        # Reshape and return
        return x.view(b, c, h, w)


class EqualizedLinear(nn.Module):
    """
    <a id="equalized_linear"></a>

    ## Learning-rate Equalized Linear Layer

    This uses [learning-rate equalized weights](#equalized_weights) for a linear layer.
    """

    def __init__(self, in_features: int, out_features: int, bias: float = 0.):
        """
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        * `bias` is the bias initialization constant
        """

        super().__init__()
        # [Learning-rate equalized weights](#equalized_weights)
        self.weight = EqualizedWeight([out_features, in_features])
        # Bias
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        # Linear transformation
        return F.linear(x, self.weight(), bias=self.bias)


class EqualizedConv2d(nn.Module):
    """
    <a id="equalized_conv2d"></a>

    ## Learning-rate Equalized 2D Convolution Layer

    This uses [learning-rate equalized weights](#equalized_weights) for a convolution layer.
    """

    def __init__(self, in_features: int, out_features: int,
                 kernel_size: int, padding: int = 0):
        """
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        * `kernel_size` is the size of the convolution kernel
        * `padding` is the padding to be added on both sides of each size dimension
        """
        super().__init__()
        # Padding size
        self.padding = padding
        # [Learning-rate equalized weights](#equalized_weights)
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        # Bias
        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor):
        # Convolution
        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)


class EqualizedWeight(nn.Module):
    """
    <a id="equalized_weight"></a>

    ## Learning-rate Equalized Weights Parameter

    This is based on equalized learning rate introduced in the Progressive GAN paper.
    Instead of initializing weights at $\mathcal{N}(0,c)$ they initialize weights
    to $\mathcal{N}(0, 1)$ and then multiply them by $c$ when using it.
    $$w_i = c \hat{w}_i$$

    The gradients on stored parameters $\hat{w}$ get multiplied by $c$ but this doesn't have
    an affect since optimizers such as Adam normalize them by a running mean of the squared gradients.

    The optimizer updates on $\hat{w}$ are proportionate to the learning rate $\lambda$.
    But the effective weights $w$ get updated proportionately to $c \lambda$.
    Without equalized learning rate, the effective weights will get updated proportionately to just $\lambda$.

    So we are effectively scaling the learning rate by $c$ for these weight parameters.
    """

    def __init__(self, shape: List[int]):
        """
        * `shape` is the shape of the weight parameter
        """
        super().__init__()

        # He initialization constant
        self.c = 1 / math.sqrt(np.prod(shape[1:]))
        # Initialize the weights with $\mathcal{N}(0, 1)$
        self.weight = nn.Parameter(torch.randn(shape))
        # Weight multiplication coefficient

    def forward(self):
        # Multiply the weights by $c$ and return
        return self.weight * self.c

class DiscriminatorBlock(nn.Module):
    """
    <a id="discriminator_black"></a>

    ### Discriminator Block

    ![Discriminator block](discriminator_block.svg)

    Discriminator block consists of two $3 \times 3$ convolutions with a residual connection.
    """

    def __init__(self, in_features, out_features):
        """
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        """
        super().__init__()
        # Down-sampling and $1 \times 1$ convolution layer for the residual connection
        self.residual = nn.Sequential(DownSample(),
                                      EqualizedConv2d(in_features, out_features, kernel_size=1))

        # Two $3 \times 3$ convolutions
        self.block = nn.Sequential(
            EqualizedConv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            EqualizedConv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        # Down-sampling layer
        self.down_sample = DownSample()

        # Scaling factor $\frac{1}{\sqrt 2}$ after adding the residual
        self.scale = 1 / math.sqrt(2)

    def forward(self, x):
        # Get the residual connection
        residual = self.residual(x)

        # Convolutions
        x = self.block(x)
        # Down-sample
        x = self.down_sample(x)

        # Add the residual and scale
        return (x + residual) * self.scale


class MiniBatchStdDev(nn.Module):
    """
    <a id="mini_batch_std_dev"></a>

    ### Mini-batch Standard Deviation

    Mini-batch standard deviation calculates the standard deviation
    across a mini-batch (or a subgroups within the mini-batch)
    for each feature in the feature map. Then it takes the mean of all
    the standard deviations and appends it to the feature map as one extra feature.
    """

    def __init__(self, group_size: int = 4):
        """
        * `group_size` is the number of samples to calculate standard deviation across.
        """
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor):
        """
        * `x` is the feature map
        """
        # Check if the batch size is divisible by the group size
        assert x.shape[0] % self.group_size == 0
        # Split the samples into groups of `group_size`, we flatten the feature map to a single dimension
        # since we want to calculate the standard deviation for each feature.
        grouped = x.view(self.group_size, -1)
        # Calculate the standard deviation for each feature among `group_size` samples
        #
        # \begin{align}
        # \mu_{i} &= \frac{1}{N} \sum_g x_{g,i} \\
        # \sigma_{i} &= \sqrt{\frac{1}{N} \sum_g (x_{g,i} - \mu_i)^2  + \epsilon}
        # \end{align}
        std = torch.sqrt(grouped.var(dim=0) + 1e-8)
        # Get the mean standard deviation
        std = std.mean().view(1, 1, 1, 1)
        # Expand the standard deviation to append to the feature map
        b, _, h, w = x.shape
        std = std.expand(b, -1, h, w)
        # Append (concatenate) the standard deviations to the feature map
        return torch.cat([x, std], dim=1)


class DownSample(nn.Module):
    """
    <a id="down_sample"></a>

    ### Down-sample

    The down-sample operation [smoothens](#smooth) each feature channel and
     scale $2 \times$ using bilinear interpolation.
    This is based on the paper
     [Making Convolutional Networks Shift-Invariant Again](https://arxiv.org/abs/1904.11486).
    """

    def __init__(self):
        super().__init__()
        # Smoothing layer
        self.smooth = Smooth()

    def forward(self, x: torch.Tensor):
        # Smoothing or blurring
        x = self.smooth(x)
        # Scaled down
        return F.interpolate(x, (x.shape[2] // 2, x.shape[3] // 2), mode='bicubic', align_corners=False)

class Discriminator(nn.Module):
    """
    <a id="discriminator"></a>

    ## StyleGAN 2 Discriminator

    ![Discriminator](style_gan2_disc.svg)

    Discriminator first transforms the image to a feature map of the same resolution and then
    runs it through a series of blocks with residual connections.
    The resolution is down-sampled by $2 \times$ at each block while doubling the
    number of features.
    """

    def __init__(self, log_resolution: int, n_features: int = 64, max_features: int = 512):
        """
        * `log_resolution` is the $\log_2$ of image resolution
        * `n_features` number of features in the convolution layer at the highest resolution (first block)
        * `max_features` maximum number of features in any generator block
        """
        super().__init__()

        # Layer to convert RGB image to a feature map with `n_features` number of features.
        self.from_rgb = nn.Sequential(
            EqualizedConv2d(3, n_features, 1),
            nn.LeakyReLU(0.2, True),
        )

        # Calculate the number of features for each block.
        #
        # Something like `[64, 128, 256, 512, 512, 512]`.
        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 1)]
        # Number of [discirminator blocks](#discriminator_block)
        n_blocks = len(features) - 1
        # Discriminator blocks
        blocks = [DiscriminatorBlock(features[i], features[i + 1]) for i in range(n_blocks)]
        self.blocks = nn.Sequential(*blocks)

        # [Mini-batch Standard Deviation](#mini_batch_std_dev )
        self.std_dev = MiniBatchStdDev()
        # Number of features after adding the standard deviations map
        final_features = features[-1] + 1
        # Final $3 \times 3$ convolution layer
        self.conv = EqualizedConv2d(final_features, final_features, 3)
        # Final linear layer to get the classification
        self.final = EqualizedLinear(2 * 2 * final_features, 1)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the input image of shape `[batch_size, 3, height, width]`
        """

        # Try to normalize the image (this is totally optional, but sped up the early training a little)
        # x = x - 0.5
        # Convert from RGB
        x = self.from_rgb(x)
        # Run through the [discriminator blocks](#discriminator_block)
        x = self.blocks(x)

        # Calculate and append [mini-batch standard deviation](#mini_batch_std_dev)
        x = self.std_dev(x)
        # $3 \times 3$ convolution
        x = self.conv(x)
        # Flatten
        x = x.reshape(x.shape[0], -1)
        # Return the classification score
        return self.final(x)