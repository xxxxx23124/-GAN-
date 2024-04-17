'''
获得判别器
'''
import math

from discriminators import discriminator_1, discriminator_2, discriminator_3_wgangp, discriminator_3_wgangpsn, \
    discriminator_4, \
    discriminator_5, discriminator_5_wgangp, discriminator_5_wgangpsn, discriminator_6_wgangp, discriminator_7_wgangp, \
    discriminator_7_wgangpsn, discriminator_3_wgangp_progan, discriminator_8, discriminator_9
import torch
import torch.nn as nn
from units import Weight_Initialization


def get_1(device, image_size):
    discriminator = discriminator_1.Discriminator(image_size=image_size).to(device)
    return discriminator


def get_2(ngpu, device, ndf, nc):
    discriminator = discriminator_2.Discriminator(ngpu=ngpu, ndf=ndf, nc=nc).to(device)
    # if (device.type == 'cuda') and (ngpu > 1):
    # discriminator = nn.DataParallel(discriminator, list(range(ngpu)))
    discriminator.apply(Weight_Initialization.weights_init)
    return discriminator


def get_3_wgan(ngpu, device, ndf, nc):
    discriminator = discriminator_3_wgangp.Discriminator(ngpu=ngpu, ndf=ndf, nc=nc).to(device)
    # if (device.type == 'cuda') and (ngpu > 1):
    # discriminator = nn.DataParallel(discriminator, list(range(ngpu)))
    # discriminator.apply(Weight_Initialization.weights_init)
    return discriminator


def get_3_wgan_progan(ngpu, device, ndf, nc):
    discriminator = discriminator_3_wgangp_progan.Discriminator(ngpu=ngpu, ndf=ndf, nc=nc).to(device)
    # if (device.type == 'cuda') and (ngpu > 1):
    # discriminator = nn.DataParallel(discriminator, list(range(ngpu)))
    # discriminator.apply(Weight_Initialization.weights_init)
    return discriminator


def get_3_wgansn(ngpu, device, ndf, nc):
    discriminator = discriminator_3_wgangpsn.Discriminator(ngpu=ngpu, ndf=ndf, nc=nc).to(device)
    return discriminator


def get_4(ngpu, device):
    discriminator = discriminator_4.Discriminator().to(device)
    return discriminator


def get_5(ngpu, device):
    discriminator = discriminator_5.Discriminator().to(device)
    return discriminator


def get_5_wgan(ngpu, device):
    discriminator = discriminator_5_wgangp.Discriminator().to(device)
    return discriminator


def get_5_wgansn(ngpu, device):
    discriminator = discriminator_5_wgangpsn.Discriminator().to(device)
    return discriminator


def get_6_wgan(ngpu, device):
    discriminator = discriminator_6_wgangp.Discriminator().to(device)
    return discriminator


def get_7_wgan(ngpu, device):
    discriminator = discriminator_7_wgangp.Discriminator().to(device)
    return discriminator


def get_7_wgansn(ngpu, device):
    discriminator = discriminator_7_wgangpsn.Discriminator().to(device)
    return discriminator


def get_8(ngpu, device):
    discriminator = discriminator_8.Discriminator(int(math.log2(64))).to(device)
    return discriminator

def get_9(ngpu, device):
    discriminator = discriminator_9.Discriminator().to(device)
    return discriminator

