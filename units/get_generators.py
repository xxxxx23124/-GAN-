'''
获得生成器
'''

from generators import generator_1, generator_2, generator_3_progan, generator_4, generator_5, generator_6, generator_7, generator_8, \
    generator_9, generator_10, generator_10_2, generator_11, generator_12, generator_13
import torch
import torch.nn as nn
from units import Weight_Initialization


def get_1(device, z_dim, target_image_size):
    generator = generator_1.Generator(z_dim=z_dim, target_image_size=target_image_size).to(device)
    return generator


def get_2(ngpu, device, nz, ngf, nc):
    generator = generator_2.Generator(ngpu=ngpu, nz=nz, ngf=ngf, nc=nc).to(device)
    # if (device.type == 'cuda') and (ngpu > 1):
    # generator = nn.DataParallel(generator, list(range(ngpu)))
    generator.apply(Weight_Initialization.weights_init)
    return generator

def get_3_progan(ngpu, device, nz, ngf, nc):
    generator = generator_3_progan.Generator(ngpu=ngpu, nz=nz, ngf=ngf, nc=nc).to(device)
    # if (device.type == 'cuda') and (ngpu > 1):
    # generator = nn.DataParallel(generator, list(range(ngpu)))
    return generator


def get_4(ngpu, device, z_dim=128):
    generator = generator_4.Generator(z_dim=z_dim).to(device)
    # if (device.type == 'cuda') and (ngpu > 1):
    # generator = nn.DataParallel(generator, list(range(ngpu)))
    # generator.apply(Weight_Initialization.weights_init)
    return generator


def get_5(ngpu, device, z_dim=128):
    generator = generator_5.Generator(z_dim=z_dim).to(device)
    # if (device.type == 'cuda') and (ngpu > 1):
    # generator = nn.DataParallel(generator, list(range(ngpu)))
    # generator.apply(Weight_Initialization.weights_init)
    return generator


def get_6(ngpu, device, z_dim=128):
    generator = generator_6.Generator(z_dim=z_dim).to(device)
    # if (device.type == 'cuda') and (ngpu > 1):
    # generator = nn.DataParallel(generator, list(range(ngpu)))
    # generator.apply(Weight_Initialization.weights_init)
    return generator


def get_7(ngpu, device, z_dim=128):
    generator = generator_7.Generator(z_dim=z_dim).to(device)
    # if (device.type == 'cuda') and (ngpu > 1):
    # generator = nn.DataParallel(generator, list(range(ngpu)))
    # generator.apply(Weight_Initialization.weights_init)
    return generator


def get_8(ngpu, device, z_dim=128):
    generator = generator_8.Generator(z_dim=z_dim).to(device)
    # if (device.type == 'cuda') and (ngpu > 1):
    # generator = nn.DataParallel(generator, list(range(ngpu)))
    # generator.apply(Weight_Initialization.weights_init)
    return generator


def get_9(ngpu, device, z_dim=128):
    generator = generator_9.Generator(z_dim=z_dim).to(device)
    # if (device.type == 'cuda') and (ngpu > 1):
    # generator = nn.DataParallel(generator, list(range(ngpu)))
    # generator.apply(Weight_Initialization.weights_init)
    return generator

def get_10(ngpu, device, z_dim=128):
    generator = generator_10.Generator(z_dim=z_dim).to(device)
    # if (device.type == 'cuda') and (ngpu > 1):
    # generator = nn.DataParallel(generator, list(range(ngpu)))
    # generator.apply(Weight_Initialization.weights_init)
    return generator

def get_10_2(ngpu, device, z_dim=128):
    generator = generator_10_2.Generator(z_dim=z_dim).to(device)
    # if (device.type == 'cuda') and (ngpu > 1):
    # generator = nn.DataParallel(generator, list(range(ngpu)))
    # generator.apply(Weight_Initialization.weights_init)
    return generator

def get_11(ngpu, device, z_dim=128):
    generator = generator_11.Generator(z_dim=z_dim).to(device)
    # if (device.type == 'cuda') and (ngpu > 1):
    # generator = nn.DataParallel(generator, list(range(ngpu)))
    # generator.apply(Weight_Initialization.weights_init)
    return generator

def get_12(ngpu, device, z_dim=128):
    generator = generator_12.Generator(z_dim=z_dim).to(device)
    # if (device.type == 'cuda') and (ngpu > 1):
    # generator = nn.DataParallel(generator, list(range(ngpu)))
    # generator.apply(Weight_Initialization.weights_init)
    return generator

def get_13(ngpu, device, z_dim=128):
    generator = generator_13.Generator(z_dim=z_dim).to(device)
    # if (device.type == 'cuda') and (ngpu > 1):
    # generator = nn.DataParallel(generator, list(range(ngpu)))
    # generator.apply(Weight_Initialization.weights_init)
    return generator