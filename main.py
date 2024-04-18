'''
对抗生成网络
主函数，程序主体为生成生成器和判别器，然后选择训练模式进行训练
backup存放项目备份
checkpoint存放模型储存文件
checkpoint records存放训练日志
discriminators存放判别器
generated_images存放生成的图片
generators存放生成器
images存放训练数据
record backup暂时没用
stylegan2为实验子项目，没用
train存放训练过程文件
Unet没用
units获取辅助小工具
'''

from units import dataloader
import torch
from train import gan, wgangp, ganR2, wgangpR2, wgansn, wgangpsnR2, wgansnR2
from units import get_generators, get_discriminator

ngpu = 1
# device为指定运算设备
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
#device = torch.device("cpu")
# batch_size为批大小
batch_size = 8

# 使用units文件夹里的dataloader选择读取数据集
# images_dataset = dataloader.get_dataset(dataroot="./images/CatDogWild")
images_dataset = dataloader.get_dataset(dataroot="./images/images-png-512x512", size=64)
# images_dataset = dataloader.get_dataset(dataroot="./images/Cat", size=64)
# images_dataset = dataloader.get_MNIST("./images/MNIST")
# images_dataset = dataloader.get_dataset(dataroot="./images/Flower")

# 使用dataloader的get_dataloader将数据进行batch_size的准备
images_loader = dataloader.get_dataloader(dataset=images_dataset, batch_size=batch_size)

# 获得判别器
# discriminator = get_discriminator.get_2(ngpu, device, ndf=64, nc=3)
discriminator = get_discriminator.get_3_wgan(ngpu, device, ndf=128, nc=3)
# discriminator = get_discriminator.get_3_wgansn(ngpu, device, ndf=64, nc=3)
# discriminator = get_discriminator.get_3_wgan_progan(ngpu, device, ndf=64, nc=3)
# discriminator = get_discriminator.get_4(ngpu, device)
# discriminator = get_discriminator.get_5(ngpu, device)
# discriminator = get_discriminator.get_5_wgan(ngpu, device)
# discriminator = get_discriminator.get_5_wgansn(ngpu, device)
# discriminator = get_discriminator.get_6_wgan(ngpu, device)
# discriminator = get_discriminator.get_7_wgan(ngpu, device)
# discriminator = get_discriminator.get_7_wgansn(ngpu, device)
# discriminator = get_discriminator.get_8(ngpu, device)
#discriminator = get_discriminator.get_9(ngpu, device)
# nz为生成器的噪声
nz = 256

# 获得生成器
generator = get_generators.get_1(device,nz,(3, 64, 64))
# generator = get_generators.get_2(ngpu,device,nz,256,3)
# generator = get_generators.get_3_progan(ngpu,device,nz,256,3)
# generator = get_generators.get_4(ngpu, device, z_dim=nz)
# generator = get_generators.get_5(ngpu, device, z_dim=nz)
# generator = get_generators.get_6(ngpu, device, z_dim=nz)
# generator = get_generators.get_7(ngpu, device, z_dim=nz)
# generator = get_generators.get_8(ngpu, device, z_dim=nz)
# generator = get_generators.get_9(ngpu, device, z_dim=nz)
# generator = get_generators.get_10(ngpu, device, z_dim=nz)
# generator = get_generators.get_10_2(ngpu, device, z_dim=nz)
# generator = get_generators.get_11(ngpu, device, z_dim=nz)
#generator = get_generators.get_12(ngpu, device, z_dim=nz)
#generator = get_generators.get_13(ngpu, device, z_dim=nz)

# 获得训练类
# p = wgangpsnR2.Train(images_loader, device, 400, nz, generator, 'G9', discriminator, 'D3WGANSN')
# p = wgansnR2.Train(images_loader, device, 400, nz, generator, 'G2', discriminator, 'D3WGANSN')
# p = wgansn.Train(images_loader, device, 400, nz, generator, 'G2', discriminator, 'D3WGANSN')
p = wgangpR2.Train(images_loader, device, 403, nz, generator, 'G13', discriminator, 'D3_WGAN')
# p = wgangp.Train(images_loader, device, 400, nz, generator, 'G2', discriminator, 'D3WGAN')
# p = ganR2.Train(images_loader, device, 400, nz, generator, 'G12', discriminator, 'D3_WGAN_PROGAN')
# p = gan.Train(images_loader, device, 400, nz, generator, 'G2', discriminator, 'D2')
# 训练
p.train()
