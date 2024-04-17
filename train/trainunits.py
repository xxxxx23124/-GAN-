import math

import torch
import torchvision.utils as vutils
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt
from pathlib import Path


class Units():
    def __init__(self, generator, generator_name, discriminator, discriminator_name, fixed_noise, epoch_len):
        self.generator = generator
        self.generator_name = generator_name
        self.fixed_noise = fixed_noise
        self.discriminator = discriminator
        self.discriminator_name = discriminator_name
        self.optimizer_G = torch.optim.AdamW(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_G, T_max=epoch_len)
        self.optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=0.0004, betas=(0.5, 0.999))
        self.scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_D, T_max=epoch_len)
        plt.ion()
        self.show_model()
        self.epoch = 0
        self.epoch_len = epoch_len
        self.i = 0
        self.record = {}
        self.record.update({'epoch': self.epoch, 'i': self.i})

    def make_record(self, record_set):
        for name, value in record_set:
            list_pos = self.record.get(name, [])
            list_pos.append(value)
            self.record.update({name: list_pos})

    def show_model(self):
        print('# generator parameters:', sum(param.numel() for param in self.generator.parameters()))
        print('# discriminator parameters:', sum(param.numel() for param in self.discriminator.parameters()))

    def show_images(self, gen_imgs, b_size):
        with torch.no_grad():
            plt.clf()
            fake = gen_imgs.cpu()
            plt.imshow(
                vutils.make_grid(fake, nrow=math.ceil(math.sqrt(b_size)),
                                 padding=2, normalize=True).squeeze().permute(1, 2, 0)#, cmap ='gray'
            )
    def save_images(self, path, epoch, i):
        with torch.no_grad():
            gen_imgs = self.generator(self.fixed_noise).cpu()
            save_image(vutils.make_grid(gen_imgs, padding=2, normalize=True).squeeze(),
                       path + str(epoch) + "-" + str(i) + ".png")

    def save_ckpt(self, train_name, epoch, i):
        print('Saving.....')
        with torch.no_grad():
            state = {
                'generator': self.generator,
                'generator_name': self.generator_name,
                'discriminator': self.discriminator,
                'discriminator_name': self.discriminator_name,
                'method': train_name,
                'epoch': epoch + self.epoch + (i + self.i) // self.epoch_len,
                'i': (i + self.i) % self.epoch_len,
            }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state,
                   './checkpoint/' + self.generator_name + ' ' + self.discriminator_name + ' ' + train_name + ' epoch_' +
                   str(epoch + self.epoch + (i + self.i) // self.epoch_len) + ' i_' + str((i + self.i) % self.epoch_len)
                   + '_ckpt.pth')
        state.clear()

    def save_record(self, train_name, epoch, i):
        state = {
            'epoch': epoch + self.epoch + (i + self.i) // self.epoch_len,
            'i': (i + self.i) % self.epoch_len,
            'generator_name': self.generator_name,
            'discriminator_name': self.discriminator_name,
            'method': train_name,
            'record': self.record
        }
        if not os.path.isdir('checkpoint records'):
            os.mkdir('checkpoint records')
        torch.save(state,
                   './checkpoint records/' + self.generator_name + ' ' + self.discriminator_name + ' ' + train_name + ' epoch_' +
                   str(epoch + self.epoch + (i + self.i) // self.epoch_len) + ' i_' + str((i + self.i) % self.epoch_len)
                   + '_record.pth')
        self.record.clear()
        self.record.update({'epoch': self.epoch, 'i': self.i})

    def load_generator_ckpt(self, name):
        print('==> Resuming generator from checkpoint..')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        net_path = './checkpoint/' + name + '.pth'
        ckpt = Path(net_path)
        if ckpt.is_file():
            print('find ckpt')
            checkpoint = torch.load(ckpt)
            print('generator ' + checkpoint['generator_name'])
            self.generator = checkpoint['generator']
            self.epoch = checkpoint['epoch']
            self.i = checkpoint['i']
            self.record.update({'epoch': self.epoch, 'i': self.i})
            print('generator epoch: ', checkpoint['epoch'])
            print('generator i: ', checkpoint['i'])
            checkpoint.clear()
        else:
            print('not find ckpt')

    def load_discriminator_ckpt(self, name):
        print('==> Resuming discriminator from checkpoint..')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        net_path = './checkpoint/' + name + '.pth'
        ckpt = Path(net_path)
        if ckpt.is_file():
            print('find ckpt')
            checkpoint = torch.load(ckpt)
            print('discriminator ' + checkpoint['discriminator_name'])
            self.discriminator = checkpoint['discriminator']
            print('discriminator epoch: ', checkpoint['epoch'])
            print('discriminator i: ', checkpoint['i'])
            checkpoint.clear()
        else:
            print('not find ckpt')



def test():
    net_path = '../checkpoint records/' + '' + '.pth'
    ckpt = Path(net_path)
    if ckpt.is_file():
        print('find ckpt')
        checkpoint = torch.load(ckpt)
        record = checkpoint['record']
        for i in record:
            print(i, record[i])

    else:
        print('not find ckpt')