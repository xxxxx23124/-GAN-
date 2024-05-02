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
        self.optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.99))
        self.optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0004, betas=(0.0, 0.99))
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

    def write_record_to_txt(self, file_name, record_set):
        with open(file_name + '.txt', 'a', encoding='utf-8') as file:
            for name, value in record_set:
                file.write(name + ' ' + str(value) + ' ')
            file.write('\n')

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

    def draw_plt_record(self, name_png, per_iteration):
        fig1 = plt.figure(num=name_png, figsize=(16, 9), clear=True)
        ax = fig1.add_subplot(111)
        i = 0
        linestyles = [
            (0, (3, 3, 1, 2)),
            (0, (1, 1)),
            (0, (5, 5)),
            (0, (5, 3, 1, 2)),
            (0, (3, 1, 1, 1, 2, 1)),
            (0, (3, 4, 1, 2, 1, 2)),
        ]
        for key in self.record:
            if (key != 'epoch') and (key != 'i'):
                ax.plot(self.record[key], label=key, linewidth=1.0, linestyle=linestyles[i])
                i = i + 1

        ax.legend(fontsize='x-large', loc=1)
        ax.set_xlabel('every ' + str(per_iteration) + ' iterations')
        ax.set_ylabel('loss amount')
        plt.savefig(name_png + '.png', dpi=960)



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
