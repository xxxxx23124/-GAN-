'''
原始GAN训练
继承trainunits的Units类
'''
import torch
from tqdm.tk import tqdm
from train import trainunits


class Train(trainunits.Units):

    def __init__(self, dataloader, device, num_epochs, nz, generator, generator_name, discriminator,
                 discriminator_name):
        super(Train, self).__init__(generator, generator_name, discriminator, discriminator_name,
                                    torch.randn(64, nz, 1, 1, device=device), len(dataloader))

        self.dataloader = dataloader
        self.device = device
        self.num_epochs = num_epochs
        self.nz = nz
        self.criterion = torch.nn.BCELoss().to(device)

        # self.optimizer_G = torch.optim.SGD(generator.parameters(), lr=0.0001)
        # self.optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=0.0004)

    def generator_trainstep(self, b_size):
        valid = torch.full((b_size, 1), 0.95, dtype=torch.float32, device=self.device)
        valid_ = valid + 0.05 * torch.rand(valid.shape, dtype=torch.float32, device=self.device)
        self.optimizer_G.zero_grad()
        z = torch.randn(b_size, self.nz, 1, 1, device=self.device)
        gen_imgs = self.generator(z)
        g_loss = self.criterion(self.discriminator(gen_imgs), valid_)
        g_loss.backward()
        self.optimizer_G.step()
        return gen_imgs, g_loss

    def discriminator_trainstep(self, images, b_size):
        valid = torch.full((b_size, 1), 0.95, dtype=torch.float32, device=self.device)
        valid_ = valid + 0.05 * torch.rand(valid.shape, dtype=torch.float32, device=self.device)
        fake = torch.full((b_size, 1), 0., dtype=torch.float32, device=self.device)
        fake_ = fake + 0.05 * torch.rand(fake.shape, dtype=torch.float32, device=self.device)
        z = torch.randn(b_size, self.nz, 1, 1, device=self.device)

        self.optimizer_D.zero_grad()
        with torch.no_grad():
            gen_imgs = self.generator(z)
        gen_imgs.requires_grad_()
        real_loss = self.criterion(self.discriminator(images), valid_)
        real_loss.backward()
        fake_loss = self.criterion(self.discriminator(gen_imgs), fake_)
        fake_loss.backward()
        self.optimizer_D.step()
        return real_loss, fake_loss

    def train(self):
        self.load_generator_ckpt('')
        self.load_discriminator_ckpt('')
        proc_bar = tqdm(total=len(self.dataloader))
        print("Starting Training Loop...")
        for epoch in range(self.num_epochs):
            for i, (images, _) in enumerate(self.dataloader):
                images = images.to(self.device)
                b_size = images.shape[0]

                real_loss, fake_loss = self.discriminator_trainstep(images, b_size)
                gen_imgs, g_loss = self.generator_trainstep(b_size)

                self.show_images(torch.cat([images, gen_imgs]), b_size * 2)
                # if i % 10 == 0:
                # self.save_images("generated_images/", epoch, i)
                proc_bar.set_postfix(
                    {"epoch": f"{epoch}", "Loss_G": f"{g_loss.item():.4f}",
                     "Real_loss": f"{real_loss.item():.4f}", "Fake_loss": f"{fake_loss.item():.4f}"})
                self.scheduler_G.step()
                self.scheduler_D.step()
                proc_bar.update(1)
            # self.save_ckpt('GAN', epoch + 1, 0)
            proc_bar.reset()
        proc_bar.close()
