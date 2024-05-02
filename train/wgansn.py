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

    def generator_trainstep(self, b_size):
        self.optimizer_G.zero_grad()
        z = torch.randn(b_size, self.nz, 1, 1, device=self.device)
        gen_imgs = self.generator(z)
        g_loss = -torch.mean(self.discriminator(gen_imgs))
        g_loss.backward()
        self.optimizer_G.step()
        return gen_imgs, g_loss

    def discriminator_trainstep(self, images, b_size):
        z = torch.randn(b_size, self.nz, 1, 1, device=self.device)
        self.optimizer_D.zero_grad()
        with torch.no_grad():
            gen_imgs = self.generator(z)
        gen_imgs.requires_grad_()
        pred_r = self.discriminator(images)
        real_loss = -torch.mean(pred_r)
        real_loss.backward()
        pred_f = self.discriminator(gen_imgs)
        fake_loss = torch.mean(pred_f)
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
                images = images.to(self.device).requires_grad_()
                b_size = images.shape[0]

                real_loss, fake_loss = self.discriminator_trainstep(images, b_size)
                gen_imgs, g_loss = self.generator_trainstep(b_size)
                self.show_images(gen_imgs, b_size)

                if i % 10 == 0:
                    self.save_images("generated_images/", epoch, i)
                proc_bar.set_postfix(
                    {"epoch": f"{epoch}", "Loss_G": f"{g_loss.item():.4f}", "real_loss": f"{real_loss.item():.4f}",
                     "fake_loss": f"{fake_loss.item():.4f}"})
                proc_bar.update(1)
            self.save_ckpt('WGANSN', epoch + 1, 0)
            proc_bar.reset()
        proc_bar.close()
