import torch
from tqdm.tk import tqdm
from train import trainunits


class Train(trainunits.Units):

    def __init__(self, dataloader, device, num_epochs, nz, generator, generator_name, discriminator,
                 discriminator_name):
        super(Train, self).__init__(generator, generator_name, discriminator, discriminator_name,
                                    torch.randn(16, nz, 1, 1, device=device), len(dataloader))
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

    def gradient_penalty(self, x_real, x_fake, batch_size, device, center=1.):
        eps = torch.rand(batch_size, device=device).view(batch_size, 1, 1, 1)
        x_interp = (1 - eps) * x_real + eps * x_fake
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out = self.discriminator(x_interp)

        reg = (self.compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()

        return reg

    def compute_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_in.size())
        reg = grad_dout2.view(batch_size, -1).sum(1)
        return reg

    def discriminator_trainstep(self, images, b_size, idx):
        self.optimizer_D.zero_grad()
        z = torch.randn(b_size, self.nz, 1, 1, device=self.device)
        with torch.no_grad():
            gen_imgs = self.generator(z)
        gen_imgs.requires_grad_()
        pred_r = self.discriminator(images)
        real_loss = -torch.mean(pred_r)
        r2_reg_r = torch.zeros(1, device=self.device)
        if idx % 5 == 0:
            real_loss.backward(retain_graph=True)
            r2_reg_r = 5 * self.compute_grad2(pred_r, images).mean()
            r2_reg_r.backward()
        else:
            real_loss.backward()
        pred_f = self.discriminator(gen_imgs)
        fake_loss = torch.mean(pred_f)
        r2_reg_f = torch.zeros(1, device=self.device)
        if idx % 5 == 0:
            fake_loss.backward(retain_graph=True)
            r2_reg_f = 5 * self.compute_grad2(pred_f, gen_imgs).mean()
            r2_reg_f.backward()
        else:
            fake_loss.backward()
        gp = torch.zeros(1, device=self.device)
        if idx % 5 == 0:
            gp = 10 * self.gradient_penalty(images, gen_imgs, b_size, self.device) * 5
            gp.backward()
        self.optimizer_D.step()
        return real_loss, fake_loss, gp, r2_reg_r, r2_reg_f

    def train(self):
        self.load_generator_ckpt('')
        self.load_discriminator_ckpt('')
        proc_bar = tqdm(total=len(self.dataloader))
        print("Starting Training Loop...")
        for epoch in range(self.num_epochs):
            for i, (images, _) in enumerate(self.dataloader):
                images = images.to(self.device).requires_grad_()
                b_size = images.shape[0]

                real_loss, fake_loss, gp, r2_reg_r, r2_reg_f = self.discriminator_trainstep(images, b_size, i)
                # if i % 3 == 0:
                gen_imgs, g_loss = self.generator_trainstep(b_size)
                #if i % 100 == 0:
                    #self.show_images(torch.cat([images, gen_imgs], dim=0), b_size*2)
                if i % 30 == 0:
                    record = [
                        ('Discriminator real loss', real_loss.item()),
                        ('Discriminator fake loss', fake_loss.item()),
                        ('Gradient penalties', gp.item()),
                        ('Simplified gradient penalties R1', r2_reg_r.item()),
                        ('Simplified gradient penalties R2', r2_reg_f.item()),
                        ('Generator loss', g_loss.item()),
                        ]
                    self.make_record(record)
                    self.write_record_to_txt("wganlazygpR2",record)
                if i % 30 == 0:
                    self.save_images("generated_images/", epoch, i)

                proc_bar.set_postfix(
                    {"epoch": f"{epoch}", "Loss_G": f"{g_loss.item():.4f}", "real_loss": f"{real_loss.item():.4f}",
                     "fake_loss": f"{fake_loss.item():.4f}", "gp": f"{gp.item():.4f}",
                     "r2_reg_r": f"{r2_reg_r.item():.4f}",
                     "r2_reg_f": f"{r2_reg_f.item():.4f}"})

                proc_bar.update(1)
            proc_bar.reset()
        self.draw_plt_record("wganlazygpR2", 30)
        proc_bar.close()
        #self.save_ckpt('WGANGPR2', self.num_epochs, 0)
