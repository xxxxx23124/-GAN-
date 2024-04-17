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
        pred_r = self.discriminator(images)
        real_loss = self.criterion(pred_r, valid_)
        real_loss.backward(retain_graph=True)
        r2_reg_r = self.compute_grad2(pred_r, images).mean()
        r2_reg_r.backward()
        pred_f = self.discriminator(gen_imgs)
        fake_loss = self.criterion(pred_f, fake_)
        fake_loss.backward(retain_graph=True)
        r2_reg_f = self.compute_grad2(pred_f, gen_imgs).mean()
        r2_reg_f.backward()
        self.optimizer_D.step()
        return real_loss, fake_loss, r2_reg_r, r2_reg_f

    def train(self):
        self.load_generator_ckpt('')
        self.load_discriminator_ckpt('')
        proc_bar = tqdm(total=len(self.dataloader))
        print("Starting Training Loop...")
        for epoch in range(self.num_epochs):
            for i, (images, _) in enumerate(self.dataloader):
                images = images.to(self.device).requires_grad_()
                b_size = images.shape[0]

                real_loss, fake_loss, r2_reg_r, r2_reg_f = self.discriminator_trainstep(images, b_size)
                gen_imgs, g_loss = self.generator_trainstep(b_size)
                self.show_images(gen_imgs, b_size)

                # if i % 10 == 0:
                # self.save_images("generated_images/", epoch, i)
                '''
                self.make_record([
                    ('real_loss', real_loss.item()),
                    ('fake_loss', fake_loss.item()),
                    ('r2_reg_r', r2_reg_r.item()),
                    ('r2_reg_f', r2_reg_f.item()),
                    ('g_loss', g_loss.item()),
                ])'''
                proc_bar.set_postfix(
                    {"epoch": f"{epoch}", "Loss_G": f"{g_loss.item():.4f}", "real_loss": f"{real_loss.item():.4f}",
                     "fake_loss": f"{fake_loss.item():.4f}", "r2_reg_r": f"{r2_reg_r.item():.4f}",
                     "r2_reg_f": f"{r2_reg_f.item():.4f}"})
                self.scheduler_G.step()
                self.scheduler_D.step()
                proc_bar.update(1)
            # self.save_record('GANR2', epoch + 1, 0)
            # self.save_ckpt('GANR2', epoch + 1, 0)
            proc_bar.reset()
        proc_bar.close()

# print('[%d/%d][%d/%d]\tLoss_G: %.4f\tLoss_D: %.4f\tReal_loss: %.4f\tFake_loss: %.4f' %
# (epoch, num_epochs, i, len(dataloader), g_loss.item(), d_loss.item(), real_loss.item(),
# fake_loss.item()))
