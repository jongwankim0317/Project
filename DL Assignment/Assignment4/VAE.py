import os
from torchvision.datasets import MNIST
import torchvision.transforms as T
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


mnist_data_dir = './data/mnist'
dataset = MNIST(root=mnist_data_dir,
               transform=T.ToTensor(), train=True, download=True)
print(dataset.train_data.shape)
print(dataset.train_labels.shape)

GPU_NUM = 1
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
print ('Current cuda device ', torch.cuda.current_device()) # check

class Encoder(nn.Module):
    def __init__(self, img_dim=784, latent_dim=30):
        super().__init__()
        self.ec1 = nn.Linear(img_dim, 400)
        self.ec2 = nn.Linear(400, 400)
        self.ec31 = nn.Linear(400, latent_dim)
        self.ec32 = nn.Linear(400, latent_dim)

    def forward(self, imgs):
        x = F.relu(self.ec1(imgs))
        x = F.relu(self.ec2(x))
        mu = self.ec31(x)
        logvar = self.ec32(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=30, img_dim=784):
        super().__init__()
        self.dc1 = nn.Linear(latent_dim, 400)
        self.dc2 = nn.Linear(400, 400)
        self.dc3 = nn.Linear(400, img_dim)

    def forward(self, z):
        x = F.relu(self.dc1(z))
        x = F.relu(self.dc2(x))
        out = F.sigmoid(self.dc3(x))

        return out

class VAE(nn.Module):
    def __init__(self, img_dim=784, latent_dim=10):
        super().__init__()
        self.encoder = Encoder(img_dim, latent_dim)
        self.decoder = Decoder(latent_dim, img_dim)
        self.latent_dim = latent_dim

    def encode(self, imgs):
        mu, logvar = self.encoder(imgs)
        return mu, logvar

    def decode(self, z):
        recon_imgs = self.decoder(z)
        return recon_imgs

    def reparameterize(self, eps, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, imgs):
        mu, logvar = self.encode(imgs)
        eps = torch.randn_like(mu)
        z = self.reparameterize(eps, mu, logvar)
        recon_imgs = self.decode(z)
        return recon_imgs, mu, logvar

    def sample(self, num_samples=16):
        device = next(self.parameters()).device
        eps = torch.randn((num_samples, self.latent_dim)).to(device)
        gen_imgs = self.decode(eps)
        return gen_imgs

# hyperparameter setting
img_dim=784
latent_dim=50

batch_size = 64 # it should be larger than 16
learning_rate = 1e-4
total_iter = 50000

log_freq = 10
viz_freq = 50

from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
dataloader_iter = iter(dataloader)

vae = VAE(img_dim, latent_dim).to(device)

optimizer = torch.optim.Adam(vae.parameters(), learning_rate)

for it in range(total_iter):
    try:
        imgs, _ = next(dataloader_iter)
    except:
        dataloader_iter = iter(dataloader)
        imgs, _ = next(dataloader_iter)

    imgs = imgs.view((-1, img_dim)).to(device)
    recon_imgs, mu, logvar = vae(imgs)

    recon_loss = F.binary_cross_entropy(recon_imgs, imgs.view(-1, 784), reduction='sum')
    kldiv_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = (recon_loss + kldiv_loss)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if (it + 1) % log_freq == 0:
        print("Iter: %05d/%d, Recon loss: %.4f, KL div loss: %.4f, Total loss: %.4f" % (it + 1, total_iter,
                                                                                        recon_loss.data.item(),
                                                                                        kldiv_loss.data.item(),
                                                                                        total_loss.data.item()))

    if (it + 1) % viz_freq == 0:
        with torch.no_grad():
            gen_imgs = vae.sample(16)
        org_imgs = make_grid(imgs[:16, :].view((16, 1, 28, 28)), nrow=4).permute(1, 2, 0).cpu().detach().numpy()
        plt.subplot(1, 3, 1)
        plt.imshow(org_imgs)
        recon_imgs = make_grid(recon_imgs[:16].view((16, 1, 28, 28)), nrow=4).permute(1, 2, 0).cpu().detach().numpy()
        plt.subplot(1, 3, 2)
        plt.imshow(recon_imgs)
        gen_imgs = make_grid(gen_imgs[:16].view((16, 1, 28, 28)), nrow=4).permute(1, 2, 0).cpu().detach().numpy()
        plt.subplot(1, 3, 3)
        plt.imshow(gen_imgs)

        plt.show()
