import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, ResNet


def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss (assuming Bernoulli distribution)
    try:
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
        # KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    except Exception as E:
        print(E)
        try:
            recon_loss = F.mse_loss(recon_x, x, reduction="sum")
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        except Exception as E:
            print(E)
            # set loss to a default value
            recon_loss = torch.tensor(0.1).to(recon_x.device)
            kl_div = torch.tensor(0.1).to(recon_x.device)
    return recon_loss + kl_div


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 512, 1, 1)


class VAE(nn.Module):
    def __init__(self, image_channels=3, latent_dim=128):
        super(VAE, self).__init__()
        # Encoder (ResNet-18)
        self.encoder = nn.Sequential(
            *list(ResNet(BasicBlock, [2, 2, 2, 2]).children())[:-1], nn.Flatten()
        )
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 512)
        self.decoder = nn.Sequential(
            UnFlatten(),  # Output: 512x1x1
            nn.ConvTranspose2d(
                512, 256, kernel_size=4, stride=2, padding=1
            ),  # Output: 256x2x2
            nn.ReLU(),
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1
            ),  # Output: 128x4x4
            nn.ReLU(),
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),  # Output: 64x8x8
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),  # Output: 32x16x16
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, image_channels, kernel_size=4, stride=2, padding=1
            ),  # Output: 3x32x32
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        x_encoded = self.encoder(x)
        mu = self.fc_mu(x_encoded)
        logvar = self.fc_logvar(x_encoded)
        z = self.reparameterize(mu, logvar)
        # Decode
        x_reconstructed = self.decoder(self.decoder_input(z))
        return x_reconstructed, mu, logvar
