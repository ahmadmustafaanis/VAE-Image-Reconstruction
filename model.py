import torch
import torch.nn as nn
import torch.nn.functional as F


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    # Reconstruction loss for continuous pixel values [0,1]
    recon_loss = F.mse_loss(recon_x, x, reduction="sum")
    
    # KL divergence
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Beta-VAE: balance reconstruction vs regularization
    return recon_loss + beta * kl_div


class VAE(nn.Module):
    def __init__(self, image_channels=3, latent_dim=128):
        super(VAE, self).__init__()
        # Custom CNN Encoder (lighter than ResNet-18)
        self.encoder = nn.Sequential(
            # Input: 3x32x32
            nn.Conv2d(image_channels, 64, 4, 2, 1),  # 64x16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 128x8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),  # 256x4x4
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),  # 512x2x2
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Flatten(),  # 512*2*2 = 2048
        )
        
        # Progressive latent dimension reduction
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        # Enhanced Decoder with more capacity
        self.decoder_input = nn.Linear(latent_dim, 512)
        self.decoder_fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
        )
        self.unflatten = nn.Unflatten(1, (512, 2, 2))
        
        self.decoder = nn.Sequential(
            # Input: 512x2x2
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 256x4x4
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 128x8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 64x16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 32x32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, image_channels, 3, 1, 1), # 3x32x32
            nn.Tanh(),  # Output range [-1, 1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Normalize input to [-1, 1] for Tanh output
        x = (x - 0.5) * 2.0
        
        # Encode
        x_encoded = self.encoder(x)
        h1 = F.relu(self.fc1(x_encoded))
        h2 = F.relu(self.fc2(h1))
        mu = self.fc_mu(h2)
        logvar = self.fc_logvar(h2)
        z = self.reparameterize(mu, logvar)
        
        # Decode
        h3 = F.relu(self.decoder_input(z))
        h4 = self.decoder_fc(h3)
        h5 = self.unflatten(h4)
        x_reconstructed = self.decoder(h5)
        
        # Convert back to [0, 1] range
        x_reconstructed = (x_reconstructed + 1.0) / 2.0
        
        return x_reconstructed, mu, logvar
