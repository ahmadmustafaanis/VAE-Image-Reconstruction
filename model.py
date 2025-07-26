import torch
import torch.nn as nn
import torch.nn.functional as F


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    Compute VAE loss with proper error handling and beta-VAE support.
    
    Args:
        recon_x: Reconstructed images
        x: Original images  
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence term (beta-VAE)
    """
    # Ensure inputs are in valid range [0, 1]
    recon_x = torch.clamp(recon_x, 0, 1)
    x = torch.clamp(x, 0, 1)
    
    # Reconstruction loss - use MSE for continuous values
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence loss
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Beta-VAE: weight the KL divergence
    total_loss = recon_loss + beta * kl_div
    
    return total_loss, recon_loss, kl_div


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 512, 1, 1)


class VAE(nn.Module):
    def __init__(self, image_channels=3, latent_dim=256):
        super(VAE, self).__init__()
        
        # Custom Encoder (symmetric to decoder)
        self.encoder = nn.Sequential(
            # Input: 3x32x32 -> 32x16x16
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # 32x16x16 -> 64x8x8
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # 64x8x8 -> 128x4x4
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # 128x4x4 -> 256x2x2
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # 256x2x2 -> 512x1x1
            nn.Conv2d(256, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Flatten()
        )
        
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        # Decoder (improved with batch norm and skip connections concept)
        self.decoder_input = nn.Linear(latent_dim, 512)
        self.decoder = nn.Sequential(
            UnFlatten(),  # Output: 512x1x1
            
            # 512x1x1 -> 256x2x2
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # 256x2x2 -> 128x4x4
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # 128x4x4 -> 64x8x8
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # 64x8x8 -> 32x16x16
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # 32x16x16 -> 3x32x32
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),
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
