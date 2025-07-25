import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, ResNet


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    Enhanced VAE loss with β-VAE support and better error handling
    """
    batch_size = x.size(0)
    
    # Reconstruction loss - use MSE for continuous pixel values
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence with proper normalization
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # β-VAE formulation with proper scaling
    total_loss = recon_loss + beta * kl_div
    
    # Return normalized loss per sample
    return total_loss / batch_size, recon_loss / batch_size, kl_div / batch_size


class ResidualBlock(nn.Module):
    """Enhanced residual block with batch normalization and dropout"""
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class EnhancedEncoder(nn.Module):
    """Custom encoder designed for VAE reconstruction tasks"""
    def __init__(self, image_channels=3, latent_dim=256):
        super(EnhancedEncoder, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Progressive feature extraction layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Latent space projection
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        # Store intermediate features for skip connections
        features = []
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        features.append(x)  # 64x8x8
        
        x = self.layer1(x)
        features.append(x)  # 64x8x8
        
        x = self.layer2(x)
        features.append(x)  # 128x4x4
        
        x = self.layer3(x)
        features.append(x)  # 256x2x2
        
        x = self.layer4(x)
        features.append(x)  # 512x1x1
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar, features


class UnFlatten(nn.Module):
    def __init__(self, channels, height, width):
        super(UnFlatten, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width
        
    def forward(self, input):
        return input.view(input.size(0), self.channels, self.height, self.width)


class EnhancedDecoder(nn.Module):
    """Enhanced decoder with skip connections and better upsampling"""
    def __init__(self, latent_dim=256, image_channels=3):
        super(EnhancedDecoder, self).__init__()
        
        # Initial projection
        self.decoder_input = nn.Linear(latent_dim, 512)
        self.unflatten = UnFlatten(512, 1, 1)
        
        # Upsampling layers with skip connections
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Skip connection fusion
        self.skip_conv1 = nn.Conv2d(512, 256, kernel_size=1)  # 256 + 256 -> 256
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.skip_conv2 = nn.Conv2d(256, 128, kernel_size=1)  # 128 + 128 -> 128
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.skip_conv3 = nn.Conv2d(128, 64, kernel_size=1)  # 64 + 64 -> 64
        
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.skip_conv4 = nn.Conv2d(96, 32, kernel_size=1)  # 32 + 64 -> 32
        
        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, image_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, z, encoder_features=None):
        x = self.decoder_input(z)
        x = self.unflatten(x)  # 512x1x1
        
        x = self.up1(x)  # 256x2x2
        if encoder_features is not None and len(encoder_features) >= 4:
            # Skip connection from encoder layer3 (256x2x2)
            skip = encoder_features[3]  # 256x2x2
            x = torch.cat([x, skip], dim=1)  # 512x2x2
            x = self.skip_conv1(x)  # 256x2x2
        
        x = self.up2(x)  # 128x4x4
        if encoder_features is not None and len(encoder_features) >= 3:
            # Skip connection from encoder layer2 (128x4x4)
            skip = encoder_features[2]  # 128x4x4
            x = torch.cat([x, skip], dim=1)  # 256x4x4
            x = self.skip_conv2(x)  # 128x4x4
        
        x = self.up3(x)  # 64x8x8
        if encoder_features is not None and len(encoder_features) >= 2:
            # Skip connection from encoder layer1 (64x8x8)
            skip = encoder_features[1]  # 64x8x8
            x = torch.cat([x, skip], dim=1)  # 128x8x8
            x = self.skip_conv3(x)  # 64x8x8
        
        x = self.up4(x)  # 32x16x16
        if encoder_features is not None and len(encoder_features) >= 1:
            # Skip connection from initial conv (64x8x8) -> upsample to 64x16x16
            skip = F.interpolate(encoder_features[0], size=(16, 16), mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)  # 96x16x16
            x = self.skip_conv4(x)  # 32x16x16
        
        # Final upsampling to 32x32
        x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        x = self.final_conv(x)  # 3x32x32
        
        return x


class EnhancedVAE(nn.Module):
    """Enhanced VAE with custom encoder-decoder and skip connections"""
    def __init__(self, image_channels=3, latent_dim=256):
        super(EnhancedVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Enhanced encoder and decoder
        self.encoder = EnhancedEncoder(image_channels, latent_dim)
        self.decoder = EnhancedDecoder(latent_dim, image_channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        mu, logvar, encoder_features = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        
        # Decode with skip connections
        x_reconstructed = self.decoder(z, encoder_features)
        
        return x_reconstructed, mu, logvar


# Keep original VAE for backward compatibility
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
            UnFlatten(512, 1, 1),  # Output: 512x1x1
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
