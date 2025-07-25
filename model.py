import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, ResNet
import torchvision.models as models
import math


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.feature_layers = nn.Sequential(*list(vgg.features)[:23]).eval()
        for param in self.feature_layers.parameters():
            param.requires_grad = False
    
    def forward(self, x, y):
        x_features = self.feature_layers(x)
        y_features = self.feature_layers(y)
        return F.mse_loss(x_features, y_features)

def beta_vae_loss(recon_x, x, mu, logvar, beta=1.0, perceptual_loss_fn=None, perceptual_weight=0.1):
    batch_size = x.size(0)
    
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / batch_size
    
    # KL divergence
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    
    # Perceptual loss if provided
    perceptual_loss = 0
    if perceptual_loss_fn is not None:
        # Ensure images are in the right format for VGG (3 channels)
        if recon_x.size(1) == 3:
            perceptual_loss = perceptual_loss_fn(recon_x, x) * perceptual_weight
    
    return recon_loss + beta * kl_div + perceptual_loss, recon_loss, kl_div, perceptual_loss

def vae_loss(recon_x, x, mu, logvar):
    # Backward compatibility wrapper
    loss, _, _, _ = beta_vae_loss(recon_x, x, mu, logvar, beta=1.0)
    return loss


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)
        
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        return self.gamma * out + x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class EnhancedEncoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=128):
        super(EnhancedEncoder, self).__init__()
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # Residual blocks with progressive feature extraction
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )
        
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128)
        )
        
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
            SelfAttention(256)  # Add attention after conv layers
        )
        
        self.layer4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512),
            SelfAttention(512)  # Add attention for high-level features
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1024)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dropout(x)
        
        return x

class UnFlatten(nn.Module):
    def __init__(self, size):
        super(UnFlatten, self).__init__()
        self.size = size
        
    def forward(self, input):
        return input.view(input.size(0), self.size, 1, 1)


class EnhancedDecoder(nn.Module):
    def __init__(self, latent_dim=128, output_channels=3):
        super(EnhancedDecoder, self).__init__()
        
        self.decoder_input = nn.Linear(latent_dim, 512 * 4 * 4)
        
        self.decoder = nn.Sequential(
            # 512x4x4 -> 256x8x8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 256x8x8 -> 128x16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 128x16x16 -> 64x32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Final layer: 64x32x32 -> 3x32x32
            nn.ConvTranspose2d(64, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.decoder_input(x)
        x = x.view(x.size(0), 512, 4, 4)
        return self.decoder(x)

class VAE(nn.Module):
    def __init__(self, image_channels=3, latent_dim=256):
        super(VAE, self).__init__()
        
        # Enhanced encoder with attention
        self.encoder = EnhancedEncoder(image_channels, latent_dim)
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)
        
        # Enhanced decoder with batch norm
        self.decoder = EnhancedDecoder(latent_dim, image_channels)
        
        # Perceptual loss for better reconstruction
        self.perceptual_loss = PerceptualLoss()
        
        # Learnable beta for beta-VAE
        self.log_beta = nn.Parameter(torch.tensor(0.0))  # Initial beta = 1.0

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @property
    def beta(self):
        return torch.exp(self.log_beta)
    
    def forward(self, x):
        # Encode
        x_encoded = self.encoder(x)
        mu = self.fc_mu(x_encoded)
        logvar = self.fc_logvar(x_encoded)
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar
    
    def compute_loss(self, x_reconstructed, x, mu, logvar, use_perceptual=True):
        perceptual_loss_fn = self.perceptual_loss if use_perceptual else None
        return beta_vae_loss(
            x_reconstructed, x, mu, logvar, 
            beta=self.beta, 
            perceptual_loss_fn=perceptual_loss_fn
        )
