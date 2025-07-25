#!/usr/bin/env python3
"""
Compare performance between original VAE and Enhanced VAE
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from data import ShuffledCIFAR10
from model import VAE, EnhancedVAE

def calculate_metrics(pred, target):
    """Calculate SSIM and PSNR metrics"""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    batch_size = pred_np.shape[0]
    ssim_scores = []
    psnr_scores = []
    
    for i in range(batch_size):
        pred_img = np.transpose(pred_np[i], (1, 2, 0))
        target_img = np.transpose(target_np[i], (1, 2, 0))
        
        pred_img = np.clip(pred_img, 0, 1)
        target_img = np.clip(target_img, 0, 1)
        
        ssim_score = ssim(target_img, pred_img, multichannel=True, data_range=1.0)
        ssim_scores.append(ssim_score)
        
        psnr_score = psnr(target_img, pred_img, data_range=1.0)
        psnr_scores.append(psnr_score)
    
    return np.mean(ssim_scores), np.mean(psnr_scores)

def test_untrained_models():
    """Test both models without training to compare architectures"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize models
    original_vae = VAE(image_channels=3, latent_dim=128).to(device)
    enhanced_vae = EnhancedVAE(image_channels=3, latent_dim=256).to(device)
    
    # Set to eval mode
    original_vae.eval()
    enhanced_vae.eval()
    
    # Load test data
    test_dataset = ShuffledCIFAR10(train=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print("Comparing untrained model architectures...")
    print(f"Original VAE parameters: {sum(p.numel() for p in original_vae.parameters()):,}")
    print(f"Enhanced VAE parameters: {sum(p.numel() for p in enhanced_vae.parameters()):,}")
    
    # Test with a small batch
    with torch.no_grad():
        data, target = next(iter(test_loader))
        data, target = data.to(device), target.to(device)
        
        # Original VAE
        recon_orig, _, _ = original_vae(data)
        ssim_orig, psnr_orig = calculate_metrics(recon_orig, target)
        
        # Enhanced VAE
        recon_enh, _, _ = enhanced_vae(data)
        ssim_enh, psnr_enh = calculate_metrics(recon_enh, target)
        
        print(f"\nUntrained Performance (random weights):")
        print(f"Original VAE - SSIM: {ssim_orig:.4f}, PSNR: {psnr_orig:.2f} dB")
        print(f"Enhanced VAE - SSIM: {ssim_enh:.4f}, PSNR: {psnr_enh:.2f} dB")
        
        # Visualize comparison
        fig, axes = plt.subplots(4, 8, figsize=(16, 8))
        
        for i in range(8):
            # Shuffled input
            shuffled_img = data[i].cpu().numpy().transpose(1, 2, 0)
            axes[0, i].imshow(np.clip(shuffled_img, 0, 1))
            if i == 0:
                axes[0, i].set_ylabel('Shuffled')
            axes[0, i].axis('off')
            
            # Original target
            original_img = target[i].cpu().numpy().transpose(1, 2, 0)
            axes[1, i].imshow(np.clip(original_img, 0, 1))
            if i == 0:
                axes[1, i].set_ylabel('Original')
            axes[1, i].axis('off')
            
            # Original VAE reconstruction
            recon_orig_img = recon_orig[i].cpu().numpy().transpose(1, 2, 0)
            axes[2, i].imshow(np.clip(recon_orig_img, 0, 1))
            if i == 0:
                axes[2, i].set_ylabel('Original VAE')
            axes[2, i].axis('off')
            
            # Enhanced VAE reconstruction
            recon_enh_img = recon_enh[i].cpu().numpy().transpose(1, 2, 0)
            axes[3, i].imshow(np.clip(recon_enh_img, 0, 1))
            if i == 0:
                axes[3, i].set_ylabel('Enhanced VAE')
            axes[3, i].axis('off')
        
        plt.suptitle('Model Architecture Comparison (Untrained)', fontsize=16)
        plt.tight_layout()
        plt.savefig('model_comparison_untrained.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\nComparison visualization saved as 'model_comparison_untrained.png'")

def analyze_model_complexity():
    """Analyze the complexity and architecture differences"""
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE ANALYSIS")
    print("="*60)
    
    # Original VAE
    original_vae = VAE(image_channels=3, latent_dim=128)
    original_params = sum(p.numel() for p in original_vae.parameters())
    
    # Enhanced VAE  
    enhanced_vae = EnhancedVAE(image_channels=3, latent_dim=256)
    enhanced_params = sum(p.numel() for p in enhanced_vae.parameters())
    
    print(f"Original VAE:")
    print(f"  - Latent dimension: 128")
    print(f"  - Architecture: ResNet-18 backbone + simple decoder")
    print(f"  - Parameters: {original_params:,}")
    print(f"  - Features: Basic architecture, no skip connections")
    
    print(f"\nEnhanced VAE:")
    print(f"  - Latent dimension: 256 (2x larger)")
    print(f"  - Architecture: Custom encoder + decoder with skip connections")
    print(f"  - Parameters: {enhanced_params:,}")
    print(f"  - Features: Residual blocks, batch norm, dropout, skip connections")
    
    param_increase = (enhanced_params - original_params) / original_params * 100
    print(f"\nParameter increase: {param_increase:.1f}%")
    
    print(f"\nKey Improvements in Enhanced VAE:")
    print(f"  1. Larger latent space (256 vs 128) for better representation")
    print(f"  2. Skip connections between encoder-decoder for better gradients")
    print(f"  3. Batch normalization and dropout for better training stability")
    print(f"  4. Custom residual blocks designed for reconstruction tasks")
    print(f"  5. β-VAE loss formulation with proper error handling")
    print(f"  6. Progressive feature extraction with attention to spatial details")

if __name__ == "__main__":
    analyze_model_complexity()
    test_untrained_models()