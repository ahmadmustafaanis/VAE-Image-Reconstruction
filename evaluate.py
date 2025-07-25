#!/usr/bin/env python3
"""
Evaluation script for the Enhanced VAE model
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import argparse
import os

from data import ShuffledCIFAR10
from model import EnhancedVAE, VAE

def load_model(checkpoint_path, model_type='enhanced', latent_dim=256):
    """Load trained model from checkpoint"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == 'enhanced':
        model = EnhancedVAE(image_channels=3, latent_dim=latent_dim).to(device)
    else:
        model = VAE(image_channels=3, latent_dim=128).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, device

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

def evaluate_model(model, data_loader, device):
    """Evaluate model performance"""
    model.eval()
    total_ssim = 0
    total_psnr = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            recon_batch, _, _ = model(data)
            
            ssim_score, psnr_score = calculate_metrics(recon_batch, target)
            total_ssim += ssim_score * data.size(0)
            total_psnr += psnr_score * data.size(0)
            total_samples += data.size(0)
    
    avg_ssim = total_ssim / total_samples
    avg_psnr = total_psnr / total_samples
    
    return avg_ssim, avg_psnr

def visualize_results(model, data_loader, device, num_samples=8, save_path=None):
    """Visualize reconstruction results"""
    model.eval()
    
    with torch.no_grad():
        data, target = next(iter(data_loader))
        data, target = data.to(device), target.to(device)
        recon_batch, _, _ = model(data)
        
        # Select first num_samples
        data = data[:num_samples]
        target = target[:num_samples]
        recon_batch = recon_batch[:num_samples]
        
        # Create visualization
        fig, axes = plt.subplots(3, num_samples, figsize=(2*num_samples, 6))
        
        for i in range(num_samples):
            # Shuffled input
            shuffled_img = data[i].cpu().numpy().transpose(1, 2, 0)
            axes[0, i].imshow(np.clip(shuffled_img, 0, 1))
            axes[0, i].set_title('Shuffled')
            axes[0, i].axis('off')
            
            # Original target
            original_img = target[i].cpu().numpy().transpose(1, 2, 0)
            axes[1, i].imshow(np.clip(original_img, 0, 1))
            axes[1, i].set_title('Original')
            axes[1, i].axis('off')
            
            # Reconstruction
            recon_img = recon_batch[i].cpu().numpy().transpose(1, 2, 0)
            axes[2, i].imshow(np.clip(recon_img, 0, 1))
            axes[2, i].set_title('Reconstructed')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()

def main():
    parser = argparse.ArgumentParser(description='Evaluate VAE model')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='enhanced',
                       choices=['enhanced', 'original'],
                       help='Type of model to evaluate')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--save_viz', type=str, default=None,
                       help='Path to save visualization')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading {args.model_type} model from {args.checkpoint}")
    model, device = load_model(args.checkpoint, args.model_type, args.latent_dim)
    
    # Load test data
    test_dataset = ShuffledCIFAR10(train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Evaluate model
    print("Evaluating model...")
    avg_ssim, avg_psnr = evaluate_model(model, test_loader, device)
    
    print(f"\nEvaluation Results:")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    
    # Visualize results
    print("Generating visualizations...")
    visualize_results(model, test_loader, device, save_path=args.save_viz)

if __name__ == "__main__":
    main()