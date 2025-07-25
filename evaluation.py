import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.manifold import TSNE
import seaborn as sns
from scipy.linalg import sqrtm
from torchvision.models import inception_v3
from torchvision.transforms import Resize, Normalize

from data import ShuffledCIFAR10
from model import VAE

class VAEEvaluator:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.inception_model = None
        
    def _load_model(self, model_path):
        """Load trained VAE model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        model = VAE(image_channels=3, latent_dim=256).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def compute_reconstruction_metrics(self, data_loader, max_batches=50):
        """Compute reconstruction quality metrics"""
        mse_scores = []
        ssim_scores = []
        psnr_scores = []
        
        with torch.no_grad():
            for i, (shuffled_data, original_data) in enumerate(tqdm(data_loader, desc="Computing metrics")):
                if i >= max_batches:
                    break
                    
                shuffled_data = shuffled_data.to(self.device)
                original_data = original_data.to(self.device)
                
                # Get reconstructions
                reconstructed, mu, logvar = self.model(shuffled_data)
                
                # MSE
                mse = F.mse_loss(reconstructed, original_data, reduction='none')
                mse_scores.extend(mse.view(mse.size(0), -1).mean(dim=1).cpu().numpy())
                
                # PSNR
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse.view(mse.size(0), -1).mean(dim=1)))
                psnr_scores.extend(psnr.cpu().numpy())
                
                # SSIM (simplified version)
                ssim = self._compute_ssim(reconstructed, original_data)
                ssim_scores.extend(ssim)
        
        return {
            'mse': np.mean(mse_scores),
            'mse_std': np.std(mse_scores),
            'psnr': np.mean(psnr_scores),
            'psnr_std': np.std(psnr_scores),
            'ssim': np.mean(ssim_scores),
            'ssim_std': np.std(ssim_scores)
        }
    
    def _compute_ssim(self, img1, img2, window_size=11, sigma=1.5):
        """Simplified SSIM computation"""
        # Convert to grayscale for SSIM
        img1_gray = 0.299 * img1[:, 0] + 0.587 * img1[:, 1] + 0.114 * img1[:, 2]
        img2_gray = 0.299 * img2[:, 0] + 0.587 * img2[:, 1] + 0.114 * img2[:, 2]
        
        mu1 = F.avg_pool2d(img1_gray.unsqueeze(1), window_size, stride=1, padding=window_size//2)
        mu2 = F.avg_pool2d(img2_gray.unsqueeze(1), window_size, stride=1, padding=window_size//2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1_gray.unsqueeze(1).pow(2), window_size, stride=1, padding=window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2_gray.unsqueeze(1).pow(2), window_size, stride=1, padding=window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(img1_gray.unsqueeze(1) * img2_gray.unsqueeze(1), window_size, stride=1, padding=window_size//2) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean(dim=[1, 2, 3]).cpu().numpy().tolist()
    
    def visualize_reconstructions(self, data_loader, num_samples=8, save_path=None):
        """Visualize reconstruction results"""
        self.model.eval()
        
        with torch.no_grad():
            shuffled_data, original_data = next(iter(data_loader))
            shuffled_data = shuffled_data[:num_samples].to(self.device)
            original_data = original_data[:num_samples].to(self.device)
            
            reconstructed, _, _ = self.model(shuffled_data)
            
            # Create visualization
            fig, axes = plt.subplots(3, num_samples, figsize=(2*num_samples, 6))
            
            for i in range(num_samples):
                # Shuffled images
                axes[0, i].imshow(np.transpose(shuffled_data[i].cpu().numpy(), (1, 2, 0)))
                axes[0, i].set_title(f'Shuffled {i+1}')
                axes[0, i].axis('off')
                
                # Original images
                axes[1, i].imshow(np.transpose(original_data[i].cpu().numpy(), (1, 2, 0)))
                axes[1, i].set_title(f'Original {i+1}')
                axes[1, i].axis('off')
                
                # Reconstructed images
                axes[2, i].imshow(np.transpose(reconstructed[i].cpu().numpy(), (1, 2, 0)))
                axes[2, i].set_title(f'Reconstructed {i+1}')
                axes[2, i].axis('off')
            
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.show()
    
    def visualize_latent_space(self, data_loader, num_samples=1000, save_path=None):
        """Visualize latent space using t-SNE"""
        latent_vectors = []
        labels = []
        
        self.model.eval()
        with torch.no_grad():
            for i, (shuffled_data, original_data) in enumerate(tqdm(data_loader, desc="Extracting latent vectors")):
                if len(latent_vectors) >= num_samples:
                    break
                
                shuffled_data = shuffled_data.to(self.device)
                
                # Get latent representation
                encoded = self.model.encoder(shuffled_data)
                mu = self.model.fc_mu(encoded)
                
                latent_vectors.extend(mu.cpu().numpy())
                # Use first 10 classes for visualization
                labels.extend([i % 10 for _ in range(shuffled_data.size(0))])
        
        latent_vectors = np.array(latent_vectors[:num_samples])
        labels = np.array(labels[:num_samples])
        
        # Apply t-SNE
        print("Applying t-SNE...")
        tsne = TSNE(n_components=2, random_state=42)
        latent_2d = tsne.fit_transform(latent_vectors)
        
        # Visualize
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter)
        plt.title('t-SNE Visualization of Latent Space')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def latent_interpolation(self, data_loader, steps=10, save_path=None):
        """Generate latent space interpolation"""
        self.model.eval()
        
        with torch.no_grad():
            # Get two random samples
            shuffled_data, original_data = next(iter(data_loader))
            shuffled_data = shuffled_data[:2].to(self.device)
            
            # Encode to latent space
            encoded = self.model.encoder(shuffled_data)
            mu = self.model.fc_mu(encoded)
            
            # Interpolate between two latent vectors
            z1, z2 = mu[0], mu[1]
            interpolations = []
            
            for i in range(steps):
                alpha = i / (steps - 1)
                z_interp = (1 - alpha) * z1 + alpha * z2
                reconstructed = self.model.decoder(z_interp.unsqueeze(0))
                interpolations.append(reconstructed[0])
            
            # Visualize interpolation
            fig, axes = plt.subplots(2, steps, figsize=(2*steps, 4))
            
            # Original images
            axes[0, 0].imshow(np.transpose(shuffled_data[0].cpu().numpy(), (1, 2, 0)))
            axes[0, 0].set_title('Start (Shuffled)')
            axes[0, 0].axis('off')
            
            axes[0, -1].imshow(np.transpose(shuffled_data[1].cpu().numpy(), (1, 2, 0)))
            axes[0, -1].set_title('End (Shuffled)')
            axes[0, -1].axis('off')
            
            # Hide middle axes in first row
            for i in range(1, steps-1):
                axes[0, i].axis('off')
            
            # Interpolated reconstructions
            for i, recon in enumerate(interpolations):
                axes[1, i].imshow(np.transpose(recon.cpu().numpy(), (1, 2, 0)))
                axes[1, i].set_title(f'Step {i+1}')
                axes[1, i].axis('off')
            
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.show()
    
    def generate_evaluation_report(self, test_loader, save_dir='evaluation_results'):
        """Generate comprehensive evaluation report"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("Generating comprehensive evaluation report...")
        
        # 1. Reconstruction Metrics
        print("Computing reconstruction metrics...")
        metrics = self.compute_reconstruction_metrics(test_loader)
        
        # 2. Visualizations
        print("Generating visualizations...")
        self.visualize_reconstructions(test_loader, num_samples=8, 
                                     save_path=f'{save_dir}/reconstructions.png')
        
        self.visualize_latent_space(test_loader, num_samples=1000,
                                  save_path=f'{save_dir}/latent_space.png')
        
        self.latent_interpolation(test_loader, steps=10,
                                save_path=f'{save_dir}/interpolation.png')
        
        # 3. Save metrics report
        report = f"""
VAE Model Evaluation Report
==========================

Reconstruction Metrics:
- MSE: {metrics['mse']:.6f} ± {metrics['mse_std']:.6f}
- PSNR: {metrics['psnr']:.2f} ± {metrics['psnr_std']:.2f} dB
- SSIM: {metrics['ssim']:.4f} ± {metrics['ssim_std']:.4f}

Model Architecture:
- Latent Dimension: 256
- Enhanced Encoder with Attention
- Residual Decoder with Batch Normalization
- β-VAE with Perceptual Loss

Training Improvements:
- Learning Rate Scheduling
- Gradient Clipping
- KL Annealing
- AdamW Optimizer with Weight Decay

Visualizations saved:
- reconstructions.png: Sample reconstructions
- latent_space.png: t-SNE visualization of latent space
- interpolation.png: Latent space interpolation
        """
        
        with open(f'{save_dir}/evaluation_report.txt', 'w') as f:
            f.write(report)
        
        print(f"Evaluation report saved to {save_dir}/")
        print(report)
        
        return metrics

def main():
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test data
    test_dataset = ShuffledCIFAR10(train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize evaluator (you'll need to provide the path to your trained model)
    model_path = 'checkpoints/best_model_epoch_500.pth'  # Update with actual path
    
    try:
        evaluator = VAEEvaluator(model_path, device=device)
        
        # Generate comprehensive evaluation
        metrics = evaluator.generate_evaluation_report(test_loader)
        
    except FileNotFoundError:
        print(f"Model checkpoint not found at {model_path}")
        print("Please train the model first using train_enhanced.py")

if __name__ == "__main__":
    main()