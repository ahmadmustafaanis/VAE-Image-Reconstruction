import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import wandb
from data import ShuffledCIFAR10
from model import EnhancedVAE, vae_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enhanced hyperparameters
EPOCHS = 1000  # More reasonable number with better architecture
BATCH_SIZE = 32  # Smaller batch size for better gradients
LATENT_DIM = 256  # Increased latent dimension
LEARNING_RATE = 1e-3
BETA = 1.0  # β-VAE parameter
GRADIENT_CLIP = 1.0

# Create checkpoint directory
os.makedirs('checkpoints', exist_ok=True)

train_dataset = ShuffledCIFAR10(train=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

val_dataset = ShuffledCIFAR10(train=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Enhanced model
model = EnhancedVAE(image_channels=3, latent_dim=LATENT_DIM).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Improved optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)


def show_images(shuffled, original, reconstructed):
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    for ax, img, title in zip(
        axs,
        [shuffled, original, reconstructed],
        ["Shuffled", "Original", "Reconstructed"],
    ):
        ax.imshow(np.transpose(img.numpy(), (1, 2, 0)))
        ax.set_title(title)
        ax.axis("off")
    plt.show()


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_checkpoint(model, optimizer, path):
    """Load model checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

def calculate_metrics(pred, target):
    """Calculate SSIM and PSNR metrics"""
    # Convert tensors to numpy arrays
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    batch_size = pred_np.shape[0]
    ssim_scores = []
    psnr_scores = []
    
    for i in range(batch_size):
        # Convert from CHW to HWC format for skimage
        pred_img = np.transpose(pred_np[i], (1, 2, 0))
        target_img = np.transpose(target_np[i], (1, 2, 0))
        
        # Ensure values are in [0, 1] range
        pred_img = np.clip(pred_img, 0, 1)
        target_img = np.clip(target_img, 0, 1)
        
        # Calculate SSIM (multichannel=True for RGB images)
        ssim_score = ssim(target_img, pred_img, multichannel=True, data_range=1.0)
        ssim_scores.append(ssim_score)
        
        # Calculate PSNR
        psnr_score = psnr(target_img, pred_img, data_range=1.0)
        psnr_scores.append(psnr_score)
    
    return np.mean(ssim_scores), np.mean(psnr_scores)

# Initialize wandb
wandb.init(
    project="enhanced_vae_reconstruction",
    config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "latent_dim": LATENT_DIM,
        "learning_rate": LEARNING_RATE,
        "beta": BETA,
        "gradient_clip": GRADIENT_CLIP,
        "architecture": "EnhancedVAE",
        "image_channels": 3,
    },
)

# Training loop
best_val_loss = float('inf')

for epoch in tqdm(range(EPOCHS), desc="Training"):
    model.train()
    train_loss_total = 0
    train_recon_loss_total = 0
    train_kl_loss_total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        
        # Enhanced loss calculation
        total_loss, recon_loss, kl_loss = vae_loss(recon_batch, target, mu, logvar, BETA)
        
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        
        optimizer.step()
        
        train_loss_total += total_loss.item()
        train_recon_loss_total += recon_loss.item()
        train_kl_loss_total += kl_loss.item()

    # Calculate average losses
    avg_train_loss = train_loss_total / len(train_loader)
    avg_train_recon = train_recon_loss_total / len(train_loader)
    avg_train_kl = train_kl_loss_total / len(train_loader)

    # Validation
    model.eval()
    val_loss_total = 0
    val_recon_loss_total = 0
    val_kl_loss_total = 0
    val_ssim_total = 0
    val_psnr_total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            recon_batch, mu, logvar = model(data)
            total_loss, recon_loss, kl_loss = vae_loss(recon_batch, target, mu, logvar, BETA)
            
            val_loss_total += total_loss.item()
            val_recon_loss_total += recon_loss.item()
            val_kl_loss_total += kl_loss.item()
            
            # Calculate metrics
            ssim_score, psnr_score = calculate_metrics(recon_batch, target)
            val_ssim_total += ssim_score
            val_psnr_total += psnr_score

    avg_val_loss = val_loss_total / len(val_loader)
    avg_val_recon = val_recon_loss_total / len(val_loader)
    avg_val_kl = val_kl_loss_total / len(val_loader)
    avg_val_ssim = val_ssim_total / len(val_loader)
    avg_val_psnr = val_psnr_total / len(val_loader)
    
    # Learning rate scheduling
    scheduler.step(avg_val_loss)
    
    # Logging
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | SSIM: {avg_val_ssim:.4f} | PSNR: {avg_val_psnr:.2f} | LR: {current_lr:.2e}")
    
    wandb.log({
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "train_recon_loss": avg_train_recon,
        "train_kl_loss": avg_train_kl,
        "val_loss": avg_val_loss,
        "val_recon_loss": avg_val_recon,
        "val_kl_loss": avg_val_kl,
        "val_ssim": avg_val_ssim,
        "val_psnr": avg_val_psnr,
        "learning_rate": current_lr,
    })
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        save_checkpoint(model, optimizer, epoch, avg_val_loss, 'checkpoints/best_model.pth')
        print(f"New best model saved with validation loss: {best_val_loss:.4f}")
    
    # Save periodic checkpoint
    if epoch % 50 == 0:
        save_checkpoint(model, optimizer, epoch, avg_val_loss, f'checkpoints/checkpoint_epoch_{epoch}.pth')
    
    # Log images periodically
    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        model.eval()
        with torch.no_grad():
            # Get a batch for visualization
            data, target = next(iter(val_loader))
            data, target = data.to(device), target.to(device)
            recon_batch, _, _ = model(data)
            
            # Log images to wandb
            wandb.log({
                "reconstructed_images": [
                    wandb.Image(recon_batch[i].cpu(), caption=f"Reconstructed {i}")
                    for i in range(min(5, recon_batch.size(0)))
                ],
                "original_images": [
                    wandb.Image(target[i].cpu(), caption=f"Original {i}") 
                    for i in range(min(5, target.size(0)))
                ],
                "shuffled_images": [
                    wandb.Image(data[i].cpu(), caption=f"Shuffled {i}") 
                    for i in range(min(5, data.size(0)))
                ],
            })

print("Training completed!")
wandb.finish()
