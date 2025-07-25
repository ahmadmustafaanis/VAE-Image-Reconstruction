import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
import os

import wandb
from data import ShuffledCIFAR10
from model import VAE, beta_vae_loss, vae_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Improved hyperparameters
EPOCHS = 1000  # More reasonable number for better convergence
BATCH_SIZE = 32  # Smaller batch size for better gradient estimates
LEARNING_RATE = 1e-4  # Lower learning rate for stability
LATENT_DIM = 256  # Larger latent space for better feature representation
CLIP_GRAD = 1.0  # Gradient clipping

# Data loaders
train_dataset = ShuffledCIFAR10(train=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

val_dataset = ShuffledCIFAR10(train=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Enhanced model with larger latent dimension
model = VAE(image_channels=3, latent_dim=LATENT_DIM).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# Improved optimizer with weight decay
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

# Learning rate scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# Create checkpoints directory
os.makedirs('checkpoints', exist_ok=True)

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

def kl_annealing_schedule(epoch, total_epochs, annealing_epochs=200):
    """KL annealing schedule: gradually increase KL weight from 0 to 1"""
    if epoch < annealing_epochs:
        return epoch / annealing_epochs
    return 1.0

def save_checkpoint(model, optimizer, scheduler, epoch, loss, filename):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, filename)

def validate_model(model, val_loader, device, kl_weight=1.0):
    """Validate the model on validation set"""
    model.eval()
    val_loss = 0
    val_recon_loss = 0
    val_kl_loss = 0
    val_perceptual_loss = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            recon_batch, mu, logvar = model(data)
            
            total_loss, recon_loss, kl_loss, perceptual_loss = model.compute_loss(
                recon_batch, target, mu, logvar, use_perceptual=True
            )
            
            val_loss += total_loss.item()
            val_recon_loss += recon_loss.item()
            val_kl_loss += kl_loss.item()
            val_perceptual_loss += perceptual_loss.item() if isinstance(perceptual_loss, torch.Tensor) else perceptual_loss
    
    num_batches = len(val_loader)
    return {
        'val_loss': val_loss / num_batches,
        'val_recon_loss': val_recon_loss / num_batches,
        'val_kl_loss': val_kl_loss / num_batches,
        'val_perceptual_loss': val_perceptual_loss / num_batches
    }

wandb.init(
    project="enhanced_vae_reconstruction",
    config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "latent_dim": LATENT_DIM,
        "image_channels": 3,
        "clip_grad": CLIP_GRAD,
        "architecture": "Enhanced VAE with Attention",
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
    },
)

# Track model architecture
wandb.watch(model, log="all", log_freq=100)

best_val_loss = float('inf')

for epoch in tqdm(range(EPOCHS), desc="Training"):
    model.train()
    train_loss = 0
    train_recon_loss = 0
    train_kl_loss = 0
    train_perceptual_loss = 0
    
    # KL annealing
    kl_weight = kl_annealing_schedule(epoch, EPOCHS)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        
        # Use the model's enhanced loss function
        total_loss, recon_loss, kl_loss, perceptual_loss = model.compute_loss(
            recon_batch, target, mu, logvar, use_perceptual=True
        )
        
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
        
        optimizer.step()
        
        # Accumulate losses
        train_loss += total_loss.item()
        train_recon_loss += recon_loss.item()
        train_kl_loss += kl_loss.item()
        train_perceptual_loss += perceptual_loss.item() if isinstance(perceptual_loss, torch.Tensor) else perceptual_loss
    
    # Update learning rate
    scheduler.step()
    
    # Calculate average losses
    num_batches = len(train_loader)
    avg_train_loss = train_loss / num_batches
    avg_train_recon = train_recon_loss / num_batches
    avg_train_kl = train_kl_loss / num_batches
    avg_train_perceptual = train_perceptual_loss / num_batches
    
    # Validation every 10 epochs
    if epoch % 10 == 0:
        val_metrics = validate_model(model, val_loader, device, kl_weight)
        
        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, 
                          f'checkpoints/best_model_epoch_{epoch}.pth')
        
        print(f"Epoch {epoch:4d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f} | "
              f"Beta: {model.beta.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_recon_loss": avg_train_recon,
            "train_kl_loss": avg_train_kl,
            "train_perceptual_loss": avg_train_perceptual,
            "val_loss": val_metrics['val_loss'],
            "val_recon_loss": val_metrics['val_recon_loss'],
            "val_kl_loss": val_metrics['val_kl_loss'],
            "val_perceptual_loss": val_metrics['val_perceptual_loss'],
            "beta": model.beta.item(),
            "kl_weight": kl_weight,
            "learning_rate": scheduler.get_last_lr()[0]
        })
        
        # Log sample images
        model.eval()
        with torch.no_grad():
            sample_data, sample_target = next(iter(val_loader))
            sample_data, sample_target = sample_data[:5].to(device), sample_target[:5].to(device)
            sample_recon, _, _ = model(sample_data)
            
            wandb.log({
                "reconstructed_images": [
                    wandb.Image(sample_recon[i].cpu(), caption=f"Reconstructed {i}")
                    for i in range(5)
                ],
                "original_images": [
                    wandb.Image(sample_target[i].cpu(), caption=f"Original {i}") 
                    for i in range(5)
                ],
                "shuffled_images": [
                    wandb.Image(sample_data[i].cpu(), caption=f"Shuffled {i}") 
                    for i in range(5)
                ],
            })
    
    # Save checkpoint every 100 epochs
    if epoch % 100 == 0 and epoch > 0:
        save_checkpoint(model, optimizer, scheduler, epoch, avg_train_loss, 
                      f'checkpoints/checkpoint_epoch_{epoch}.pth')

# Final model save
save_checkpoint(model, optimizer, scheduler, EPOCHS-1, avg_train_loss, 
              'checkpoints/final_model.pth')

print("Training completed!")
wandb.finish()