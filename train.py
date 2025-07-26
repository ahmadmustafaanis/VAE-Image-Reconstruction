import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import wandb
from data import ShuffledCIFAR10
from model import VAE, vae_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Improved hyperparameters
EPOCHS = 100  # More reasonable number
BATCH_SIZE = 64
LATENT_DIM = 256
LEARNING_RATE = 1e-3
BETA_START = 0.0  # Start with low beta for warmup
BETA_END = 1.0    # Final beta value
WARMUP_EPOCHS = 10  # Number of epochs for beta warmup

# Data loaders
train_dataset = ShuffledCIFAR10(train=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

val_dataset = ShuffledCIFAR10(train=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# Model and optimizer
model = VAE(latent_dim=LATENT_DIM).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# Beta annealing schedule
def get_beta(epoch):
    if epoch < WARMUP_EPOCHS:
        return BETA_START + (BETA_END - BETA_START) * (epoch / WARMUP_EPOCHS)
    return BETA_END


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


wandb.init(
    project="image_reconstruction_vae_improved",
    config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "latent_dim": LATENT_DIM,
        "learning_rate": LEARNING_RATE,
        "beta_start": BETA_START,
        "beta_end": BETA_END,
        "warmup_epochs": WARMUP_EPOCHS,
        "image_channels": 3,
    },
)

best_val_loss = float('inf')
patience_counter = 0
early_stop_patience = 20

for epoch in tqdm(range(EPOCHS), desc="Training"):
    # Get beta for current epoch
    beta = get_beta(epoch)
    
    # Training phase
    model.train()
    train_loss = 0
    train_recon_loss = 0
    train_kl_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        
        # Updated loss function call
        total_loss, recon_loss, kl_loss = vae_loss(recon_batch, target, mu, logvar, beta)
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        train_loss += total_loss.item()
        train_recon_loss += recon_loss.item()
        train_kl_loss += kl_loss.item()
        optimizer.step()

    # Calculate average training losses
    avg_train_loss = train_loss / len(train_loader)
    avg_train_recon = train_recon_loss / len(train_loader)
    avg_train_kl = train_kl_loss / len(train_loader)

    # Validation phase
    model.eval()
    val_loss = 0
    val_recon_loss = 0
    val_kl_loss = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            recon_batch, mu, logvar = model(data)
            total_loss, recon_loss, kl_loss = vae_loss(recon_batch, target, mu, logvar, beta)
            val_loss += total_loss.item()
            val_recon_loss += recon_loss.item()
            val_kl_loss += kl_loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_recon = val_recon_loss / len(val_loader)
    avg_val_kl = val_kl_loss / len(val_loader)
    
    # Learning rate scheduling
    scheduler.step(avg_val_loss)
    
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), 'best_vae_model.pth')
    else:
        patience_counter += 1
    
    # Print progress
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"  Beta: {beta:.4f}")
    print(f"  Train Loss: {avg_train_loss:.4f} (Recon: {avg_train_recon:.4f}, KL: {avg_train_kl:.4f})")
    print(f"  Val Loss: {avg_val_loss:.4f} (Recon: {avg_val_recon:.4f}, KL: {avg_val_kl:.4f})")
    print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Log to wandb
    wandb.log({
        "epoch": epoch,
        "beta": beta,
        "train_loss": avg_train_loss,
        "train_recon_loss": avg_train_recon,
        "train_kl_loss": avg_train_kl,
        "val_loss": avg_val_loss,
        "val_recon_loss": avg_val_recon,
        "val_kl_loss": avg_val_kl,
        "learning_rate": optimizer.param_groups[0]['lr'],
    })
    
    # Log images every 10 epochs
    if epoch % 10 == 0:
        with torch.no_grad():
            # Get a batch for visualization
            sample_data, sample_target = next(iter(val_loader))
            sample_data, sample_target = sample_data[:5].to(device), sample_target[:5].to(device)
            sample_recon, _, _ = model(sample_data)
            
            # Denormalize for visualization
            sample_data_vis = train_dataset.denormalize(sample_data.cpu())
            sample_target_vis = train_dataset.denormalize(sample_target.cpu())
            sample_recon_vis = train_dataset.denormalize(sample_recon.cpu())
            
            # Clamp to [0, 1] for display
            sample_data_vis = torch.clamp(sample_data_vis, 0, 1)
            sample_target_vis = torch.clamp(sample_target_vis, 0, 1)
            sample_recon_vis = torch.clamp(sample_recon_vis, 0, 1)
            
            wandb.log({
                "reconstructed_images": [
                    wandb.Image(sample_recon_vis[i], caption=f"Reconstructed {i}")
                    for i in range(5)
                ],
                "original_images": [
                    wandb.Image(sample_target_vis[i], caption=f"Original {i}")
                    for i in range(5)
                ],
                "shuffled_images": [
                    wandb.Image(sample_data_vis[i], caption=f"Shuffled {i}")
                    for i in range(5)
                ],
            })
    
    # Early stopping check
    if patience_counter >= early_stop_patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

print("Training completed!")
wandb.finish()
