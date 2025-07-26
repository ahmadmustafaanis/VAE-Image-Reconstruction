import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from data import ShuffledCIFAR10
from model import VAE, vae_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 200  # Reasonable number for initial training
BATCH_SIZE = 64
BETA = 1.0  # Beta-VAE weighting parameter
train_dataset = ShuffledCIFAR10(train=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = ShuffledCIFAR10(train=False)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False
)  # No need to shuffle the validation dataset


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)


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
    project="image_reconstruction_vae",
    config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "image_channels": 3,
        "CUDA_LAUNCH_BLOCKING=1": True,
    },
)

for epoch in tqdm(range(EPOCHS)):
    model.train()
    train_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = vae_loss(recon_batch, target, mu, logvar, BETA)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    avg_train_loss = train_loss / len(train_loader)
    wandb.log({"epoch": epoch, "train_loss": avg_train_loss})

    # Validation loop
    model.eval()
    val_loss = 0
    val_recon_loss = 0
    val_kl_loss = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            recon_batch, mu, logvar = model(data)
            
            # Calculate individual loss components
            recon_loss = torch.nn.functional.mse_loss(recon_batch, target, reduction="sum")
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + BETA * kl_div
            
            val_loss += loss.item()
            val_recon_loss += recon_loss.item()
            val_kl_loss += kl_div.item()
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_recon = val_recon_loss / len(val_loader)
    avg_val_kl = val_kl_loss / len(val_loader)
    
    # Learning rate scheduling
    scheduler.step(avg_val_loss)
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    wandb.log({
        "epoch": epoch, 
        "val_loss": avg_val_loss,
        "val_recon_loss": avg_val_recon,
        "val_kl_loss": avg_val_kl,
        "learning_rate": optimizer.param_groups[0]['lr']
    })

    shuffled_img, original_img, reconstructed_img = (
        data[0].cpu(),
        target[0].cpu(),
        recon_batch[0].cpu(),
    )
    # print(len(data))
    wandb.log(
        {
            "reconstructed_images": [
                wandb.Image(recon_batch[i].cpu(), caption="Reconstructed Image")
                for i in range(5)
            ],
            "original_images": [
                wandb.Image(target[i].cpu(), caption="Original Image") for i in range(5)
            ],
            "shuffled_images": [
                wandb.Image(data[i].cpu(), caption="Shuffled Image") for i in range(5)
            ],
        }
    )
