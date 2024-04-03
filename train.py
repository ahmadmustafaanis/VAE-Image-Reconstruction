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
EPOCHS = 100000  # can we have grokking kinda effect with this insane number???
BATCH_SIZE = 64
train_dataset = ShuffledCIFAR10(train=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = ShuffledCIFAR10(train=False)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False
)  # No need to shuffle the validation dataset


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)


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
        loss = vae_loss(recon_batch, target, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch}, Training Loss: {avg_train_loss}")
    wandb.log({"epoch": epoch, "train_loss": avg_train_loss})

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            recon_batch, _, _ = model(data)

            loss = vae_loss(recon_batch, target, mu, logvar)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    wandb.log({"epoch": epoch, "val_loss": avg_val_loss})

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
