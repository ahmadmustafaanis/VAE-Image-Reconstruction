import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms


class ShuffledCIFAR10(Dataset):
    def __init__(self, train=True):
        self.dataset = torchvision.datasets.CIFAR10(
            root="./data", train=train, download=True, transform=transforms.ToTensor()
        )
        self.permutation = torch.randperm(32 * 32)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img_flat = img.view(-1, 32 * 32)
        shuffled_img_flat = img_flat[:, self.permutation]
        shuffled_img = shuffled_img_flat.view_as(img)
        return shuffled_img, img
