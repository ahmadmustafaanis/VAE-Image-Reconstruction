import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms


class ShuffledCIFAR10(Dataset):
    def __init__(self, train=True):
        # Proper CIFAR-10 normalization
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.dataset = torchvision.datasets.CIFAR10(
            root="./data", train=train, download=True, transform=transform
        )
        self.permutation = torch.randperm(32 * 32)
        
        # Store normalization params for denormalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img_flat = img.view(-1, 32 * 32)
        shuffled_img_flat = img_flat[:, self.permutation]
        shuffled_img = shuffled_img_flat.view_as(img)
        return shuffled_img, img
    
    def denormalize(self, tensor):
        """Denormalize tensor back to [0, 1] range"""
        return tensor * self.std + self.mean
