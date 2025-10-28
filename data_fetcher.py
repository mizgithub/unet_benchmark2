import kagglehub
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

class ThyroidDataset(Dataset):
    def __init__(self, p_image, p_mask, transform=None):
        self.p_image = p_image
        self.p_mask = p_mask
        self.transform = transform

        self.image_files = sorted([f for f in os.listdir(p_image) if f.endswith(('.png', '.jpg', '.jpeg','.PNG'))])
        self.mask_files = sorted([f for f in os.listdir(p_mask) if f.endswith(('.png', '.jpg', '.jpeg','.PNG'))])

        assert len(self.image_files) == len(self.mask_files), "Image and mask counts do not match!"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.p_image, self.image_files[idx])
        mask_path = os.path.join(self.p_mask, self.mask_files[idx])

        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# ===== Dataset Initialization and Split =====

def load_thyroid_datasets(batch_size=8, seed=42):
    # path = kagglehub.dataset_download("eiraoi/thyroidultrasound")
    # print("Dataset downloaded. arranging ... ")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # dataset = ThyroidDataset(path+"/p_image", path+"/p_mask", transform=transform)
    dataset = ThyroidDataset("../trimmed_data/image", "../trimmed_data/mask", transform=transform)

    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    torch.manual_seed(seed)
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader