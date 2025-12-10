import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.fft
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class DynamicDataset(Dataset):
    def __init__(self, roots, mode='spatial', img_size=224, augment=True):
        """
        roots: list of paths to raw data (e.g. ["../rawdata/kaggle_a"])
        mode: 'spatial' (returns RGB) or 'freq' (returns FFT)
        """
        self.mode = mode
        self.img_size = img_size
        self.paths = []
        self.labels = []
        
        # 1. Gather all file paths from raw folders
        for root in roots:
            root = Path(root)
            for label, cls_name in enumerate(['real', 'fake']):
                folder = root / cls_name
                if not folder.exists(): continue
                
                for f in folder.iterdir():
                    if f.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        self.paths.append(str(f))
                        self.labels.append(label)

        # 2. Define Spatial Transforms (Augmentations)
        if augment and mode == 'spatial':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                # Normalize is critical for Spatial ResNet
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        
        try:
            img = Image.open(path).convert('RGB')
        except:
            # Handle corrupted images gracefully
            return self.__getitem__((idx + 1) % len(self))

        if self.mode == 'spatial':
            return self.transform(img), label
            
        elif self.mode == 'freq':
            # --- THE MATH HAPPENS HERE (ON THE FLY) ---
            
            # 1. Resize to ensure consistent frequency scale
            img = img.resize((self.img_size, self.img_size), Image.BICUBIC)
            
            # 2. Convert to Grayscale
            img_gray = img.convert('L')
            
            # 3. To Tensor
            img_t = transforms.ToTensor()(img_gray) # Shape: [1, H, W]
            
            # 4. Compute FFT
            fft = torch.fft.fft2(img_t)
            fft_shift = torch.fft.fftshift(fft)
            magnitude = torch.abs(fft_shift)
            
            # 5. Log Scale (so we can see the patterns)
            magnitude = torch.log(magnitude + 1e-8)
            
            # 6. Normalize (CRITICAL FIX for your "Unstable Gradients")
            # We subtract mean and divide by std to keep numbers small
            magnitude = (magnitude - magnitude.mean()) / (magnitude.std() + 1e-8)
            
            return magnitude, label

def get_dataloaders(raw_roots, mode='spatial', batch_size=32, val_split=0.2):
    """
    Automatically splits data and returns Train/Val loaders
    """
    full_dataset = DynamicDataset(roots=raw_roots, mode=mode)
    
    # Calculate split sizes
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    # Random Split
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    # Create Loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader