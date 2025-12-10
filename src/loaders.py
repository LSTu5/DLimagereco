import os
from pathlib import Path
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class DynamicDataset(Dataset):
    """
    On-the-fly dataset that can emit:
    - spatial (RGB, augmented/normalized)
    - freq    (log-magnitude FFT of grayscale view)
    - fusion  (both views, aligned)
    """

    def __init__(self, roots: List[str], mode: str = "spatial", img_size: int = 224, augment: bool = True):
        self.mode = mode
        self.img_size = img_size
        self.paths: List[str] = []
        self.labels: List[int] = []

        # Collect image paths
        for root in roots:
            root_path = Path(root)
            for label, cls_name in enumerate(["real", "fake"]):
                cls_dir = root_path / cls_name
                if not cls_dir.exists():
                    continue
                for f in cls_dir.iterdir():
                    if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        self.paths.append(str(f))
                        self.labels.append(label)

        # Spatial transforms
        if augment and mode in ["spatial", "fusion"]:
            self.spatial_transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.spatial_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        # Frequency view uses deterministic resize to keep FFT scale consistent
        self.freq_resize = transforms.Resize((img_size, img_size))

    def __len__(self) -> int:
        return len(self.paths)

    def _load_freq_view(self, img: Image.Image) -> torch.Tensor:
        """Create log-FFT magnitude from a PIL RGB image."""
        img_resized = self.freq_resize(img)
        img_gray = img_resized.convert("L")
        img_t = transforms.ToTensor()(img_gray)  # [1, H, W]
        fft = torch.fft.fft2(img_t)
        fft_shift = torch.fft.fftshift(fft)
        magnitude = torch.abs(fft_shift)
        magnitude = torch.log(magnitude + 1e-8)
        magnitude = (magnitude - magnitude.mean()) / (magnitude.std() + 1e-8)
        return magnitude

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        label = self.labels[idx]

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Skip corrupted images by sampling the next one
            return self.__getitem__((idx + 1) % len(self))

        if self.mode == "spatial":
            return self.spatial_transform(img), label

        freq_view = self._load_freq_view(img)
        if self.mode == "freq":
            return freq_view, label

        if self.mode == "fusion":
            rgb = self.spatial_transform(img)
            return rgb, freq_view, label

        raise ValueError(f"Unknown mode: {self.mode}")


def get_dataloaders(
    raw_roots: List[str],
    mode: str = "spatial",
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 0,
):
    """
    Create train/val loaders from raw roots. Uses on-the-fly transforms/FFT.
    """
    dataset = DynamicDataset(roots=raw_roots, mode=mode)
    total = len(dataset)
    if total == 0:
        raise RuntimeError(f"No images found under roots: {raw_roots}")
    val_size = int(total * val_split)
    train_size = total - val_size
    if train_size == 0:
        train_size, val_size = total, 0
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
