import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class AIDetectorDataset(Dataset):
    def __init__(self, root, img_size=224):
        self.root = root
        self.img_size = img_size
        self.classes = ["real", "fake"]

        self.paths = []
        for label, cls in enumerate(self.classes):
            folder = os.path.join(root, cls)
            if not os.path.exists(folder):
                continue
            for f in os.listdir(folder):
                self.paths.append((os.path.join(folder, f), label))

        self.spatial_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def compute_dft(self, img_tensor):
        gray = img_tensor.mean(dim=0, keepdim=True)
        fft = torch.fft.fft2(gray)
        mag = torch.abs(fft)
        mag = torch.log1p(mag)
        mag = mag / mag.max()
        return mag

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path, label = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Skip and load the next valid image
            print("Corrupted image skipped:", path)
            return self.__getitem__((idx + 1) % len(self.paths))

        x = self.spatial_tf(img)
        freq = self.compute_dft(x)
        return x, freq, label, path


def save_processed(dataset, name, output_root="../dataset"):
    out_base = Path(output_root) / name
    out_spatial = out_base / "spatial"
    out_freq = out_base / "freq"

    # Create dirs
    for cls in ["real", "fake"]:
        (out_spatial / cls).mkdir(parents=True, exist_ok=True)
        (out_freq / cls).mkdir(parents=True, exist_ok=True)

    print(f"Processing dataset {name}")

    for i in range(len(dataset)):
        try:
            img_tensor, freq, label, src_path = dataset[i]
        except Exception:
            print(f"Corrupted file skipped: {dataset.paths[i][0]}")
            continue
        cls = "real" if label == 0 else "fake"

        # Save spatial image (denormalize first)
        spatial_img = (img_tensor * 0.5 + 0.5).permute(1, 2, 0).numpy()
        spatial_img = (spatial_img * 255).clip(0, 255).astype(np.uint8)
        spatial_pil = Image.fromarray(spatial_img)

        fname = os.path.basename(src_path)
        spatial_pil.save(out_spatial / cls / fname)

        # Save FFT image
        freq_img = (freq.squeeze().numpy() * 255).astype(np.uint8)
        freq_pil = Image.fromarray(freq_img)
        freq_pil.save(out_freq / cls / fname)

    print(f"Saved processed dataset to {out_base}\n")

def show_samples(dataset, n=2):
    for i in range(n):
        img_tensor, freq, label, _ = dataset[i]

        img = (img_tensor * 0.5 + 0.5).permute(1, 2, 0).numpy()
        freq_img = freq.squeeze().numpy()

        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(img)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("FFT")
        plt.imshow(freq_img, cmap="gray")
        plt.axis("off")

        plt.show()

def main():

    dsA = AIDetectorDataset("../rawdata/kaggle_a")
    dsB = AIDetectorDataset("../rawdata/kaggle_b")
    dsHF = AIDetectorDataset("../rawdata/hf")

    print(f"A: {len(dsA)}, B: {len(dsB)}, HF: {len(dsHF)}")

    # Show samples first (optional)
    show_samples(dsA)
    show_samples(dsB)
    show_samples(dsHF)

    # Save processed datasets
    save_processed(dsA, "kaggle_a")
    save_processed(dsB, "kaggle_b")
    save_processed(dsHF, "hf")


if __name__ == "__main__":
    main()
