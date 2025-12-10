"""Data processing pipeline for dual-view (spatial + frequency) image dataset.

This module creates a PyTorch Dataset that provides two complementary views of each image:
1. Spatial view: Standard RGB image (224x224) for traditional CNN features
2. Frequency view: FFT-based grayscale magnitude spectrum (320x320) to detect AI artifacts

The frequency domain often reveals subtle patterns and artifacts that are characteristic
of AI-generated images but invisible in the spatial domain.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFile

# Allow loading of truncated/corrupted images (common in large datasets)
ImageFile.LOAD_TRUNCATED_IMAGES = True

class AIDetectorDataset(Dataset):
    """
    Dual-view dataset for AI-generated image detection.
    
    For each image, this dataset generates:
    1. Spatial view: RGB image resized to 224x224, normalized to [-1, 1]
    2. Frequency view: Grayscale FFT magnitude resized to 320x320, log-scaled and normalized
    
    The frequency domain view helps detect high-frequency artifacts and patterns
    that are characteristic of generative models (GANs, diffusion models, etc.).
    
    Args:
        root: Path to dataset root containing 'real/' and 'fake/' subdirectories
        spatial_size: Target size for spatial RGB images (default: 224)
        freq_size: Target size for frequency domain images (default: 320)
    """
    def __init__(self, root, spatial_size=224, freq_size=320):
        self.root = root
        self.spatial_size = spatial_size
        self.freq_size = freq_size
        self.classes = ["real", "fake"]  # Binary classification: 0=real, 1=fake

        # Build list of (image_path, label) tuples by scanning real/ and fake/ folders
        self.paths = []
        for label, cls in enumerate(self.classes):
            folder = os.path.join(root, cls)
            if not os.path.exists(folder):
                continue
            for f in os.listdir(folder):
                self.paths.append((os.path.join(folder, f), label))

        # Spatial transform pipeline: resize to 224x224 and normalize to [-1, 1]
        # This standardization matches common pretrained model expectations
        self.spatial_tf = transforms.Compose([
            transforms.Resize((spatial_size, spatial_size)),
            transforms.ToTensor(),  # Convert PIL to tensor [0, 1]
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # Scale to [-1, 1]
        ])

    def compute_dft(self, img_pil):
        """Compute frequency-domain representation via 2D Fourier Transform.
        
        This method extracts the frequency spectrum of an image, which can reveal
        patterns and artifacts invisible in the spatial domain. AI-generated images
        often have characteristic signatures in their frequency spectra.
        
        Steps:
        1. Resize to target frequency size using high-quality bicubic interpolation
        2. Convert to grayscale (frequency analysis works on single channel)
        3. Apply 2D Fast Fourier Transform (FFT)
        4. Take magnitude (absolute value) to get power spectrum
        5. Apply log transform to compress dynamic range (log1p = log(1+x))
        6. Normalize to [0, 1] range per image for consistent training
        
        Args:
            img_pil: PIL Image in RGB format
            
        Returns:
            mag: Tensor of shape (1, freq_size, freq_size) with normalized log-magnitude spectrum
        """
        # Resize for freq branch using bicubic interpolation (smoother than bilinear)
        img_freq = img_pil.resize((self.freq_size, self.freq_size), resample=Image.BICUBIC)
        # Convert to single-channel grayscale
        gray = transforms.functional.to_grayscale(img_freq, num_output_channels=1)
        gray_t = transforms.ToTensor()(gray)  # Shape: (1, H, W) in range [0, 1]
        
        # Compute 2D FFT (returns complex numbers representing frequency components)
        fft = torch.fft.fft2(gray_t)
        
        # Extract magnitude (power) of frequency components
        mag = torch.abs(fft)
        
        # Log transform to compress dynamic range (low frequencies dominate otherwise)
        mag = torch.log1p(mag)  # log1p(x) = log(1+x) avoids log(0)
        
        # Per-image min-max normalization to [0, 1] for consistent training
        mag_min = mag.min()
        mag_max = mag.max()
        mag = (mag - mag_min) / (mag_max - mag_min + 1e-8)  # epsilon prevents division by zero
        return mag

    def __len__(self):
        """Return total number of images in dataset."""
        return len(self.paths)

    def __getitem__(self, idx):
        """Load and process a single image, returning both spatial and frequency views.
        
        Args:
            idx: Index of the image to load
            
        Returns:
            tuple: (spatial_tensor, freq_tensor, label, path)
                - spatial_tensor: RGB image (3, 224, 224) normalized to [-1, 1]
                - freq_tensor: FFT magnitude (1, 320, 320) normalized to [0, 1]
                - label: 0 for real, 1 for fake
                - path: Original file path (useful for debugging)
        """
        path, label = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Handle corrupted images by skipping to next (common in web-scraped datasets)
            print("Corrupted image skipped:", path)
            return self.__getitem__((idx + 1) % len(self.paths))

        # Generate spatial view (standard RGB)
        x = self.spatial_tf(img)
        # Generate frequency view (FFT magnitude spectrum)
        freq = self.compute_dft(img)
        return x, freq, label, path


def save_processed(dataset, name, output_root="../dataset"):
    """Save preprocessed spatial and frequency images to disk.
    
    This function processes the entire dataset and saves both views (spatial RGB and
    frequency FFT) as PNG images. This preprocessing step:
    1. Saves computational resources during training (no need to recompute FFTs)
    2. Enables faster data loading with simple image readers
    3. Makes the processed data portable and inspectable
    
    Output structure:
        output_root/
            {name}/
                spatial/
                    real/
                    fake/
                freq/
                    real/
                    fake/
    
    Args:
        dataset: AIDetectorDataset instance to process
        name: Name for this dataset (e.g., 'kaggle_a', 'kaggle_b', 'hf')
        output_root: Base directory for saving processed data
    """
    out_base = Path(output_root) / name
    out_spatial = out_base / "spatial"
    out_freq = out_base / "freq"

    # Create directory structure for both modalities and both classes
    for cls in ["real", "fake"]:
        (out_spatial / cls).mkdir(parents=True, exist_ok=True)
        (out_freq / cls).mkdir(parents=True, exist_ok=True)

    # Process and save each image in the dataset
    for i in tqdm(range(len(dataset)), desc=f"Saving {name}", ncols=120):
        try:
            img_tensor, freq, label, src_path = dataset[i]
        except Exception:
            print(f"Corrupted file skipped: {dataset.paths[i][0]}")
            continue
        cls = "real" if label == 0 else "fake"

        # === Save spatial image ===
        # Denormalize from [-1, 1] back to [0, 1], then to [0, 255]
        spatial_img = (img_tensor * 0.5 + 0.5).permute(1, 2, 0).numpy()  # CHW -> HWC
        spatial_img = (spatial_img * 255).clip(0, 255).astype(np.uint8)
        spatial_pil = Image.fromarray(spatial_img)

        # Use original filename stem to maintain traceability
        fname = os.path.basename(src_path)
        stem = Path(fname).stem
        out_name = f"{stem}.png"
        spatial_pil.save(out_spatial / cls / out_name)

        # === Save frequency image ===
        # Convert normalized [0, 1] frequency data to [0, 255] grayscale
        freq_img = (freq.squeeze().numpy() * 255).astype(np.uint8)
        freq_pil = Image.fromarray(freq_img)
        freq_pil.save(out_freq / cls / out_name)

    print(f"Saved processed dataset to {out_base}\n")

def show_samples(dataset, n=2):
    """Visualize spatial and frequency views side-by-side for inspection.
    
    This helper function is useful for:
    - Verifying that FFT computation is working correctly
    - Understanding what frequency patterns distinguish real from fake images
    - Quality checking the preprocessing pipeline
    
    Args:
        dataset: AIDetectorDataset instance
        n: Number of samples to display
    """
    for i in range(n):
        img_tensor, freq, label, _ = dataset[i]

        # Denormalize spatial image for display
        img = (img_tensor * 0.5 + 0.5).permute(1, 2, 0).numpy()
        freq_img = freq.squeeze().numpy()

        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.title("Original (Spatial)")
        plt.imshow(img)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("FFT (Frequency Domain)")
        plt.imshow(freq_img, cmap="gray")
        plt.axis("off")

        plt.show()

def main():
    """Main entry point: load raw data, process to dual views, and save.
    
    This script should be run after data_downloader.py has populated ../rawdata/
    with the three datasets (kaggle_a, kaggle_b, hf).
    
    Output: Creates ../dataset/ with processed spatial and frequency images
            organized by source dataset.
    """

    # Load each raw dataset
    dsA = AIDetectorDataset("../rawdata/kaggle_a")
    dsB = AIDetectorDataset("../rawdata/kaggle_b")
    dsHF = AIDetectorDataset("../rawdata/hf")

    print(f"Dataset sizes - A: {len(dsA)}, B: {len(dsB)}, HF: {len(dsHF)}")

    # Uncomment to visualize samples before saving (useful for debugging)
    # show_samples(dsA)
    # show_samples(dsB)
    # show_samples(dsHF)

    # Process and save all three datasets
    save_processed(dsA, "kaggle_a")
    save_processed(dsB, "kaggle_b")
    save_processed(dsHF, "hf")


if __name__ == "__main__":
    main()
