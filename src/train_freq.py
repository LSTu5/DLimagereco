"""Training script for the frequency branch model.

This script trains a CNN on FFT magnitude spectra to classify real vs AI-generated
images. The frequency domain can reveal artifacts invisible in spatial domain:
- GAN upsampling artifacts appear as periodic patterns
- Compression artifacts have characteristic frequency signatures
- Diffusion models may have distinct spectral biases

Key differences from spatial training:
- Simpler augmentation (frequency data is more fragile)
- Single-channel grayscale images (320x320)
- No mixup (frequency mixing can destroy meaningful patterns)
"""

from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from freq_model import FrequencyBranch

class FrequencyImageDataset(Dataset):
    """Dataset loader for preprocessed frequency (FFT) images.
    
    This dataset loads grayscale FFT magnitude spectra that were precomputed
    by data_processing.py. Augmentation is minimal compared to spatial training
    because frequency-domain data is more sensitive to distortions.
    
    Augmentation strategy:
    - RandomResizedCrop: Limited scale variation (80-100%)
    - RandomHorizontalFlip: FFT magnitude is symmetric, so flip is safe
    - No color jittering (grayscale data)
    - No perspective or erasing (would corrupt frequency patterns)
    
    Args:
        roots: List of root directories containing real/ and fake/ folders
        img_size: Target image size (default: 224, but model uses 320)
        augment: Whether to apply data augmentation (default: True for training)
    """
    def __init__(self, roots, img_size=224, augment=True):
        self.paths = []
        self.labels = []

        # Collect all FFT image paths and labels
        for root in roots:
            for cls, label in [("real", 0), ("fake", 1)]:
                folder = Path(root) / cls
                if not folder.exists():
                    continue
                for f in folder.iterdir():
                    if f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                        self.paths.append(str(f))
                        self.labels.append(label)

        if augment:
            # Gentle augmentation for frequency data
            self.tf = transforms.Compose([
                # Less aggressive scale variation than spatial (80-100% vs 60-100%)
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                # Horizontal flip is safe (FFT magnitude is symmetric)
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            # No augmentation for validation
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])

        # Normalize to [-1, 1] for single-channel grayscale
        self.normalize = transforms.Normalize([0.5], [0.5])

    def __len__(self):
        """Return total number of FFT images in dataset."""
        return len(self.paths)

    def __getitem__(self, idx):
        """Load and transform a single FFT image.
        
        Args:
            idx: Index of image to load
            
        Returns:
            tuple: (image_tensor, label)
                - image_tensor: (1, img_size, img_size) grayscale tensor normalized to [-1, 1]
                - label: 0 for real, 1 for fake
        """
        img_path = self.paths[idx]
        label = self.labels[idx]

        # Load grayscale FFT image (single channel)
        img = Image.open(img_path).convert("L")
        img = self.normalize(self.tf(img))

        return img, label

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """Train model for one epoch on frequency data.
    
    Simpler than spatial training:
    - No mixup (frequency mixing destroys meaningful patterns)
    - Standard AMP for efficiency
    - Basic gradient descent with AdamW
    
    Args:
        model: FrequencyBranch neural network
        loader: DataLoader providing training batches
        criterion: Loss function (CrossEntropyLoss)
        optimizer: AdamW optimizer
        scaler: GradScaler for automatic mixed precision
        device: Device to run on ('cuda' or 'cpu')
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    total_loss = 0
    correct = 0

    pbar = tqdm(loader, ncols=120)
    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass with automatic mixed precision
        with torch.cuda.amp.autocast():
            logits = model(imgs)
            loss = criterion(logits, labels)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Track metrics
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()

        # Update progress bar
        seen = max(1, pbar.n * loader.batch_size)
        pbar.set_description(
            f"loss={loss.item():.4f} acc={(correct/seen):.4f}"
        )
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def validate(model, loader, criterion, device):
    """Evaluate model on validation set.
    
    Args:
        model: FrequencyBranch neural network
        loader: DataLoader providing validation batches
        criterion: Loss function
        device: Device to run on
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def main():
    """Main training loop for frequency branch model.
    
    This script uses a different train/val split than spatial training:
    - Training: kaggle_a + kaggle_b frequency data
    - Validation: hf (Hugging Face) frequency data
    
    This split helps test generalization across different dataset sources.
    """
    # Define data paths for frequency domain training
    train_roots = [
        "../dataset/kaggle_b/freq",
        "../dataset/kaggle_a/freq"
    ]

    # Use HF dataset for validation (different source for better generalization test)
    val_roots = ["../dataset/hf/freq"]

    # Create datasets with appropriate augmentation
    train_set = FrequencyImageDataset(train_roots, augment=True)
    val_set = FrequencyImageDataset(val_roots, augment=False)

    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)

    print("Train samples:", len(train_set))
    print("Val samples:", len(val_set))

    # === Setup training configuration ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FrequencyBranch(num_classes=2).to(device)

    # Simpler training setup than spatial (no label smoothing, no scheduler)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Gradient scaler for automatic mixed precision
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    best_acc = 0.0
    epochs = 10

    # === Training loop ===
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # Train and validate
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")

        # Save best model (no early stopping in this script)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_frequency.pth")
            print("Saved: best_frequency.pth")

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()