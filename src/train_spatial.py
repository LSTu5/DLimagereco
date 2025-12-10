"""Training script for the spatial branch model.

This script trains a CNN on RGB images to classify real vs AI-generated images.
Key features:
- Strong data augmentation (RandAugment, color jitter, perspective, erasing)
- Mixup regularization for better generalization
- Automatic mixed precision (AMP) for faster training
- Learning rate scheduling and early stopping
- Support for both pre-split and in-memory splitting
"""

from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from spatial_model import SpatialBranch


class SpatialImageDataset(Dataset):
    """Dataset loader for preprocessed spatial (RGB) images.
    
    This dataset loads images from disk that were already processed by
    data_processing.py. It applies augmentation during training to improve
    model robustness and generalization.
    
    Augmentation strategy (when augment=True):
    - RandomResizedCrop: Simulates different scales and perspectives
    - RandAugment: Automated augmentation policy from Google Research
    - ColorJitter: Simulates lighting variations
    - RandomPerspective: Simulates camera angles
    - RandomErasing: Simulates occlusions (Cutout-like)
    
    Args:
        roots: List of root directories containing real/ and fake/ folders
        img_size: Target image size (default: 224)
        augment: Whether to apply data augmentation (default: True for training)
    """
    def __init__(self, roots, img_size=224, augment=True):
        self.paths = []
        self.labels = []

        # Collect all image paths and labels from specified roots
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
            # Strong augmentation pipeline for training
            self.tf = transforms.Compose([
                # Random crop with scale variation (60-100% of original)
                transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
                # RandAugment: learnable augmentation policy
                transforms.RandAugment(num_ops=2, magnitude=9),
                # Horizontal flip (vertical not used - most images have natural orientation)
                transforms.RandomHorizontalFlip(),
                # Color variations to handle different lighting/cameras
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
                ),
                # Perspective distortion to simulate camera angles
                transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
                transforms.ToTensor(),
                # Random erasing (Cutout) to improve robustness to occlusions
                transforms.RandomErasing(p=0.6, scale=(0.02, 0.2), ratio=(0.3, 3.3))
            ])
        else:
            # Simple pipeline for validation/testing (no augmentation)
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])

        # Normalize to [-1, 1] range (applied after augmentation)
        self.normalize = transforms.Normalize([0.5] * 3, [0.5] * 3)

    def __len__(self):
        """Return total number of images in dataset."""
        return len(self.paths)

    def __getitem__(self, idx):
        """Load and transform a single image.
        
        Args:
            idx: Index of image to load
            
        Returns:
            tuple: (image_tensor, label)
                - image_tensor: (3, 224, 224) RGB tensor normalized to [-1, 1]
                - label: 0 for real, 1 for fake
        """
        img_path = self.paths[idx]
        label = self.labels[idx]

        # Load image and apply augmentation
        img = Image.open(img_path).convert("RGB")
        img = self.tf(img)
        img = self.normalize(img)

        return img, label


def mixup_data(x, y, alpha=0.4):
    """Apply mixup data augmentation by blending two samples.
    
    Mixup creates synthetic training examples by linearly interpolating between
    pairs of images and their labels. This technique:
    - Improves generalization by encouraging smooth decision boundaries
    - Acts as a strong regularizer to prevent overfitting
    - Increases robustness to label noise
    
    Reference: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2018)
    
    Args:
        x: Input batch of images (batch_size, 3, H, W)
        y: Labels (batch_size,)
        alpha: Beta distribution parameter (default: 0.4)
               Higher alpha = more mixing, lower = less mixing
    
    Returns:
        tuple: (mixed_x, y_a, y_b, lam)
            - mixed_x: Interpolated images
            - y_a, y_b: Original labels for loss computation
            - lam: Mixing coefficient [0, 1]
    """
    if alpha <= 0:
        return x, y, y, 1.0
    
    # Sample mixing coefficient from symmetric Beta distribution
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = x.size(0)
    
    # Randomly permute batch to create pairs
    index = torch.randperm(batch_size, device=x.device)
    
    # Linear interpolation: mixed = λ * x + (1-λ) * x_permuted
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, preds, y_a, y_b, lam):
    """Compute loss for mixup training.
    
    The loss is a weighted combination of losses for both original labels:
    L = λ * L(pred, y_a) + (1-λ) * L(pred, y_b)
    
    Args:
        criterion: Loss function (e.g., CrossEntropyLoss)
        preds: Model predictions
        y_a, y_b: Original labels from both samples
        lam: Mixing coefficient
        
    Returns:
        Weighted loss value
    """
    return lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)


def _effective_len(loader):
    """Get effective dataset size accounting for samplers.
    
    This helper handles different loader configurations:
    - With SubsetRandomSampler: returns subset size
    - Without sampler: returns full dataset size
    """
    if hasattr(loader, "sampler") and loader.sampler is not None:
        try:
            return len(loader.sampler)  # SubsetRandomSampler supports __len__
        except TypeError:
            return len(loader.dataset)
    return len(loader.dataset)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, mixup_alpha=0.4, max_grad_norm=1.0):
    """Train model for one epoch with mixup and AMP.
    
    This function implements a complete training loop with:
    - Mixup data augmentation for regularization
    - Automatic mixed precision (AMP) for faster training
    - Gradient clipping to prevent exploding gradients
    - Real-time progress tracking with tqdm
    
    Args:
        model: Neural network to train
        loader: DataLoader providing training batches
        criterion: Loss function (e.g., CrossEntropyLoss)
        optimizer: Optimizer (e.g., AdamW)
        scaler: GradScaler for automatic mixed precision
        device: Device to run on ('cuda' or 'cpu')
        mixup_alpha: Mixup parameter (default: 0.4)
        max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()  # Enable training mode (dropout, batch norm training)
    total_loss = 0
    correct = 0
    total_samples = _effective_len(loader)

    pbar = tqdm(loader, ncols=120)
    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)

        # Clear gradients from previous step
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

        # Apply mixup augmentation
        imgs, labels_a, labels_b, lam = mixup_data(imgs, labels, mixup_alpha)

        # Forward pass with automatic mixed precision (FP16/FP32)
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            logits = model(imgs)
            loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)

        # Backward pass with gradient scaling for AMP
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # Unscale before gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        # Track metrics
        total_loss += loss.item() * imgs.size(0)
        # Approximate accuracy using original (unmixed) labels
        correct += (logits.argmax(dim=1) == labels).sum().item()

        # Update progress bar
        seen = max(1, pbar.n * loader.batch_size)
        pbar.set_description(
            f"loss={loss.item():.4f} acc={(correct / seen):.4f}"
        )

    return total_loss / total_samples, correct / total_samples


def validate(model, loader, criterion, device):
    """Evaluate model on validation set.
    
    Runs the model in evaluation mode (no dropout, batch norm uses running stats)
    without gradient computation for efficiency.
    
    Args:
        model: Neural network to evaluate
        loader: DataLoader providing validation batches
        criterion: Loss function
        device: Device to run on ('cuda' or 'cpu')
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()  # Disable training-specific layers (dropout, etc.)
    total_loss = 0
    correct = 0
    total_samples = _effective_len(loader)

    # Disable gradient computation for efficiency
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()

    return total_loss / total_samples, correct / total_samples


def main():
    """Main training loop for spatial branch model.
    
    This function:
    1. Loads data (either pre-split or creates split on-the-fly)
    2. Initializes model, optimizer, and training utilities
    3. Trains for specified epochs with validation
    4. Implements early stopping based on validation accuracy
    5. Saves best model checkpoint
    """
    # Resolve dataset paths relative to repo root
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "dataset"
    split_root = data_root / "dataset_split"

    # Prefer split dataset if it exists; else use kaggle_a + kaggle_b and split in-memory
    if split_root.exists():
        train_roots = [str(split_root / "train" / "spatial")]
        val_roots = [str(split_root / "val" / "spatial")]

        train_set = SpatialImageDataset(train_roots, augment=True)
        val_set = SpatialImageDataset(val_roots, augment=False)

        if len(train_set) == 0:
            raise RuntimeError(f"No training images found in: {train_roots}")
        if len(val_set) == 0:
            raise RuntimeError(f"No validation images found in: {val_roots}")

        train_indices = list(range(len(train_set)))
        val_indices = list(range(len(val_set)))

        train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)
    else:
        roots = [
            str(data_root / "kaggle_b" / "spatial"),
            str(data_root / "kaggle_a" / "spatial"),
        ]
        full_set = SpatialImageDataset(roots, augment=True)
        val_set = SpatialImageDataset(roots, augment=False)

        if len(full_set) < 2:
            raise RuntimeError(f"Not enough images found in: {roots}")

        val_ratio = 0.1
        val_size = max(1, int(len(full_set) * val_ratio))
        indices = torch.randperm(len(full_set), generator=torch.Generator().manual_seed(42)).tolist()
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        train_loader = DataLoader(
            full_set,
            batch_size=32,
            sampler=SubsetRandomSampler(train_indices),
            num_workers=4,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=32,
            sampler=SubsetRandomSampler(val_indices),
            num_workers=4,
        )
        # for reporting
        train_set = full_set

    train_count = len(train_indices) if 'train_indices' in locals() else len(train_set)
    val_count = len(val_indices) if 'val_indices' in locals() else len(val_set)
    print("Train samples:", train_count)
    print("Val samples:", val_count)

    # === Setup training configuration ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SpatialBranch(num_classes=2).to(device)

    # Loss with label smoothing to prevent overconfidence
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    # AdamW optimizer with weight decay for regularization
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=3e-4, weight_decay=5e-4)

    # Gradient scaler for automatic mixed precision training
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    # Training state tracking
    best_acc = 0.0
    best_val_loss = float('inf')
    epochs = 10
    patience = 10  # Early stopping patience
    patience_counter = 0
    
    # Learning rate scheduler: reduce LR when validation accuracy plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # === Main training loop ===
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            mixup_alpha=0.4, max_grad_norm=1.0
        )

        # Evaluate on validation set
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")

        # Adjust learning rate based on validation performance
        scheduler.step(val_acc)

        # Save best model and implement early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_spatial.pth")
            print("Saved: best_spatial.pth")
        else:
            # No improvement: increment patience counter
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()