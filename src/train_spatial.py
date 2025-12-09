from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from spatial_model import SpatialBranch  # Ensure this file contains the code above


class SpatialImageDataset(Dataset):
    def __init__(self, roots, img_size=224, augment=True):
        self.paths = []
        self.labels = []

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
            # ADDED STRONGER AUGMENTATIONS: RandomAffine and RandomErasing
            self.tf = transforms.Compose([
                transforms.RandomAffine(
                    degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
                ),
                transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1
                ),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])

        # NOTE: Using your original normalization, but ImageNet stats
        # (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) are often better.
        self.normalize = transforms.Normalize([0.5] * 3, [0.5] * 3)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.tf(img)
        img = self.normalize(img)

        return img, label


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    correct = 0

    pbar = tqdm(loader, ncols=120)
    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            logits = model(imgs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()

        seen = max(1, pbar.n * loader.batch_size)
        pbar.set_description(
            f"loss={loss.item():.4f} acc={(correct / seen):.4f}"
        )

    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def main():
    train_roots = [
        "../dataset/kaggle_b/spatial",
        "../dataset/kaggle_a/spatial",
    ]

    val_roots = [
        "../dataset/hf/spatial",
    ]

    train_set = SpatialImageDataset(train_roots, augment=True)
    val_set = SpatialImageDataset(val_roots, augment=False)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)

    print("Train samples:", len(train_set))
    print("Val samples:", len(val_set))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SpatialBranch(num_classes=2).to(device)

    # ADDED Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # Increased initial LR for training from scratch
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    best_acc = 0.0
    epochs = 50
    # ADDED Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")

        # STEP the scheduler
        scheduler.step()

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_spatial.pth")
            print("Saved: best_spatial.pth")

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()