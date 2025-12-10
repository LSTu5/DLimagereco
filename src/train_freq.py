from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from freq_model import FrequencyBranch

class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class FrequencyImageDataset(Dataset):
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
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                # Add robustness against compression/resize artifacts
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                # Add noise after normalization? No, usually before or on tensor
                GaussianNoise(0., 0.05),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])

        # Grayscale normalize for single-scale DFT
        self.normalize = transforms.Normalize([0.5], [0.5])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("L")
        img = self.tf(img)
        img = self.normalize(img)

        return img, label


def _effective_len(loader):
    if hasattr(loader, "sampler") and loader.sampler is not None:
        try:
            return len(loader.sampler)
        except TypeError:
            return len(loader.dataset)
    return len(loader.dataset)

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, max_grad_norm=1.0):
    model.train()
    total_loss = 0
    correct = 0
    total_samples = _effective_len(loader)

    pbar = tqdm(loader, ncols=120)
    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            logits = model(imgs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()

        seen = max(1, pbar.n * loader.batch_size)
        pbar.set_description(
            f"loss={loss.item():.4f} acc={(correct/seen):.4f}"
        )
    return total_loss / total_samples, correct / total_samples

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = _effective_len(loader)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()

    return total_loss / total_samples, correct / total_samples

def main():
    # Resolve dataset paths
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "dataset"
    split_root = data_root / "dataset_split"

    if split_root.exists():
        train_roots = [str(split_root / "train")]
        val_roots = [str(split_root / "val")]

        train_set = FrequencyImageDataset(train_roots, augment=True)
        val_set = FrequencyImageDataset(val_roots, augment=False)

        if len(train_set) == 0 or len(val_set) == 0:
            raise RuntimeError("Empty train/val set in dataset_split.")

        train_indices = list(range(len(train_set)))
        val_indices = list(range(len(val_set)))

        train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)
    else:
        roots = [
            str(data_root / "kaggle_b"),
            str(data_root / "kaggle_a"),
            str(data_root / "hf"),
        ]
        full_set = FrequencyImageDataset(roots, augment=True)
        val_set = FrequencyImageDataset(roots, augment=False)

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FrequencyBranch(num_classes=2).to(device)

    # Label smoothing + higher weight decay; slightly higher LR since regularized
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    best_acc = 0.0
    best_val_loss = float('inf')
    epochs = 10
    patience = 10
    patience_counter = 0

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    best_path = repo_root / "best_frequency.pth"
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            max_grad_norm=1.0
        )

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")

        scheduler.step(val_acc)

        # Early stopping and save best
        if val_acc > best_acc:
            best_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            print(f"Saved: {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()