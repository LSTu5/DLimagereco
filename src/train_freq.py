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
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])

        self.normalize = transforms.Normalize([0.5], [0.5])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("L")       # single path
        img = self.normalize(self.tf(img))

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

        with torch.cuda.amp.autocast():
            logits = model(imgs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()

        seen = max(1, pbar.n * loader.batch_size)
        pbar.set_description(
            f"loss={loss.item():.4f} acc={(correct/seen):.4f}"
        )
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def validate(model, loader, criterion, device):
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
    train_roots = [
        "../dataset/kaggle_b/freq",
        "../dataset/kaggle_a/freq"
    ]

    val_roots = ["../dataset/hf/freq"]

    train_set = FrequencyImageDataset(train_roots, augment=True)
    val_set = FrequencyImageDataset(val_roots, augment=False)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)

    print("Train samples:", len(train_set))
    print("Val samples:", len(val_set))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FrequencyBranch(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    best_acc = 0.0
    epochs = 10

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_frequency.pth")
            print("Saved: best_frequency.pth")

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()