import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path

from fusion_model import FusionModel
from loaders import get_dataloaders


def mixup_data(rgb, fft, y, alpha=0.4):
    """MixUp both branches with the same permutation."""
    if alpha <= 0:
        return rgb, fft, y, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = rgb.size(0)
    index = torch.randperm(batch_size, device=rgb.device)
    mixed_rgb = lam * rgb + (1 - lam) * rgb[index, :]
    mixed_fft = lam * fft + (1 - lam) * fft[index, :]
    y_a, y_b = y, y[index]
    return mixed_rgb, mixed_fft, y_a, y_b, lam


def mixup_criterion(criterion, preds, y_a, y_b, lam):
    return lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, mixup_alpha=0.4, max_grad_norm=1.0):
    model.train()
    total_loss = 0
    correct = 0
    total_samples = len(loader.dataset)

    pbar = tqdm(loader, ncols=120)
    for rgb, fft_mag, labels in pbar:
        rgb = rgb.to(device)
        fft_mag = fft_mag.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        rgb_m, fft_m, labels_a, labels_b, lam = mixup_data(rgb, fft_mag, labels, mixup_alpha)

        with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            logits = model(rgb_m, fft_m)
            loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * rgb.size(0)
        preds = logits.argmax(dim=1)
        correct += (lam * (preds == labels_a).float() + (1 - lam) * (preds == labels_b).float()).sum().item()

        seen = max(1, pbar.n * loader.batch_size)
        pbar.set_description(f"loss={loss.item():.4f} acc={(correct / seen):.4f}")

    return total_loss / total_samples, correct / total_samples


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = len(loader.dataset)

    with torch.no_grad():
        for rgb, fft_mag, labels in loader:
            rgb = rgb.to(device)
            fft_mag = fft_mag.to(device)
            labels = labels.to(device)

            logits = model(rgb, fft_mag)
            loss = criterion(logits, labels)

            total_loss += loss.item() * rgb.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()

    return total_loss / total_samples, correct / total_samples


def main():
    repo_root = Path(__file__).resolve().parents[1]
    raw_roots = [
        str(repo_root / "rawdata" / "kaggle_a"),
        str(repo_root / "rawdata" / "kaggle_b"),

    ]

    print("Initializing Fusion Loaders (RGB + FFT views)...")
    # Using 8 workers and batch size 48 to keep the GPU fed
    train_loader, val_loader = get_dataloaders(
        raw_roots, 
        mode="fusion", 
        batch_size=48, 
        num_workers=8, 
        pin_memory=True
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FusionModel(num_classes=2, dropout=0.4).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-4)
    scaler = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_acc = 0.0
    best_path = repo_root / "best_fusion.pth"
    epochs = 12
    patience = 8
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            mixup_alpha=0.4, max_grad_norm=1.0
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")

        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
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


if __name__ == "__main__":
    main()
