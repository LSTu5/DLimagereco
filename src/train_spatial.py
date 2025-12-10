import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from spatial_model import SpatialBranch

# --- IMPORT THE NEW DYNAMIC LOADER ---
from loaders import get_dataloaders

# --- HELPER FUNCTIONS ---

def mixup_data(x, y, alpha=0.4):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, preds, y_a, y_b, lam):
    return lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, mixup_alpha=0.4, max_grad_norm=1.0):
    model.train()
    total_loss = 0
    correct = 0
    total_samples = len(loader.dataset)

    pbar = tqdm(loader, ncols=120)
    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Apply MixUp
        imgs, labels_a, labels_b, lam = mixup_data(imgs, labels, mixup_alpha)

        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            logits = model(imgs)
            loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        
        # Calculate accuracy (using the stronger label)
        predicted = logits.argmax(dim=1)
        correct += (lam * (predicted == labels_a).float() + (1 - lam) * (predicted == labels_b).float()).sum().item()

        seen = max(1, pbar.n * loader.batch_size)
        pbar.set_description(
            f"loss={loss.item():.4f} acc={(correct / seen):.4f}"
        )

    return total_loss / total_samples, correct / total_samples

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = len(loader.dataset)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()

    return total_loss / total_samples, correct / total_samples

# --- MAIN TRAINING LOOP ---

def main():
    # 1. DEFINE PATHS (Relative to repo root)
    repo_root = Path(__file__).resolve().parents[1]
    
    # Point to the RAW data folders
    raw_roots = [
        str(repo_root / "rawdata" / "kaggle_a"),
        str(repo_root / "rawdata" / "kaggle_b"),
        str(repo_root / "rawdata" / "hf"),
    ]

    print("Initializing Dynamic Spatial Loaders...")
    train_loader, val_loader = get_dataloaders(raw_roots, mode='spatial', batch_size=32)
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")

    # 2. SETUP MODEL & OPTIMIZER
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SpatialBranch(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- FIXED: REMOVED verbose=True ---
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    # 3. TRAINING LOOP
    best_acc = 0.0
    best_val_loss = float('inf')
    epochs = 10
    patience = 8
    patience_counter = 0
    
    best_path = repo_root / "best_spatial.pth"

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            mixup_alpha=0.4, max_grad_norm=1.0
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")

        # Step Scheduler
        scheduler.step(val_acc)

        # Save Best Model
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

if __name__ == "__main__":
    main()