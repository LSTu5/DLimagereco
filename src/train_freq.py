import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from freq_model import FrequencyBranch

# --- IMPORT THE NEW DYNAMIC LOADER ---
# This handles the FFT calculation and Normalization automatically
from loaders import get_dataloaders

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, max_grad_norm=1.0):
    model.train()
    total_loss = 0
    correct = 0
    total_samples = len(loader.dataset)

    pbar = tqdm(loader, ncols=120)
    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Mixed Precision Training (Speed + Stability)
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
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
    total_samples = len(loader.dataset)

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
    # 1. SETUP PATHS
    repo_root = Path(__file__).resolve().parents[1]
    
    # Point directly to the RAW data (bypassing the 'dataset' folder)
    raw_roots = [
        str(repo_root / "rawdata" / "kaggle_a"),
        str(repo_root / "rawdata" / "kaggle_b"),
        str(repo_root / "rawdata" / "hf"),
    ]

    print("Initializing Dynamic Frequency Loaders (Calculating FFT on the fly)...")
    
    # mode='freq' tells the loader to run the FFT math and Normalization
    train_loader, val_loader = get_dataloaders(raw_roots, mode='freq', batch_size=32)
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")

    # 2. SETUP MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize model
    # Note: FrequencyBranch expects 1-channel input, which our loader provides
    model = FrequencyBranch(num_classes=2).to(device)

    # 3. SETUP OPTIMIZER
    # Label smoothing helps with noisy data
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # AdamW with decent weight decay to prevent overfitting
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    # 4. TRAINING LOOP
    best_acc = 0.0
    best_val_loss = float('inf')
    epochs = 15  # Frequency needs a bit more time to converge
    patience = 8
    patience_counter = 0

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
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

        # Save Best
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