from pathlib import Path
import torch
from torch.utils.data import DataLoader
from train_freq import FrequencyImageDataset
from freq_model import FrequencyBranch


def evaluate(checkpoint="best_frequency.pth", batch_size=32):
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "dataset"

    test_roots = [str(data_root / "hf")]
    test_set = FrequencyImageDataset(test_roots, augment=False)
    if len(test_set) == 0:
        raise RuntimeError(f"No test images found in: {test_roots}")

    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FrequencyBranch(num_classes=2).to(device)
    if not Path(checkpoint).exists():
        # try src/..
        alt_ckpt = repo_root / checkpoint
        if alt_ckpt.exists():
            checkpoint = str(alt_ckpt)
        else:
             # try src/
            alt_ckpt = repo_root / "src" / checkpoint
            if alt_ckpt.exists():
                checkpoint = str(alt_ckpt)

    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    print(f"Test samples: {total}")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    evaluate()


