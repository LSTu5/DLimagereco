from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fusion_model import FusionModel
from loaders import DynamicDataset


def evaluate(checkpoint: str = "best_fusion.pth", batch_size: int = 32):
    repo_root = Path(__file__).resolve().parents[1]
    test_roots = [str(repo_root / "rawdata" / "hf")]

    test_set = DynamicDataset(test_roots, mode="fusion", augment=False)
    if len(test_set) == 0:
        raise RuntimeError(f"No test images found in: {test_roots}")

    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FusionModel(num_classes=2).to(device)

    # Flexible checkpoint lookup: cwd, repo root, src/
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        if (repo_root / checkpoint).exists():
            ckpt_path = repo_root / checkpoint
        elif (repo_root / "src" / checkpoint).exists():
            ckpt_path = repo_root / "src" / checkpoint
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for rgb, fft_mag, labels in loader:
            rgb = rgb.to(device)
            fft_mag = fft_mag.to(device)
            labels = labels.to(device)

            logits = model(rgb, fft_mag)
            loss = criterion(logits, labels)

            total_loss += loss.item() * rgb.size(0)
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
