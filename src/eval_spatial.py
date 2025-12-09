from pathlib import Path
import torch
from torch.utils.data import DataLoader
from train_spatial import SpatialImageDataset
from spatial_model import SpatialBranch


def evaluate(checkpoint="best_spatial.pth", batch_size=32):
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "dataset"

    test_roots = [str(data_root / "hf" / "spatial")]
    test_set = SpatialImageDataset(test_roots, augment=False)
    if len(test_set) == 0:
        raise RuntimeError(f"No test images found in: {test_roots}")

    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SpatialBranch(num_classes=2).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / max(1, total)
    print(f"Test samples: {total}")
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    evaluate()

