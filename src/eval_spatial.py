"""Evaluation script for the trained spatial branch model.

This script loads a trained model checkpoint and evaluates it on a held-out test set.
By default, it uses the Hugging Face (hf) dataset which contains art-focused images,
providing a good test of generalization to a different image domain.

Usage:
    python eval_spatial.py

The script will:
1. Load the best_spatial.pth checkpoint
2. Evaluate on the hf dataset (art images)
3. Report overall accuracy

For more detailed evaluation (per-class metrics, confusion matrix, etc.),
consider extending this script with additional metrics.
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from train_spatial import SpatialImageDataset
from spatial_model import SpatialBranch


def evaluate(checkpoint="best_spatial.pth", batch_size=32):
    """Evaluate a trained spatial model on the test set.
    
    This function:
    1. Loads the specified checkpoint
    2. Creates a test dataset (no augmentation)
    3. Runs inference on all test samples
    4. Reports accuracy
    
    Args:
        checkpoint: Path to model checkpoint file (default: "best_spatial.pth")
        batch_size: Batch size for evaluation (default: 32)
        
    Returns:
        None (prints results to console)
    """
    # Locate test dataset
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "dataset"

    # Use HF dataset as test set (different domain from training data)
    test_roots = [str(data_root / "hf" / "spatial")]
    test_set = SpatialImageDataset(test_roots, augment=False)
    if len(test_set) == 0:
        raise RuntimeError(f"No test images found in: {test_roots}")

    # Create data loader (no shuffling needed for evaluation)
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SpatialBranch(num_classes=2).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()  # Set to evaluation mode

    # Run evaluation
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient computation for efficiency
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # Report results
    acc = correct / max(1, total)
    print(f"Test samples: {total}")
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    evaluate()

