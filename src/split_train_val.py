import random
import shutil
from pathlib import Path
from typing import Iterable, List

VAL_RATIO = 0.1
SEED = 42
IMG_EXTS = {".png", ".jpg", ".jpeg"}


def list_images(root: Path, modality: str, cls: str) -> List[Path]:
    folder = root / modality / cls
    if not folder.exists():
        return []
    return [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]


def split_and_copy(sources: Iterable[Path], out_root: Path, val_ratio: float = VAL_RATIO) -> None:
    rng = random.Random(SEED)
    out_train = out_root / "train"
    out_val = out_root / "val"

    for modality in ("spatial", "freq"):
        for cls in ("real", "fake"):
            files: List[Path] = []
            for src in sources:
                files.extend(list_images(src, modality, cls))

            if not files:
                print(f"[skip] {modality}/{cls}: none found")
                continue

            rng.shuffle(files)
            val_count = max(1, int(len(files) * val_ratio))
            val_files = files[:val_count]
            train_files = files[val_count:]

            dst_train = out_train / modality / cls
            dst_val = out_val / modality / cls
            dst_train.mkdir(parents=True, exist_ok=True)
            dst_val.mkdir(parents=True, exist_ok=True)

            for p in train_files:
                shutil.copy2(p, dst_train / p.name)
            for p in val_files:
                shutil.copy2(p, dst_val / p.name)

            print(f"{modality}/{cls}: total={len(files)} train={len(train_files)} val={len(val_files)}")


def main() -> None:
    """Main entry point: create train/val split from processed datasets.
    
    This script should be run after data_processing.py has created the processed
    datasets in ../dataset/. By default, it combines kaggle_a and kaggle_b for
    training, leaving hf dataset for testing.
    
    You can modify the 'sources' list to include different datasets or change
    the validation ratio as needed.
    """
    # Navigate to repo root and locate processed datasets
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "dataset"
    
    # Specify which processed datasets to include in train/val split
    # Note: hf dataset is reserved for final testing
    sources = [data_root / "kaggle_a", data_root / "kaggle_b"]

    # Create output directory for split dataset
    out_root = data_root / "dataset_split"
    out_root.mkdir(parents=True, exist_ok=True)

    # Perform the split and copy files
    split_and_copy(sources, out_root, val_ratio=VAL_RATIO)
    print(f"Done. Train/val splits at: {out_root}")


if __name__ == "__main__":
    main()

