import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageFile
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from combine_model import DualBranchFusion

IMG_EXTS = (".png", ".jpg", ".jpeg")
ImageFile.LOAD_TRUNCATED_IMAGES = True


class GaussianNoise(object):
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor) * self.std + self.mean

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class DualBranchDataset(Dataset):
    """
    Provides paired tensors for the spatial and frequency branches. If no
    dedicated frequency roots are supplied, the same file path is used for both
    branches and separate transforms create the two views.
    """

    def __init__(
        self,
        spatial_roots: Sequence[str],
        freq_roots: Optional[Sequence[str]] = None,
        img_size: int = 224,
        augment: bool = True,
    ) -> None:
        if not spatial_roots:
            raise ValueError("spatial_roots must not be empty")

        if freq_roots is None:
            freq_roots = spatial_roots
        if len(freq_roots) != len(spatial_roots):
            raise ValueError("freq_roots must match spatial_roots in length")

        self.samples: List[Tuple[str, str, int]] = []
        for s_root, f_root in zip(spatial_roots, freq_roots):
            self._collect_samples(Path(s_root), Path(f_root), self.samples)

        if not self.samples:
            raise RuntimeError("DualBranchDataset found zero images.")

        if augment:
            self.spatial_tf = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
                ),
                transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.6, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
            ])
            self.freq_tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                GaussianNoise(0.0, 0.05),
            ])
        else:
            self.spatial_tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])
            self.freq_tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])

        self.spatial_norm = transforms.Normalize([0.5] * 3, [0.5] * 3)
        self.freq_norm = transforms.Normalize([0.5], [0.5])

    @staticmethod
    def _collect_samples(spatial_root: Path, freq_root: Path, sink: List[Tuple[str, str, int]]) -> None:
        for cls, label in (("real", 0), ("fake", 1)):
            s_folder = spatial_root / cls
            if not s_folder.exists():
                continue

            f_folder = freq_root / cls
            freq_candidates = {}
            if f_folder.exists():
                freq_candidates = {
                    f.name: f for f in f_folder.iterdir()
                    if f.is_file() and f.suffix.lower() in IMG_EXTS
                }

            for file in s_folder.iterdir():
                if not file.is_file() or file.suffix.lower() not in IMG_EXTS:
                    continue
                freq_path = freq_candidates.get(file.name, file)
                sink.append((str(file), str(freq_path), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        spatial_path, freq_path, label = self.samples[idx]

        spatial_img = Image.open(spatial_path).convert("RGB")
        spatial_tensor = self.spatial_tf(spatial_img)
        spatial_tensor = self.spatial_norm(spatial_tensor)

        freq_img = Image.open(freq_path).convert("L")
        freq_tensor = self.freq_tf(freq_img)
        freq_tensor = self.freq_norm(freq_tensor)

        return spatial_tensor, freq_tensor, label


def _effective_len(loader: DataLoader) -> int:
    if hasattr(loader, "sampler") and loader.sampler is not None:
        try:
            return len(loader.sampler)
        except TypeError:
            return len(loader.dataset)
    return len(loader.dataset)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, device_type):
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = _effective_len(loader)

    pbar = tqdm(loader, ncols=120)
    for spatial_imgs, freq_imgs, labels in pbar:
        spatial_imgs = spatial_imgs.to(device, non_blocking=True)
        freq_imgs = freq_imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device_type):
            logits = model(spatial_imgs, freq_imgs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * spatial_imgs.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()

        seen = max(1, pbar.n * loader.batch_size)
        pbar.set_description(f"loss={loss.item():.4f} acc={(correct/seen):.4f}")

    return total_loss / total_samples, correct / total_samples


def validate(model, loader, criterion, device, device_type):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = _effective_len(loader)

    with torch.no_grad():
        for spatial_imgs, freq_imgs, labels in loader:
            spatial_imgs = spatial_imgs.to(device, non_blocking=True)
            freq_imgs = freq_imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device_type):
                logits = model(spatial_imgs, freq_imgs)
                loss = criterion(logits, labels)

            total_loss += loss.item() * spatial_imgs.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()

    return total_loss / total_samples, correct / total_samples


def build_datasets(repo_root: Path, batch_size: int):
    data_root = repo_root / "dataset"
    split_root = data_root / "dataset_split"

    if not split_root.exists():
        raise RuntimeError(
            "Expected dataset/dataset_split to exist. Run src/split_train_val.py first."
        )

    train_root = split_root / "train"
    val_root = split_root / "val"
    train_spatial = train_root / "spatial"
    val_spatial = val_root / "spatial"
    train_freq = train_root / "freq"
    val_freq = val_root / "freq"

    if train_spatial.exists() and val_spatial.exists():
        spatial_train_roots = [str(train_spatial)]
        spatial_val_roots = [str(val_spatial)]
        freq_train_roots = [str(train_freq)] if train_freq.exists() else None
        freq_val_roots = [str(val_freq)] if val_freq.exists() else None
    else:
        if not train_root.exists() or not val_root.exists():
            raise RuntimeError(
                "dataset/dataset_split is missing train/val directories. "
                "Ensure split_train_val.py finished successfully."
            )
        spatial_train_roots = [str(train_root)]
        spatial_val_roots = [str(val_root)]
        freq_train_roots = None
        freq_val_roots = None

    train_set = DualBranchDataset(spatial_train_roots, freq_train_roots, augment=True)
    val_set = DualBranchDataset(spatial_val_roots, freq_val_roots, augment=False)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, len(train_set), len(val_set)


def main():
    parser = argparse.ArgumentParser(description="Train dual-branch fusion model.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fusion", choices=["concat", "avg"], default="concat")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--freeze-backbones", action="store_true")
    parser.add_argument("--spatial-ckpt", type=str, default=None)
    parser.add_argument("--freq-ckpt", type=str, default=None)
    parser.add_argument("--output", type=str, default="best_combine.pth")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    train_loader, val_loader, train_count, val_count = build_datasets(repo_root, args.batch_size)
    print("Train samples:", train_count)
    print("Val samples:", val_count)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    model = DualBranchFusion(
        num_classes=2,
        fusion=args.fusion,
        freeze_backbones=args.freeze_backbones,
    ).to(device)

    if args.spatial_ckpt and args.freq_ckpt:
        spatial_ckpt = Path(args.spatial_ckpt)
        freq_ckpt = Path(args.freq_ckpt)
        if not spatial_ckpt.exists():
            spatial_ckpt = repo_root / args.spatial_ckpt
        if not freq_ckpt.exists():
            freq_ckpt = repo_root / args.freq_ckpt
        model.load_branch_weights(str(spatial_ckpt), str(freq_ckpt), map_location=device)
        print("Loaded branch checkpoints.")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found. Disable --freeze-backbones or set fusion to 'concat'.")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    scaler = torch.amp.GradScaler(device_type)

    best_acc = 0.0
    best_loss = float("inf")
    patience = 10
    patience_counter = 0
    best_path = repo_root / args.output

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, device_type
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device, device_type)

        print(f"Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")

        scheduler.step(val_acc)

        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            best_acc = val_acc
            best_loss = val_loss
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
    print(f"Best validation loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()

