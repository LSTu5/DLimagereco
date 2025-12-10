
## Project overview
This repo builds a classifier to distinguish real vs. AI-generated images with two complementary branches:

- **Spatial branch** ([src/spatial_model.py](src/spatial_model.py)): CNN with residual-style `BasicBlock`s, global pooling, dropout, and a linear head. Trained via [src/train_spatial.py](src/train_spatial.py), which includes strong data augmentation, mixup, AMP, and AdamW.
- **Frequency branch** ([src/freq_model.py](src/freq_model.py)): Operates on log-magnitude FFT views; trained via [src/train_freq.py](src/train_freq.py).

## Data pipeline
1. **Download & merge** ([src/data_downloader.py](src/data_downloader.py)): Pulls Kaggle/Hugging Face datasets and merges class folders into `rawdata/*`.
2. **Process dual views** ([src/data_processing.py](src/data_processing.py)): Creates spatial (224×224 RGB) and frequency (320×320 log-FFT) tensors and saves them under `dataset/<source>/spatial|freq/{real,fake}`.
3. **Train/val split** ([src/split_train_val.py](src/split_train_val.py)): Splits processed data into `dataset/dataset_split/train|val/spatial|freq/{real,fake}` with a fixed seed.

## Training
- **Spatial:** `python src/train_spatial.py` (auto-detects `dataset/dataset_split`; falls back to in-memory split if absent).
- **Frequency:** `python src/train_freq.py`.

## Evaluation
- Use [src/eval_spatial.py](src/eval_spatial.py) (and corresponding freq eval) on saved checkpoints.

## Key scripts
- Download: `python src/data_downloader.py`
- Process: `python src/data_processing.py`
- Split: `python src/split_train_val.py`
- Train spatial: `python src/train_spatial.py`
- Train frequency: `python src/train_freq.py`
