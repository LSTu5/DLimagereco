"""Dataset downloader for AI-generated image detection.

This script downloads three public datasets containing real and AI-generated images:
1. Kaggle dataset A: saurabhbagchi/deepfake-image-detection
2. Kaggle dataset B: tristanzhang32/ai-generated-images-vs-real-images  
3. Hugging Face dataset: Hemg/AI-Generated-vs-Real-Images-Datasets

Other relevant datasets for future consideration:
https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
https://huggingface.co/datasets/pujanpaudel/deepfake_face_classification
"""

import os
import shutil
from pathlib import Path
import kagglehub
from huggingface_hub import snapshot_download


def merge_on_download(dataset_path, real_dirs=None, fake_dirs=None, output_dir="../data/merged"):
    """
    Merge downloaded dataset folders into a unified binary classification structure.
    
    After downloading a dataset, this function organizes the specified subdirectories
    into a clean two-class structure suitable for training:
        output_dir/real/  - contains all real images
        output_dir/fake/  - contains all AI-generated/fake images
    
    Args:
        dataset_path: Root path where the dataset was downloaded
        real_dirs: List of subdirectory paths containing real images (relative to dataset_path)
        fake_dirs: List of subdirectory paths containing fake/AI-generated images
        output_dir: Destination directory for the merged dataset structure
        
    Returns:
        None (prints statistics about copied files)
    """

    real_dirs = real_dirs or []
    fake_dirs = fake_dirs or []

    # Create output directory structure for binary classification
    out_real = Path(output_dir) / "real"
    out_fake = Path(output_dir) / "fake"

    out_real.mkdir(parents=True, exist_ok=True)
    out_fake.mkdir(parents=True, exist_ok=True)

    def copy_all(src_list, dest):
        """Helper function to copy all files from source directories to destination.
        
        Args:
            src_list: List of source directory paths to copy from
            dest: Destination directory path
            
        Returns:
            count: Total number of files copied
        """
        count = 0
        for s in src_list:
            s = Path(dataset_path) / s
            if not s.exists():
                print(f"Missing folder: {s}")
                continue
            # Walk through all subdirectories and copy every file
            for root, _, files in os.walk(s):
                for f in files:
                    src_file = Path(root) / f
                    # copy2 preserves metadata (timestamps, permissions)
                    shutil.copy2(src_file, dest / f)
                    count += 1
        return count

    # Copy all real and fake images to their respective output directories
    copied_real = copy_all(real_dirs, out_real)
    copied_fake = copy_all(fake_dirs, out_fake)

    print(f"Real copied: {copied_real}, Fake copied: {copied_fake}")
    print(f"Merged → {output_dir}")


# === DOWNLOAD AND MERGE DATASET A ===
# Kaggle deepfake detection dataset with test/train splits
pathA = kagglehub.dataset_download("saurabhbagchi/deepfake-image-detection")
merge_on_download(
    dataset_path=pathA,
    # Specify the subdirectories containing real images from both test and train
    real_dirs=["test-20250112T065939Z-001/test/real", "train-20250112T065955Z-001/train/real"],
    # Specify the subdirectories containing fake images from both test and train
    fake_dirs=["test-20250112T065939Z-001/test/fake", "train-20250112T065955Z-001/train/fake"],
    output_dir="../rawdata/kaggle_a"
)

# === DOWNLOAD AND MERGE DATASET B ===
# Kaggle AI-generated vs real images dataset
pathB = kagglehub.dataset_download("tristanzhang32/ai-generated-images-vs-real-images")
merge_on_download(
    dataset_path=pathB,
    real_dirs=["test/real", "train/real"],
    fake_dirs=["test/fake", "train/fake"],
    output_dir="../rawdata/kaggle_b"
)

# === DOWNLOAD AND MERGE HUGGING FACE DATASET ===
# Art-focused dataset comparing real art with AI-generated art
pathHF = snapshot_download("Hemg/AI-Generated-vs-Real-Images-Datasets", repo_type="dataset")
merge_on_download(
    dataset_path=pathHF,
    real_dirs=["RealArt/RealArt"],
    fake_dirs=["AiArtData/AiArtData"],
    output_dir="../rawdata/hf"
)

# Following code downloads datasets with original structure preserved.
# pathKaggleDeepfakeA = kagglehub.dataset_download("saurabhbagchi/deepfake-image-detection")
# kaggle_a = "../data/kaggle_a"
# os.makedirs(kaggle_a, exist_ok=True)
# shutil.copytree(pathKaggleDeepfakeA, kaggle_a, dirs_exist_ok=True)
#
# pathKaggleDeepfakeB = kagglehub.dataset_download("tristanzhang32/ai-generated-images-vs-real-images")
# kaggle_b = "../data/kaggle_b"
# os.makedirs(kaggle_a, exist_ok=True)
# shutil.copytree(pathKaggleDeepfakeB, kaggle_b, dirs_exist_ok=True)
#
# pathHuggingFace = snapshot_download(repo_id="Hemg/AI-Generated-vs-Real-Images-Datasets", repo_type="dataset")
# hf = "../data/hf"
# os.makedirs(hf, exist_ok=True)
# shutil.copytree(pathHuggingFace, hf, dirs_exist_ok=True)
