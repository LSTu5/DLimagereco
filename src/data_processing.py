import os
import shutil
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np

def process_dataset(raw_root, output_root, dataset_name):
    """
    Copies images from rawdata structure to the unified dataset structure.
    rawdata/
      dataset_name/
        real/
        fake/
    
    to
    
    dataset/
      dataset_name/
        real/
        fake/
        
    It also standardizes images to RGB png/jpg to ensure consistency, 
    but keeps them as close to raw as possible (no resizing here).
    """
    raw_path = Path(raw_root) / dataset_name
    out_path = Path(output_root) / dataset_name
    
    if not raw_path.exists():
        print(f"Raw dataset not found: {raw_path}")
        return

    print(f"Processing {dataset_name} from {raw_path} to {out_path}...")

    # Statistics
    stats = {"real": 0, "fake": 0}

    for cls in ["real", "fake"]:
        src_dir = raw_path / cls
        dst_dir = out_path / cls
        
        if not src_dir.exists():
            continue
            
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        # List all files
        files = [f for f in src_dir.iterdir() if f.is_file()]
        
        for f in tqdm(files, desc=f"{dataset_name}/{cls}", ncols=100):
            try:
                # We simply copy the file if it's a valid image
                # Check if valid image first
                with Image.open(f) as img:
                    img.verify() 
                
                # If valid, copy
                shutil.copy2(f, dst_dir / f.name)
                stats[cls] += 1
                
            except Exception as e:
                print(f"Skipping corrupted/invalid file {f}: {e}")

    print(f"Finished {dataset_name}: {stats}")


def main():
    # Adjust these paths to where your raw data actually lives
    # Assuming standard structure:
    # project_root/
    #   rawdata/
    #     kaggle_a/
    #     kaggle_b/
    #     hf/
    #   dataset/ (output)
    
    repo_root = Path(__file__).resolve().parents[1]
    raw_root = repo_root / "rawdata" 
    output_root = repo_root / "dataset"
    
    if not raw_root.exists():
        # Fallback for user's potential path
        raw_root = Path("../rawdata")
        if not raw_root.exists():
             print("Could not find 'rawdata' folder in ../rawdata or project root.")
             return

    # Process all three
    process_dataset(raw_root, output_root, "kaggle_a")
    process_dataset(raw_root, output_root, "kaggle_b")
    process_dataset(raw_root, output_root, "hf")

if __name__ == "__main__":
    main()

