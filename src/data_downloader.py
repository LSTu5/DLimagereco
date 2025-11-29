#other relevant datasets
# https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
# https://huggingface.co/datasets/pujanpaudel/deepfake_face_classification
import os
import shutil
from pathlib import Path
import kagglehub
from huggingface_hub import snapshot_download


def merge_on_download(dataset_path, real_dirs=None, fake_dirs=None, output_dir="../data/merged"):
    """
    After downloading a dataset, instantly merge the folders you specify
    into a unified structure:
        output_dir/real/
        output_dir/fake/
    """

    real_dirs = real_dirs or []
    fake_dirs = fake_dirs or []

    out_real = Path(output_dir) / "real"
    out_fake = Path(output_dir) / "fake"

    out_real.mkdir(parents=True, exist_ok=True)
    out_fake.mkdir(parents=True, exist_ok=True)

    def copy_all(src_list, dest):
        count = 0
        for s in src_list:
            s = Path(dataset_path) / s
            if not s.exists():
                print(f"Missing folder: {s}")
                continue
            for root, _, files in os.walk(s):
                for f in files:
                    src_file = Path(root) / f
                    shutil.copy2(src_file, dest / f)
                    count += 1
        return count

    copied_real = copy_all(real_dirs, out_real)
    copied_fake = copy_all(fake_dirs, out_fake)

    print(f"Real copied: {copied_real}, Fake copied: {copied_fake}")
    print(f"Merged → {output_dir}")


pathA = kagglehub.dataset_download("saurabhbagchi/deepfake-image-detection")
merge_on_download(
    dataset_path=pathA,
    real_dirs=["test-20250112T065939Z-001/test/real", "train-20250112T065955Z-001/train/real"],
    fake_dirs=["test-20250112T065939Z-001/test/fake", "train-20250112T065955Z-001/train/fake"],
    output_dir="../rawdata/kaggle_a"
)

pathB = kagglehub.dataset_download("tristanzhang32/ai-generated-images-vs-real-images")
merge_on_download(
    dataset_path=pathB,
    real_dirs=["test/real", "train/real"],
    fake_dirs=["test/fake", "train/fake"],
    output_dir="../rawdata/kaggle_b"
)

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
