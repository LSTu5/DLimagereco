import kagglehub
import os
import shutil
from huggingface_hub import snapshot_download

#other relevant datasets
# https://www.kaggle.com/datasets/saurabhbagchi/deepfake-image-detection?select=Sample_fake_images
# https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
# https://huggingface.co/datasets/pujanpaudel/deepfake_face_classification

pathKaggleDeepfakeA = kagglehub.dataset_download("alessandrasala79/ai-vs-human-generated-dataset")
kaggle_a = "../data/kaggle_a"
os.makedirs(kaggle_a, exist_ok=True)
shutil.copytree(pathKaggleDeepfakeA, kaggle_a, dirs_exist_ok=True)
pathKaggleDeepfakeB = kagglehub.dataset_download("tristanzhang32/ai-generated-images-vs-real-images"
)

kaggle_b = "../data/kaggle_b"
os.makedirs(kaggle_a, exist_ok=True)
shutil.copytree(pathKaggleDeepfakeB, kaggle_b, dirs_exist_ok=True)
pathHuggingFace = snapshot_download(repo_id="Hemg/AI-Generated-vs-Real-Images-Datasets", repo_type="dataset")

hf = "../data/hf"
os.makedirs(hf, exist_ok=True)
shutil.copytree(pathHuggingFace, hf, dirs_exist_ok=True)
