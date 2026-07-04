import os

from tqdm import tqdm
from huggingface_hub import login
from datasets import load_dataset

from pcld.utils.consts import RESOURCES_DATASETS_DIR

hf_token = os.getenv("HF_TOKEN")
login(hf_token)
ds = load_dataset("uoft-cs/cifar10", split="train", trust_remote_code=True)
output_dir = os.path.join(RESOURCES_DATASETS_DIR, "cifar10", "train")

class_names = ds.features["label"].names

print("Saving images to class-specific folders...")
for i, example in enumerate(tqdm(ds)):
    image = example["img"]
    label_idx = example["label"]
    class_name = class_names[label_idx]
    class_path = os.path.join(output_dir, class_name)
    os.makedirs(class_path, exist_ok=True)
    image.save(os.path.join(class_path, f"{i}.png"))

print(f"Done! Images saved in {output_dir}")
