import os

from huggingface_hub import login
from datasets import load_dataset
from PIL import Image
from dotenv import load_dotenv

from util.consts import IMAGENET_SHAPE


hf_token = os.getenv("HF_TOKEN")
login(hf_token)
ds = load_dataset("imagenet-1k", split="train", streaming=True, trust_remote_code=True)

images_per_class = 200
total_classes = 1000
target_total = images_per_class * total_classes
output_dir = "/home/idanbib/PCLD/data/imagenet/train"

counts = {i: 0 for i in range(total_classes)}
total_saved = 0

print(f"Starting download: Aiming for {images_per_class} images per class...")

for example in ds:
    label = example['label']

    if counts[label] < images_per_class:
        class_path = os.path.join(output_dir, str(label))
        os.makedirs(class_path, exist_ok=True)

        img_filename = f"{label}_{counts[label]}.jpg"
        example['image'].convert("RGB").resize((IMAGENET_SHAPE, IMAGENET_SHAPE), Image.Resampling.LANCZOS).save(
            os.path.join(class_path, img_filename))

        counts[label] += 1
        total_saved += 1

        if total_saved % 500 == 0:
            print(f"Saved {total_saved}/{target_total} images...")

    if total_saved >= target_total:
        break

print("Success! Balanced dataset created.")