"""Paint a small sample of CIFAR-10 images and save intermediate canvases."""

import os
import sys

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.utils import save_image

sys.path.insert(0, '/home/idanbib/PCLD/code')

from pcld.painter.painter_utils import load_painter, paint_images
from pcld.utils.consts import ACTOR_WEIGHTS_PATH, RENDERER_WEIGHTS_PATH, PainterConsts

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = '/home/ambarr/pcld/data/cifar10/val'
OUT_DIR  = '/home/idanbib/PCLD/data/examples/cifar10/painter_1'
IMAGES_PER_CLASS = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# With MAX_STEP=40, total strokes = 40*5 = 200
OUTPUT_EVERY = [10, 20, 40, 80, 120, 160, 200]

# ── Load painter ─────────────────────────────────────────────────────────────
print(f'Device: {DEVICE}')
print(f'MAX_STEP={PainterConsts.MAX_STEP}, DIVIDE={PainterConsts.DIVIDE}, WIDTH={PainterConsts.WIDTH}')
actor, renderer = load_painter(ACTOR_WEIGHTS_PATH, RENDERER_WEIGHTS_PATH, DEVICE)

# ── Collect images ───────────────────────────────────────────────────────────
to_tensor = T.ToTensor()   # → [0,1] float tensor (3, 32, 32)

images, labels, paths = [], [], []
for cls in sorted(os.listdir(DATA_DIR)):
    cls_dir = os.path.join(DATA_DIR, cls)
    if not os.path.isdir(cls_dir):
        continue
    files = sorted(os.listdir(cls_dir))[:IMAGES_PER_CLASS]
    for f in files:
        img_path = os.path.join(cls_dir, f)
        img = Image.open(img_path).convert('RGB')
        images.append(to_tensor(img))
        labels.append(cls)
        paths.append(f)

batch = torch.stack(images).to(DEVICE)          # (N, 3, 32, 32)
print(f'\nPainting {len(images)} images  shape={tuple(batch.shape)}')
print(f'output_every={OUTPUT_EVERY}')

# ── Paint ────────────────────────────────────────────────────────────────────
canvases = paint_images(batch, OUTPUT_EVERY, DEVICE, actor, renderer,
                        add_original=True)
# canvases: (N, Steps+1, 3, 32, 32)  — last step is the original
print(f'Output shape: {tuple(canvases.shape)}')

# ── Save ─────────────────────────────────────────────────────────────────────
step_names = [f'strokes_{s:03d}' for s in OUTPUT_EVERY] + ['original']

for img_i, (cls, fname) in enumerate(zip(labels, paths)):
    stem = os.path.splitext(fname)[0]
    img_dir = os.path.join(OUT_DIR, cls)
    os.makedirs(img_dir, exist_ok=True)

    for step_i, step_name in enumerate(step_names):
        out_path = os.path.join(img_dir, f'{stem}_{step_name}.png')
        save_image(canvases[img_i, step_i], out_path)

# Also save a grid: one row per image, columns = steps
from torchvision.utils import make_grid
N, S, C, H, W = canvases.shape
grid = make_grid(canvases.view(N * S, C, H, W), nrow=S, padding=2)
save_image(grid, os.path.join(OUT_DIR, 'grid_all.png'))

print(f'\nSaved {len(images) * len(step_names)} images + grid to {OUT_DIR}')
