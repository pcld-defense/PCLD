import glob
import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt

from pcld.painter.paint_utils_old import paint_images as paint_images_old
from pcld.painter.painter_utils import paint_images, load_painter
from pcld.utils.consts import ROOT_DIR, RENDERER_WEIGHTS_PATH, ACTOR_WEIGHTS_PATH
from pcld.data.datasets import load_image


def prepare_tensor_show(canvas):
    """Convert a (3, H, W) float tensor in [0,1] to a (H, W, 3) numpy array."""
    return canvas.permute(1, 2, 0).cpu().float().numpy()


def plot_painter_results(output, output_every, title="Painter output"):
    """Visualise the painting progression for the first image in a batch.

    Args:
        output:       (B, Steps+1, 3, H, W) float tensor in [0, 1]
        output_every: list of stroke-count checkpoints (length = Steps)
        title:        figure title
    """
    steps = output.shape[1]
    fig, axes = plt.subplots(1, steps, figsize=(4 * steps, 4),
                             gridspec_kw={'wspace': 0.05})
    if steps == 1:
        axes = [axes]

    labels = [f't={s}' for s in output_every] + ['t=∞']
    for ax, label, idx in zip(axes, labels, range(steps)):
        ax.imshow(prepare_tensor_show(output[0, idx]))
        ax.set_title(label, fontsize=10)
        ax.axis('off')

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def benchmark_painter(device):
    output_every = [2, 10, 50, 200, 500, 1000, 5200]

    actor, renderer = load_painter(ACTOR_WEIGHTS_PATH, RENDERER_WEIGHTS_PATH, device)
    data_path = os.path.join(ROOT_DIR, "resources", "toy_images")
    img_paths = glob.glob(os.path.join(data_path, "*.png")) + \
                glob.glob(os.path.join(data_path, "*.jpg"))

    if not img_paths:
        print(f"No images found in {data_path}")
        return

    print(f"Found {len(img_paths)} images")

    # --- Load all images ---
    imgs = []
    for img_path in img_paths:
        entry_img = load_image(img_path, width=224, height=224)
        img = torch.tensor(np.expand_dims(entry_img, axis=0)).float()
        img = img.permute(0, 3, 1, 2)  # (1, 3, H, W)
        imgs.append(img)

    batch = torch.cat(imgs, dim=0).to(device)  # (N, 3, H, W)

    # --- New painter: process the entire batch at once ---
    start_time = time.time()
    output_new = paint_images(x=batch,
                              output_every=output_every,
                              device=device,
                              actor=actor,
                              renderer=renderer,
                              add_original=True)
    new_total = time.time() - start_time

    # --- Old painter: process images one by one ---
    old_outputs = []
    start_time = time.time()
    for i in range(batch.shape[0]):
        out = paint_images_old(x=batch[i:i + 1],
                               output_every=output_every,
                               device=device,
                               actor=actor,
                               renderer=renderer,
                               add_original=True)
        old_outputs.append(out)
    old_total = time.time() - start_time
    output_old = torch.cat(old_outputs, dim=0)

    n = batch.shape[0]
    print(f"\nResults over {n} images:")
    print(f"  New painter (batched):     {new_total:.2f}s total  |  {new_total / n:.2f}s per image")
    print(f"  Old painter (sequential):  {old_total:.2f}s total  |  {old_total / n:.2f}s per image")
    print(f"  Speedup: {old_total / new_total:.2f}x")

    plot_painter_results(output_new, output_every, title="New painter (batched)")
    plot_painter_results(output_old, output_every, title="Old painter (sequential)")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    benchmark_painter(device)
