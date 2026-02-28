import os
import random
import re
from collections import defaultdict

import torch
from PIL import Image
from matplotlib import pyplot as plt


def prepare_tensor_show(x: torch.Tensor) -> torch.Tensor:
    """Converts a CHW tensor to HWC format suitable for matplotlib.

    Moves the tensor to CPU and permutes its dimensions from (C, H, W)
    to (H, W, C) so it can be passed directly to imshow.

    Args:
        x: Image tensor of shape (C, H, W) or already (H, W, C).

    Returns:
        CPU tensor of shape (H, W, C).
    """
    x = x.detach().to('cpu')
    if x.ndim == 3:
        x = x.permute(1, 2, 0)
    return x


def create_gifs(source_folder: str, destination_folder: str, n: int = 5,
                scale_factor: int = 8) -> None:
    """Creates animated GIFs from groups of sequentially numbered PNG files.

    Groups files in `source_folder` by the numeric prefix that precedes
    '_generated<number>.png', randomly selects `n` groups, and saves one GIF
    per group to `destination_folder`. Images are upscaled by `scale_factor`
    using nearest-neighbour interpolation to keep pixels crisp.

    Args:
        source_folder: Directory containing PNG files named
            '<prefix>_generated<order>.png'.
        destination_folder: Directory where the output GIF files will be saved.
        n: Maximum number of randomly selected groups to convert. If fewer
            groups exist, all of them are used.
        scale_factor: Integer factor by which to upscale each frame.
    """
    groups = defaultdict(list)
    pattern = re.compile(r'(\d+)_generated(\d+)\.png')
    os.makedirs(destination_folder, exist_ok=True)

    for filename in os.listdir(source_folder):
        match = pattern.match(filename)
        if match:
            prefix = match.group(1)
            order_num = int(match.group(2))
            groups[prefix].append((order_num, filename))

    available_prefixes = list(groups.keys())
    num_to_sample = min(n, len(available_prefixes))
    selected_prefixes = random.sample(available_prefixes, num_to_sample)

    print(f"Selecting {num_to_sample} random groups: {selected_prefixes}")

    for prefix in selected_prefixes:
        files = groups[prefix]
        files.sort()

        frames = []
        for _, filename in files:
            img_path = os.path.join(source_folder, filename)
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                new_size = (img.width * scale_factor, img.height * scale_factor)
                img_resized = img.resize(new_size, resample=Image.NEAREST)
                frames.append(img_resized)

        if frames:
            output_name = f"{destination_folder}/random_{prefix}_animation.gif"
            frames[0].save(
                output_name,
                save_all=True,
                append_images=frames[1:],
                duration=150,
                loop=0
            )
            print(f"Done: {output_name}")


def plot_painter_results(canvases: torch.Tensor, output_every: list[int]) -> None:
    """Displays a side-by-side matplotlib figure of painting progress snapshots.

    Iterates over the step dimension of `canvases` and plots each canvas in a
    separate subplot. The last canvas (index == shape[1]-1) is labelled 't=∞'
    (the original image); all others are labelled with their stroke count from
    `output_every`.

    Args:
        canvases: Canvas tensor of shape (1, Steps, 2, H, W) where the second
            sub-index selects between two canvas variants (index 1 is displayed).
        output_every: List of stroke-count checkpoints corresponding to the
            first Steps-1 canvases.
    """
    paints_dict = {}
    for c_i in range(canvases.shape[1]):
        if c_i == canvases.shape[1] - 1:
            step = f't=∞'
        else:
            step = f't={output_every[c_i]}'
        paints_dict[step] = canvases[0, c_i]

    fig, axes = plt.subplots(1, len(output_every) + 1, figsize=(26, 23),
                             gridspec_kw={'wspace': 0, 'hspace': 0, 'left': 0, 'right': 1,
                                          'bottom': 0, 'top': 1})

    for j, (step, entry_canvases) in enumerate(paints_dict.items()):
        canvas = entry_canvases[1]
        axes[j].imshow(prepare_tensor_show(canvas))
        axes[j].axis('off')
        if j == 0:
            axes[j].set_title(step, fontsize=30)
        j += 1

    plt.tight_layout()
    plt.show()
