import os
import random
import re
from collections import defaultdict

from PIL import Image
from matplotlib import pyplot as plt


def prepare_tensor_show(x):
    """
    Prepares a tensor for visualization by converting it to a CPU tensor and
    rearranging its dimensions if necessary.

    Args:
        x: The input tensor, expected to have dimensions
                          (C, H, W) or (H, W, C).

    Returns:
        The tensor rearranged to (H, W, C) for visualization.
    """
    x = x.detach().to('cpu')

    if x.ndim == 3:
        x = x.permute(1, 2, 0)

    return x


def create_gifs(source_folder, destination_folder, n=5, scale_factor=8):
    """
    Creates GIFs of randomly selected class images in the source folder.

    Args:
        source_folder: Path to the folder containing source images.
        destination_folder: Path to the folder where GIFs will be saved.
        n: Number of random groups to process. Default is 5.
        scale_factor: Factor by which to upscale the images. Default is 8.

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

    # Safety check: don't try to sample more than what exists
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



def plot_painter_results(canvases, output_every):
    paints_dict = {}
    for c_i in range(canvases.shape[1]):
        if c_i == canvases.shape[1] - 1:
            step = f't=∞'
        else:
            step = f't={output_every[c_i]}'
        paints_dict[step] = canvases[0, c_i]

    fig, axes = plt.subplots(1, len(output_every) + 1, figsize=(26, 23),
                             gridspec_kw={'wspace': 0, 'hspace': 0, 'left': 0, 'right': 1, 'bottom': 0, 'top': 1})

    for j, (step, entry_canvases) in enumerate(paints_dict.items()):
        canvas = entry_canvases[1]
        axes[j].imshow(prepare_tensor_show(canvas))
        axes[j].axis('off')
        if j == 0:
            axes[j].set_title(step, fontsize=30)
        j += 1

    plt.tight_layout()
    plt.show()