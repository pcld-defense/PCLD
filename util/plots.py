import os
import random
import re
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
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


def plot_softmax_trajectories(
        parquet_path: str,
        image: Optional[str] = None,
        epsilon: Optional[int] = None,
        attacked_model: str = 'adaptive',
        phase: str = 'val',
        top_k: int = 5,
        n_images: int = 1,
        class_names: Optional[list] = None,
        figsize: Optional[tuple] = None,
) -> plt.Figure:
    """Opens an attack parquet and plots per-class softmax trajectories across paint steps.

    Each row of the parquet (from ``paints_inference`` mode) represents one
    (image × paint-step) observation. This function groups rows by image,
    stacks the per-step softmax vectors, and plots the top-k class confidence
    curves against the stroke-count checkpoint axis.

    Args:
        parquet_path: Path to the parquet file produced by ``attacker()`` with
            ``output_type='paints_inference'``.
        image: Image name (basename without extension) to plot. If None, the
            first ``n_images`` images in the filtered data are used.
        epsilon: Epsilon filter (integer pixel-space budget). If None, all
            epsilon values present in the file are included.
        attacked_model: Row filter on the ``attacked_model`` column; typically
            ``'adaptive'`` or ``'naive'``.
        phase: Dataset split to filter on (``'train'`` or ``'val'``).
        top_k: Number of top classes to plot per image, ranked by their peak
            probability across all paint steps.
        n_images: Number of images to plot when ``image`` is None.
        class_names: Ordered list of class name strings aligned with the class
            index used during the attack. When provided, class indices are
            mapped to names in the legend; otherwise the true/predicted class
            names stored in the parquet are used for those positions and all
            other classes are labelled ``'class {idx}'``.
        figsize: Matplotlib figure size ``(width, height)``. Defaults to
            ``(10 * n_images, 5)``.

    Returns:
        The matplotlib Figure containing one subplot per selected image.

    Raises:
        ValueError: If no rows remain after applying the requested filters.
    """
    df = pd.read_parquet(parquet_path)

    mask = (df['attacked_model'] == attacked_model) & (df['phase'] == phase)
    if epsilon is not None:
        mask &= df['epsilon'] == epsilon
    df = df[mask].copy()

    if df.empty:
        raise ValueError(
            f'No rows after filtering: attacked_model={attacked_model!r}, '
            f'phase={phase!r}, epsilon={epsilon}'
        )

    available_images = df['image'].unique().tolist()
    images_to_plot = [image] if image is not None else available_images[:n_images]
    n = len(images_to_plot)

    if figsize is None:
        figsize = (10 * n, 5)

    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)

    for col, img_name in enumerate(images_to_plot):
        ax = axes[0, col]
        img_df = df[df['image'] == img_name].sort_values('t').reset_index(drop=True)

        if img_df.empty:
            ax.set_title(f'{img_name}: no data')
            continue

        t_values = img_df['t'].tolist()
        x_labels = ['orig' if t == 999999 else str(t) for t in t_values]
        x_pos = list(range(len(t_values)))

        # Shape: (steps, n_classes)
        probs_matrix = np.array(img_df['probs'].tolist())

        top_k_indices = np.argsort(probs_matrix.max(axis=0))[::-1][:top_k]

        actual_idx = int(img_df['actual'].iloc[0])
        actual_class_name = img_df['actual_class'].iloc[0]
        final_pred_class = img_df['pred_class'].iloc[-1]

        def _class_label(idx: int) -> str:
            if class_names is not None and idx < len(class_names):
                name = class_names[idx]
            elif idx == actual_idx:
                name = actual_class_name
            else:
                name = f'class {idx}'
            return f'{name} (true)' if idx == actual_idx else name

        for class_idx in top_k_indices:
            ax.plot(x_pos, probs_matrix[:, int(class_idx)],
                    label=_class_label(int(class_idx)),
                    linewidth=2, marker='o', markersize=4)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Paint step (t)')
        ax.set_ylabel('Softmax probability')
        eps_val = img_df['epsilon'].iloc[0]
        ax.set_title(
            f'{img_name}  |  {attacked_model} attack  |  ε={eps_val}\n'
            f'true: {actual_class_name}  →  final pred: {final_pred_class}'
        )
        ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
