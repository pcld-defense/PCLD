"""Visualize the painting process and save as GIF.

Paints a sample of images at native resolution and creates a GIF showing
the progressive painting at each stroke checkpoint.

Usage:
    python scripts/visualize_painting.py --width 32 \
        --actor_path models/painter_actor/actor_cifar.pkl \
        --renderer_path models/painter_renderer/renderer_cifar.pkl \
        --n_images 10 --output results/painting_cifar.gif
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from painter.painter import ActorResNet, RendererFCN
from painter.painter_config import PainterConfig
from painter.painter_utils import paint


def create_painting_gif(images: torch.Tensor, output_every: list[int],
                        actor: ActorResNet, renderer: RendererFCN,
                        config: PainterConfig, device: str,
                        output_path: str, scale: int = 4,
                        duration_ms: int = 300) -> None:
    """Paints images and saves a GIF of the painting process.

    Each frame shows all images side-by-side at a given paint step.
    The first frame shows the original images, then progressive paint steps.

    Args:
        images: Batch of images (N, 3, H, W) in [0, 1].
        output_every: Stroke checkpoints to capture.
        actor: Trained ActorResNet.
        renderer: Trained RendererFCN.
        config: PainterConfig for this dataset.
        device: Torch device.
        output_path: Path to save the GIF.
        scale: Upscale factor for visualization (32×32 is tiny).
        duration_ms: Milliseconds per frame.
    """
    N = images.shape[0]
    H, W = images.shape[2], images.shape[3]

    # Paint all images
    canvases = paint(images, output_every, device, actor, renderer, config=config)
    # canvases: (N, Steps, 3, H, W) in [0, 1]

    steps = canvases.shape[1]
    frames = []

    # Add original images as first frame
    orig_row = _make_row(images, scale)
    frames.append(orig_row)

    # Add each paint step as a frame
    for s in range(steps):
        row = _make_row(canvases[:, s], scale)
        frames.append(row)

    # Add original again as last frame (hold)
    frames.append(orig_row)

    # Create GIF
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f'Saved painting GIF ({len(frames)} frames, {N} images) to {output_path}')

    # Also save a static grid (all steps for all images)
    grid = _make_grid(images, canvases, output_every, scale)
    grid_path = output_path.replace('.gif', '_grid.png')
    Image.fromarray(grid).save(grid_path)
    print(f'Saved static grid to {grid_path}')


def _make_row(images: torch.Tensor, scale: int) -> np.ndarray:
    """Creates a horizontal row of images as a numpy array."""
    imgs = []
    for i in range(images.shape[0]):
        img = (images[i].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        if scale > 1:
            img = np.array(Image.fromarray(img).resize(
                (img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST))
        imgs.append(img)
    # Add 2px white separator between images
    sep = np.ones((imgs[0].shape[0], 2, 3), dtype=np.uint8) * 255
    row = imgs[0]
    for img in imgs[1:]:
        row = np.concatenate([row, sep, img], axis=1)
    return row


def _make_grid(originals: torch.Tensor, canvases: torch.Tensor,
               output_every: list[int], scale: int) -> np.ndarray:
    """Creates a grid: rows=images, columns=steps (original + paint steps)."""
    N = originals.shape[0]
    steps = canvases.shape[1]
    rows = []
    for i in range(N):
        # Original
        imgs = [originals[i]]
        # Paint steps
        for s in range(steps):
            imgs.append(canvases[i, s])
        row = _make_row(torch.stack(imgs), scale)
        rows.append(row)
    # Add 2px white separator between rows
    sep = np.ones((2, rows[0].shape[1], 3), dtype=np.uint8) * 255
    grid = rows[0]
    for row in rows[1:]:
        grid = np.concatenate([grid, sep, row], axis=0)
    return grid


def main():
    parser = argparse.ArgumentParser(description='Visualize painting process')
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--actor_path', type=str, required=True)
    parser.add_argument('--renderer_path', type=str, required=True)
    parser.add_argument('--n_images', type=int, default=10)
    parser.add_argument('--output', type=str, default='results/painting.gif')
    parser.add_argument('--output_every', type=str, default=None,
                        help='Comma-separated stroke checkpoints')
    parser.add_argument('--max_step', type=int, default=40)
    parser.add_argument('--scale', type=int, default=4,
                        help='Upscale factor for visualization')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    if args.output_every:
        output_every = [int(x) for x in args.output_every.split(',')]
    else:
        # Default: sample every 10 strokes up to max_step * 5
        max_strokes = args.max_step * 5
        output_every = list(range(10, max_strokes + 1, 10))

    config = PainterConfig(width=args.width, divide=1, max_step=args.max_step)

    actor = ActorResNet(width=args.width)
    actor.load_state_dict(torch.load(args.actor_path, map_location=args.device))
    actor.eval()

    renderer = RendererFCN(width=args.width)
    renderer.load_state_dict(torch.load(args.renderer_path, map_location=args.device))
    renderer.eval()

    # Generate random test images (since we may not have CIFAR-10 locally)
    print(f'Generating {args.n_images} random test images at {args.width}×{args.width}')
    images = torch.rand(args.n_images, 3, args.width, args.width)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    create_painting_gif(images, output_every, actor, renderer, config,
                        args.device, args.output, scale=args.scale)


if __name__ == '__main__':
    main()
