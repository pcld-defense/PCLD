"""Train a RendererFCN at a given resolution on random Bezier strokes.

The renderer maps 10 geometric stroke parameters to a WIDTH×WIDTH alpha mask.
Training data is generated on-the-fly: random stroke params → rasterised via
``stroke_gen.draw()`` → MSE loss against the renderer's output.

Usage:
    python scripts/train_renderer.py --width 32 --max_step 500000
    python scripts/train_renderer.py --width 128 --max_step 500000
    python scripts/train_renderer.py --width 224 --max_step 500000

Reference: Huang et al., "Learning to Paint" (ICCV 2019)
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from painter.painter import RendererFCN


# ── Bezier stroke rasterisation (from Learning to Paint) ──────────────────

def normal(x: float, width: int) -> int:
    return int(x * (width - 1) + 0.5)


def draw(f: np.ndarray, width: int = 128) -> np.ndarray:
    """Rasterises a single Bezier stroke as a width×width alpha mask.

    Args:
        f: 10-element array of normalised stroke parameters in [0, 1].
        width: Output mask side length in pixels.

    Returns:
        Inverted alpha mask of shape (width, width) in [0, 1].
    """
    x0, y0, x1, y1, x2, y2, z0, z2, w0, w2 = f
    x1 = x0 + (x2 - x0) * x1
    y1 = y0 + (y2 - y0) * y1
    x0 = normal(x0, width * 2)
    x1 = normal(x1, width * 2)
    x2 = normal(x2, width * 2)
    y0 = normal(y0, width * 2)
    y1 = normal(y1, width * 2)
    y2 = normal(y2, width * 2)
    z0 = int(1 + z0 * width // 2)
    z2 = int(1 + z2 * width // 2)
    canvas = np.zeros([width * 2, width * 2], dtype=np.float32)
    tmp = 1. / 100
    for i in range(100):
        t = i * tmp
        x = int((1 - t) * (1 - t) * x0 + 2 * t * (1 - t) * x1 + t * t * x2)
        y = int((1 - t) * (1 - t) * y0 + 2 * t * (1 - t) * y1 + t * t * y2)
        z = int((1 - t) * z0 + t * z2)
        w = (1 - t) * w0 + t * w2
        cv2.circle(canvas, (y, x), z, w, -1)
    return 1 - cv2.resize(canvas, dsize=(width, width))


# ── Training loop ─────────────────────────────────────────────────────────

def train(width: int, max_step: int, batch_size: int, save_dir: str,
          device: str, log_interval: int = 1000) -> None:
    """Trains a RendererFCN from scratch on random strokes.

    Args:
        width: Renderer output resolution.
        max_step: Total training iterations.
        batch_size: Number of random strokes per batch.
        save_dir: Directory for saving checkpoints.
        device: Torch device string.
        log_interval: Print loss every N steps.
    """
    renderer = RendererFCN(width=width).to(device)
    optimizer = torch.optim.Adam(renderer.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'renderer_{width}.pkl')
    print(f'Training RendererFCN(width={width}) for {max_step} steps')
    print(f'  Save path: {save_path}')

    t0 = time.time()
    for step in range(1, max_step + 1):
        # Learning rate schedule: 1e-4 → 1e-5 → 1e-6
        if step == max_step * 2 // 5:
            for g in optimizer.param_groups:
                g['lr'] = 1e-5
        if step == max_step * 4 // 5:
            for g in optimizer.param_groups:
                g['lr'] = 1e-6

        # Generate random strokes
        params = np.random.rand(batch_size, 10).astype(np.float32)
        targets = np.stack([draw(params[i], width) for i in range(batch_size)])

        params_t = torch.from_numpy(params).to(device)
        targets_t = torch.from_numpy(targets).to(device)

        pred = 1 - renderer(params_t)  # renderer returns inverted; invert back
        loss = criterion(pred, targets_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_interval == 0:
            elapsed = time.time() - t0
            steps_per_sec = step / elapsed
            print(f'  Step {step}/{max_step}  loss={loss.item():.6f}  '
                  f'{steps_per_sec:.0f} steps/sec')

        if step % (max_step // 10) == 0:
            torch.save(renderer.state_dict(), save_path)
            print(f'  Checkpoint saved to {save_path}')

    torch.save(renderer.state_dict(), save_path)
    print(f'Training complete. Final model: {save_path}')


def main():
    parser = argparse.ArgumentParser(description='Train RendererFCN at a given resolution')
    parser.add_argument('--width', type=int, required=True, help='Output resolution (must be divisible by 8)')
    parser.add_argument('--max_step', type=int, default=500000, help='Total training iterations')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--save_dir', type=str, default='models/painter_renderer', help='Save directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    train(args.width, args.max_step, args.batch_size, args.save_dir, args.device)


if __name__ == '__main__':
    main()
