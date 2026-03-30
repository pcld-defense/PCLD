"""Train learned up/downscale layers for the painting pipeline.

Freezes the existing 128×128 painter and trains only the LearnedUpscale
and LearnedDownscale modules to minimize reconstruction MSE between
the painted output and the original image.

The training objective:
    loss = MSE(downscale(paint(upscale(x))), x)

Only the upscale/downscale conv layers have gradients.  The painter
(actor+renderer) is frozen and runs inside torch.no_grad().

Usage:
    python scripts/train_learned_resize.py \
        --data_dir /path/to/cifar10/train \
        --actor_path /path/to/actor.pkl \
        --renderer_path /path/to/renderer.pkl \
        --save_dir /path/to/output
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from painter.painter import ActorResNet, RendererFCN
from painter.painter_utils import decode
from painter.learned_resize import LearnedUpscale, LearnedDownscale


def load_images(data_dir: str, max_images: int = 50000) -> torch.Tensor:
    """Loads images at native resolution (no resize)."""
    transform = transforms.ToTensor()
    images = []
    for root, _, files in os.walk(data_dir):
        for f in sorted(files):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = Image.open(os.path.join(root, f)).convert('RGB')
                images.append(transform(img))
                if len(images) >= max_images:
                    return torch.stack(images)
    return torch.stack(images) if images else torch.empty(0)


def paint_batch(x_128: torch.Tensor, actor: nn.Module, renderer: nn.Module,
                max_step: int, output_steps: list[int]) -> list[torch.Tensor]:
    """Paints at 128×128 with frozen painter, returns canvases at checkpoints."""
    B = x_128.shape[0]
    pw = 128
    device = x_128.device

    canvas = torch.zeros(B, 3, pw, pw, device=device)
    T = torch.ones(B, 1, pw, pw, device=device)
    i_g = torch.linspace(0, 1, pw, device=device).view(-1, 1).repeat(1, pw)
    j_g = torch.linspace(0, 1, pw, device=device).view(1, -1).repeat(pw, 1)
    coord = torch.stack([i_g, j_g], dim=0).unsqueeze(0).expand(B, -1, -1, -1)

    output_set = set(output_steps)
    canvases = []
    stroke_idx = 0

    for step in range(max_step):
        stepnum = T * (step / max_step)
        actor_input = torch.cat([canvas, x_128, stepnum, coord], 1)
        actions = actor(actor_input)
        canvas, res = decode(actions, canvas, renderer, pw)

        for j in range(5):
            stroke_idx += 1
            if stroke_idx in output_set:
                canvases.append(res[j].clone())

    return canvases  # list of (B, 3, 128, 128) tensors


def train(args):
    device = args.device
    in_size = args.in_size  # e.g. 32
    pw = 128  # painter width

    print(f'Training learned resize: {in_size}→{pw}→paint→{pw}→{in_size}')
    print(f'  Data: {args.data_dir}')
    print(f'  Device: {device}')

    # Load data
    print('Loading images...')
    all_images = load_images(args.data_dir, max_images=args.max_images)
    n_total = len(all_images)
    n_val = max(1, int(n_total * 0.1))
    perm = torch.randperm(n_total)
    val_images = all_images[perm[:n_val]]
    train_images = all_images[perm[n_val:]]
    print(f'  {n_total} images → {len(train_images)} train, {n_val} val')

    # Load frozen painter
    print('Loading frozen painter (128×128)...')
    actor = ActorResNet(width=pw).to(device)
    actor.load_state_dict(torch.load(args.actor_path, map_location=device))
    actor.eval()
    for p in actor.parameters():
        p.requires_grad_(False)

    renderer = RendererFCN(width=pw).to(device)
    renderer.load_state_dict(torch.load(args.renderer_path, map_location=device))
    renderer.eval()
    for p in renderer.parameters():
        p.requires_grad_(False)

    # Trainable resize layers
    upscaler = LearnedUpscale(in_size=in_size, out_size=pw).to(device)
    downscaler = LearnedDownscale(in_size=pw, out_size=in_size).to(device)

    n_params = sum(p.numel() for p in upscaler.parameters()) + \
               sum(p.numel() for p in downscaler.parameters())
    print(f'  Trainable params: {n_params:,} (upscaler + downscaler only)')

    optimizer = torch.optim.Adam(
        list(upscaler.parameters()) + list(downscaler.parameters()),
        lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=True
    )

    # Output steps to capture during painting
    output_steps = [int(s) for s in args.output_steps.split(',')]
    print(f'  Paint steps: {output_steps}')

    os.makedirs(args.save_dir, exist_ok=True)
    save_path_up = os.path.join(args.save_dir, 'upscaler.pth')
    save_path_down = os.path.join(args.save_dir, 'downscaler.pth')
    metrics_path = os.path.join(args.save_dir, 'resize_training_metrics.csv')

    metrics_file = open(metrics_path, 'w', newline='')
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow(['epoch', 'train_mse', 'val_mse', 'best_val_mse', 'lr', 'elapsed_sec'])
    print(f'  Metrics: {metrics_path}', flush=True)

    best_val = float('inf')
    patience_counter = 0
    t0 = time.time()

    for epoch in range(1, args.max_epochs + 1):
        # --- Train ---
        upscaler.train()
        downscaler.train()
        train_loss_sum = 0.0
        n_batches = 0

        indices = torch.randperm(len(train_images))
        for start in range(0, len(train_images), args.batch_size):
            batch_idx = indices[start:start + args.batch_size]
            if len(batch_idx) < 2:
                continue
            x = train_images[batch_idx].to(device)  # (B, 3, 32, 32)

            # Forward: upscale → paint (frozen) → downscale
            x_up = upscaler(x)  # (B, 3, 128, 128)

            with torch.no_grad():
                painted_128 = paint_batch(x_up, actor, renderer,
                                          args.max_step, output_steps)

            # Downscale each painted canvas and compute loss vs original
            loss = torch.tensor(0.0, device=device)
            for canvas_128 in painted_128:
                canvas_down = downscaler(canvas_128)  # (B, 3, 32, 32)
                loss = loss + F.mse_loss(canvas_down, x)
            loss = loss / len(painted_128)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            n_batches += 1

        avg_train = train_loss_sum / max(n_batches, 1)

        # --- Validation ---
        upscaler.eval()
        downscaler.eval()
        val_loss_sum = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for start in range(0, len(val_images), args.batch_size):
                x = val_images[start:start + args.batch_size].to(device)
                if len(x) < 2:
                    continue
                x_up = upscaler(x)
                painted_128 = paint_batch(x_up, actor, renderer,
                                          args.max_step, output_steps)
                loss = torch.tensor(0.0, device=device)
                for canvas_128 in painted_128:
                    canvas_down = downscaler(canvas_128)
                    loss = loss + F.mse_loss(canvas_down, x)
                loss = loss / len(painted_128)
                val_loss_sum += loss.item()
                n_val_batches += 1

        avg_val = val_loss_sum / max(n_val_batches, 1)
        scheduler.step(avg_val)

        improved = avg_val < best_val
        if improved:
            best_val = avg_val
            patience_counter = 0
            torch.save(upscaler.state_dict(), save_path_up)
            torch.save(downscaler.state_dict(), save_path_down)
        else:
            patience_counter += 1

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']
        print(f'  Epoch {epoch}/{args.max_epochs}  train_mse={avg_train:.6f}  '
              f'val_mse={avg_val:.6f}  best={best_val:.6f}  '
              f'patience={patience_counter}/{args.patience}  '
              f'lr={lr:.1e}  {elapsed:.0f}s'
              f'  {"★" if improved else ""}', flush=True)

        metrics_writer.writerow([
            epoch, f'{avg_train:.6f}', f'{avg_val:.6f}',
            f'{best_val:.6f}', f'{lr:.1e}', f'{elapsed:.1f}'
        ])
        metrics_file.flush()

        if patience_counter >= args.patience:
            print(f'\n  Early stopping at epoch {epoch}', flush=True)
            break

    metrics_file.close()
    print(f'Training complete.')
    print(f'  Upscaler: {save_path_up}')
    print(f'  Downscaler: {save_path_down}')
    print(f'  Metrics: {metrics_path}')


def main():
    parser = argparse.ArgumentParser(description='Train learned resize layers')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--actor_path', type=str, required=True)
    parser.add_argument('--renderer_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='models/learned_resize')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--in_size', type=int, default=32)
    parser.add_argument('--max_step', type=int, default=40)
    parser.add_argument('--output_steps', type=str, default='50,100,150,200',
                        help='Comma-separated stroke checkpoints to train on')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--max_images', type=int, default=50000)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
