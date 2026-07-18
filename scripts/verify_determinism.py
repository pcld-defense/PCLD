"""Verify that the painter is deterministic (R00 paper artifact).

Paints a fixed input image several times — both as a single image and tiled
into a batch of 4 — and asserts that every repeat produces bit-identical
canvases. The single-vs-batch comparison is reported as informational only
(batched kernels may legally differ); the per-layout repeat delta is the hard
assertion.

    # CPU/CI mode (random-initialised painter weights):
    python scripts/verify_determinism.py --random-weights --repeats 2 \
        --max-step 4 --divide 2 --image-size 32 --out determinism.json

    # Real pretrained weights (paths from .env):
    python scripts/verify_determinism.py --repeats 5

Writes a JSON report (sha256 of the canvas bytes per layout, max abs deltas,
torch version, device, config) and exits non-zero if any repeat delta != 0.
"""

import argparse
import hashlib
import json
import sys

import torch

from pcld.painter.painter import ActorResNet, RendererFCN
from pcld.painter.painter_utils import (expected_fired_checkpoints,
                                        load_painter, paint_images)
from pcld.utils.consts import ACTOR_WEIGHTS_PATH, RENDERER_WEIGHTS_PATH

_DEFAULT_OUTPUT_EVERY = [50, 100, 200, 300, 400, 500, 600, 700,
                         950, 1200, 1700, 2200, 3200, 4200, 5200]


def build_painter(random_weights: bool,
                  device: str) -> tuple[ActorResNet, RendererFCN]:
    """Builds the actor and renderer, either pretrained or random-initialised.

    Args:
        random_weights: If True, construct freshly (seeded) random-initialised
            models in eval mode instead of loading checkpoints — suitable for
            CPU/CI runs where the pretrained weights are unavailable.
        device: Target device string (e.g. 'cpu' or 'cuda').

    Returns:
        Tuple of (actor, renderer), both in eval mode on `device`.
    """
    if random_weights:
        torch.manual_seed(1234)
        actor = ActorResNet().to(device).eval()
        renderer = RendererFCN().to(device).eval()
        return actor, renderer
    return load_painter(ACTOR_WEIGHTS_PATH, RENDERER_WEIGHTS_PATH, device)


def repeat_paint(img: torch.Tensor, output_every: list[int], device: str,
                 actor: ActorResNet, renderer: RendererFCN, repeats: int,
                 max_step: int | None,
                 divide: int | None) -> list[torch.Tensor]:
    """Paints the same input `repeats` times and collects the canvases.

    Args:
        img: Input image batch of shape (B, 3, H, W) in [0, 1].
        output_every: Stroke-count checkpoints for canvas snapshots.
        device: Target device string.
        actor: ActorResNet stroke-parameter predictor.
        renderer: RendererFCN stroke renderer.
        repeats: Number of independent paint calls.
        max_step: Painter step budget (None uses the painter default).
        divide: Phase-2 patch-grid side length (None uses the default).

    Returns:
        List of `repeats` canvas tensors of shape (B, Steps+1, 3, H, W).
    """
    return [paint_images(img.clone(), output_every, device, actor, renderer,
                         add_original=True, max_step=max_step, divide=divide)
            for _ in range(repeats)]


def max_repeat_delta(canvases: list[torch.Tensor]) -> float:
    """Computes the max abs difference between the first repeat and the rest.

    Args:
        canvases: Canvas tensors from repeated identical paint calls.

    Returns:
        Maximum absolute elementwise delta across all repeats (0.0 means
        bit-identical repeats).
    """
    ref = canvases[0]
    delta = 0.0
    for c in canvases[1:]:
        delta = max(delta, (c - ref).abs().max().item())
    return delta


def canvas_sha256(canvas: torch.Tensor) -> str:
    """Hashes a canvas tensor's raw bytes.

    Args:
        canvas: Canvas tensor of any shape.

    Returns:
        Hex sha256 digest of the tensor's contiguous CPU byte representation.
    """
    return hashlib.sha256(
        canvas.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main() -> int:
    """Runs the painter determinism verification and writes the JSON report.

    Returns:
        0 when every per-layout repeat delta is exactly 0, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=5,
                        help='number of identical paint calls per layout')
    parser.add_argument('--out', type=str, default='determinism.json',
                        help='path of the JSON report to write')
    parser.add_argument('--random-weights', action='store_true',
                        help='random-init the painter (CPU/CI mode) instead '
                             'of loading pretrained weights')
    parser.add_argument('--max-step', type=int, default=None,
                        help='painter step budget (default: painter default)')
    parser.add_argument('--divide', type=int, default=None,
                        help='painter patch-grid side (default: painter default)')
    parser.add_argument('--image-size', type=int, default=32,
                        help='side length of the test image (default 32, '
                             'intended for --random-weights mode)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    actor, renderer = build_painter(args.random_weights, device)

    # Seed only the input creation; painting itself must not need seeding.
    torch.manual_seed(0)
    img = torch.rand(1, 3, args.image_size, args.image_size)

    output_every = _DEFAULT_OUTPUT_EVERY
    fired = expected_fired_checkpoints(output_every, args.max_step, args.divide)
    print(f'device={device}  fired checkpoints={fired}')

    single = repeat_paint(img, output_every, device, actor, renderer,
                          args.repeats, args.max_step, args.divide)
    batch = repeat_paint(img.repeat(4, 1, 1, 1), output_every, device, actor,
                         renderer, args.repeats, args.max_step, args.divide)

    single_delta = max_repeat_delta(single)
    batch_delta = max_repeat_delta(batch)
    # Informational only: batched kernels are allowed to differ from the
    # single-image layout; per-layout repeatability is the hard requirement.
    cross_delta = (batch[0] - single[0].expand_as(batch[0])).abs().max().item()

    report = {
        'torch_version': torch.__version__,
        'device': device,
        'config': {
            'repeats': args.repeats,
            'random_weights': args.random_weights,
            'max_step': args.max_step,
            'divide': args.divide,
            'image_size': args.image_size,
            'output_every': output_every,
            'fired_checkpoints': fired,
        },
        'sha256': {
            'single': canvas_sha256(single[0]),
            'batch': canvas_sha256(batch[0]),
        },
        'max_abs_delta': {
            'single_repeats': single_delta,
            'batch_repeats': batch_delta,
            'single_vs_batch_informational': cross_delta,
        },
    }
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report['max_abs_delta'], indent=2))
    print(f'report written to {args.out}')

    if single_delta != 0.0 or batch_delta != 0.0:
        print('FAIL: painter is not repeat-deterministic')
        return 1
    print('OK: painter output is bit-identical across repeats')
    return 0


if __name__ == '__main__':
    sys.exit(main())
