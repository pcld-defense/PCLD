"""Learned up/downscale layers for resolution-adaptive painting.

Instead of bilinear interpolation (fixed, non-learnable), these layers
use transposed/strided convolutions to resize images before and after
painting.  The learned resize can potentially filter adversarial
perturbations during upscale while preserving essential features.

The existing 128×128 painter (actor + renderer) is kept as-is.  Only
the resize layers are trained.

Usage:
    upscaler = LearnedUpscale(in_size=32, out_size=128)
    downscaler = LearnedDownscale(in_size=128, out_size=32)

    # In the painting pipeline:
    x_128 = upscaler(x_32)          # learned 32→128
    canvases_128 = painter(x_128)   # existing 128×128 painter
    canvases_32 = downscaler(canvases_128)  # learned 128→32
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedUpscale(nn.Module):
    """Learned upscaling from input resolution to painter resolution.

    Uses a small CNN with PixelShuffle for clean upscaling.  Initialized
    to approximate bilinear interpolation so the painter works out of the
    box before fine-tuning.

    Args:
        in_size: Input spatial resolution (e.g. 32 for CIFAR-10).
        out_size: Output spatial resolution (e.g. 128 for the painter).
        channels: Number of image channels (default 3 for RGB).
    """

    def __init__(self, in_size: int = 32, out_size: int = 128,
                 channels: int = 3) -> None:
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.scale_factor = out_size // in_size  # e.g. 4 for 32→128

        # Two-stage upscale with residual learning
        # Stage 1: conv + PixelShuffle for 2× upscale
        # Stage 2: conv + PixelShuffle for remaining factor
        if self.scale_factor == 4:
            self.net = nn.Sequential(
                nn.Conv2d(channels, 32, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, channels * 4, 3, 1, 1),  # 4× channels for PixelShuffle(2)
                nn.PixelShuffle(2),  # 2× spatial
                nn.Conv2d(channels, 32, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, channels * 4, 3, 1, 1),
                nn.PixelShuffle(2),  # another 2× = total 4×
            )
        else:
            # Generic: use interpolate + learned refinement
            self.net = nn.Sequential(
                nn.Conv2d(channels, 32, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, channels, 3, 1, 1),
            )
            self._generic = True

        # Skip connection: bilinear upscale (so model starts near identity)
        self._has_generic = not (self.scale_factor == 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upscales input from in_size to out_size.

        Args:
            x: Image batch (B, 3, in_size, in_size) in [0, 1].

        Returns:
            Upscaled image (B, 3, out_size, out_size) in [0, 1].
        """
        # Bilinear baseline (skip connection)
        bilinear = F.interpolate(x, size=(self.out_size, self.out_size),
                                 mode='bilinear', align_corners=False)

        if self._has_generic:
            # Generic path: interpolate first, then refine
            refined = self.net(bilinear)
            return (bilinear + refined).clamp(0, 1)
        else:
            # PixelShuffle path: learned upscale + residual from bilinear
            learned = self.net(x)
            return (bilinear + learned).clamp(0, 1)


class LearnedDownscale(nn.Module):
    """Learned downscaling from painter resolution to output resolution.

    Uses strided convolutions for clean downscaling.  Initialized to
    approximate bilinear interpolation.

    Args:
        in_size: Input spatial resolution (e.g. 128 from the painter).
        out_size: Output spatial resolution (e.g. 32 for CIFAR-10).
        channels: Number of image channels (default 3 for RGB).
    """

    def __init__(self, in_size: int = 128, out_size: int = 32,
                 channels: int = 3) -> None:
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.scale_factor = in_size // out_size  # e.g. 4 for 128→32

        if self.scale_factor == 4:
            self.net = nn.Sequential(
                nn.Conv2d(channels, 32, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 4, 2, 1),  # 2× downsample
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, channels, 4, 2, 1),  # another 2× = total 4×
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(channels, 32, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, channels, 3, 1, 1),
            )
            self._generic = True

        self._has_generic = not (self.scale_factor == 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downscales input from in_size to out_size.

        Args:
            x: Image batch (B, 3, in_size, in_size) in [0, 1].

        Returns:
            Downscaled image (B, 3, out_size, out_size) in [0, 1].
        """
        bilinear = F.interpolate(x, size=(self.out_size, self.out_size),
                                 mode='bilinear', align_corners=False)

        if self._has_generic:
            refined = self.net(bilinear)
            return (bilinear + refined).clamp(0, 1)
        else:
            learned = self.net(x)
            return (bilinear + learned).clamp(0, 1)


class PainterWithLearnedResize(nn.Module):
    """Wraps the existing 128×128 painter with learned up/downscale.

    The full pipeline:
        input (32×32) → LearnedUpscale → paint at 128×128 → LearnedDownscale → output (32×32)

    Only the upscale/downscale layers are trainable.  The painter
    (actor + renderer) is frozen.

    Args:
        upscaler: LearnedUpscale module.
        downscaler: LearnedDownscale module.
        actor: Pretrained ActorResNet (frozen).
        renderer: Pretrained RendererFCN (frozen).
        max_step: Number of painting steps.
        output_every: Stroke checkpoints to capture.
    """

    def __init__(self, upscaler: LearnedUpscale, downscaler: LearnedDownscale,
                 actor: nn.Module, renderer: nn.Module,
                 max_step: int = 40, output_every: list[int] = None) -> None:
        super().__init__()
        self.upscaler = upscaler
        self.downscaler = downscaler
        self.actor = actor
        self.renderer = renderer
        self.max_step = max_step
        self.output_every = set(output_every or [])

        # Freeze painter
        for p in self.actor.parameters():
            p.requires_grad_(False)
        for p in self.renderer.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Paints input images through the learned-resize pipeline.

        Args:
            x: Input batch (B, 3, H_in, W_in) in [0, 1].

        Returns:
            Painted canvases (B, Steps, 3, H_in, W_in) in [0, 1].
        """
        from painter.painter_utils import decode

        B = x.shape[0]
        pw = 128  # painter working resolution

        # Learned upscale
        x_up = self.upscaler(x)  # (B, 3, 128, 128)

        # Paint at 128×128 (frozen painter)
        canvas = torch.zeros(B, 3, pw, pw, device=x.device)
        T = torch.ones(B, 1, pw, pw, device=x.device)
        i_g = torch.linspace(0, 1, pw, device=x.device).view(-1, 1).repeat(1, pw)
        j_g = torch.linspace(0, 1, pw, device=x.device).view(1, -1).repeat(pw, 1)
        coord = torch.stack([i_g, j_g], dim=0).unsqueeze(0).expand(B, -1, -1, -1)

        stroke_idx = 0
        output_canvases = []

        for step in range(self.max_step):
            stepnum = T * (step / self.max_step)
            actor_input = torch.cat([canvas, x_up, stepnum, coord], 1)

            with torch.no_grad():
                actions = self.actor(actor_input)
                canvas, res = decode(actions, canvas, self.renderer, pw)

            for j in range(5):
                stroke_idx += 1
                if stroke_idx in self.output_every:
                    # Learned downscale
                    canvas_down = self.downscaler(res[j])
                    output_canvases.append(canvas_down)

        if not output_canvases:
            return torch.empty(0, device=x.device)

        return torch.stack(output_canvases, dim=1)  # (B, Steps, 3, H, W)
