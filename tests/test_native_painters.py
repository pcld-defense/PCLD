"""Tests for native-resolution painter architectures.

Verifies that ActorResNet and RendererFCN work correctly at 32×32 (CIFAR),
128×128 (legacy), and 224×224 (ImageNet) resolutions, and that the painting
pipeline produces the expected output shapes without any image resizing when
operating at native resolution.

Run with:  python -m pytest tests/test_native_painters.py -v
"""

import time
from unittest.mock import patch

import pytest
import torch

from painter.painter import ActorResNet, RendererFCN
from painter.painter_config import (
    CIFAR10_PAINTER, CIFAR100_PAINTER, IMAGENET_PAINTER, LEGACY_PAINTER,
    PainterConfig, get_painter_config,
)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestPainterConfig:
    def test_cifar10_config(self):
        cfg = get_painter_config('cifar10')
        assert cfg.width == 32
        assert cfg.divide == 1
        assert cfg.max_step == 40

    def test_cifar100_config(self):
        cfg = get_painter_config('cifar100')
        assert cfg.width == 32
        assert cfg.divide == 1

    def test_imagenet_config(self):
        cfg = get_painter_config('imagenet')
        assert cfg.width == 224
        assert cfg.divide == 5
        assert cfg.max_step == 80

    def test_unknown_falls_back_to_legacy(self):
        cfg = get_painter_config('unknown_dataset')
        assert cfg.width == LEGACY_PAINTER.width

    def test_width_must_be_divisible_by_8(self):
        with pytest.raises(ValueError, match="divisible by 8"):
            PainterConfig(width=33, divide=1, max_step=40)

    def test_width_224_is_valid(self):
        cfg = PainterConfig(width=224, divide=5, max_step=80)
        assert cfg.width == 224


# ---------------------------------------------------------------------------
# ActorResNet tests
# ---------------------------------------------------------------------------

class TestActorResNet:
    @pytest.mark.parametrize("width", [32, 64, 128, 224])
    def test_output_shape(self, width):
        actor = ActorResNet(width=width)
        x = torch.randn(2, 9, width, width)
        out = actor(x)
        assert out.shape == (2, 65)

    def test_output_in_0_1(self):
        actor = ActorResNet(width=32)
        x = torch.randn(2, 9, 32, 32)
        out = actor(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_weight_compat_128_to_32(self):
        """Pretrained 128 weights load into a 32-width actor (same layers)."""
        actor128 = ActorResNet(width=128)
        state = actor128.state_dict()
        actor32 = ActorResNet(width=32)
        actor32.load_state_dict(state)  # should not raise

    def test_weight_compat_128_to_224(self):
        """Pretrained 128 weights load into a 224-width actor."""
        actor128 = ActorResNet(width=128)
        state = actor128.state_dict()
        actor224 = ActorResNet(width=224)
        actor224.load_state_dict(state)


# ---------------------------------------------------------------------------
# RendererFCN tests
# ---------------------------------------------------------------------------

class TestRendererFCN:
    @pytest.mark.parametrize("width", [32, 64, 128, 224])
    def test_output_shape(self, width):
        renderer = RendererFCN(width=width)
        x = torch.randn(2, 10)
        out = renderer(x)
        assert out.shape == (2, width, width)

    def test_output_in_0_1(self):
        renderer = RendererFCN(width=32)
        x = torch.randn(2, 10)
        out = renderer(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_width_not_divisible_by_8_raises(self):
        with pytest.raises(ValueError):
            RendererFCN(width=33)

    def test_base_spatial_computation(self):
        r32 = RendererFCN(width=32)
        assert r32.base_spatial == 4
        r128 = RendererFCN(width=128)
        assert r128.base_spatial == 16
        r224 = RendererFCN(width=224)
        assert r224.base_spatial == 28

    def test_128_backward_compat(self):
        """Width=128 renderer has same fc4 size as the original (4096)."""
        r = RendererFCN(width=128)
        assert r.fc4.out_features == 4096


# ---------------------------------------------------------------------------
# Decode + stroke rendering tests
# ---------------------------------------------------------------------------

class TestDecode:
    @pytest.mark.parametrize("width", [32, 128])
    def test_decode_shape(self, width):
        from painter.painter_utils import decode
        renderer = RendererFCN(width=width)
        canvas = torch.zeros(2, 3, width, width)
        actions = torch.rand(2, 65)
        final, intermediates = decode(actions, canvas, renderer, width)
        assert final.shape == (2, 3, width, width)
        assert len(intermediates) == 5
        assert intermediates[0].shape == (2, 3, width, width)


# ---------------------------------------------------------------------------
# Speed benchmarks
# ---------------------------------------------------------------------------

class TestSpeed:
    def _time_actor(self, width, n_iters=50):
        actor = ActorResNet(width=width).eval()
        x = torch.randn(1, 9, width, width)
        # Warmup
        for _ in range(5):
            actor(x)
        t0 = time.time()
        with torch.inference_mode():
            for _ in range(n_iters):
                actor(x)
        return (time.time() - t0) / n_iters * 1000  # ms

    def test_cifar_faster_than_legacy(self):
        """32×32 actor should be significantly faster than 128×128."""
        ms_32 = self._time_actor(32)
        ms_128 = self._time_actor(128)
        speedup = ms_128 / ms_32
        print(f"\n  Actor 32×32: {ms_32:.1f}ms, 128×128: {ms_128:.1f}ms, speedup: {speedup:.1f}×")
        assert speedup > 2.0, f"Expected >2× speedup, got {speedup:.1f}×"


# ---------------------------------------------------------------------------
# Stroke counting
# ---------------------------------------------------------------------------

class TestStrokeCounting:
    def test_cifar_max_strokes(self):
        """With CIFAR config (divide=1, max_step=40), max strokes = 200."""
        cfg = CIFAR10_PAINTER
        max_strokes = cfg.max_step * 5  # 5 sub-strokes per step, no Phase 2
        assert max_strokes == 200

    def test_imagenet_max_strokes(self):
        """With ImageNet config (divide=5, max_step=80), max strokes = 5200."""
        cfg = IMAGENET_PAINTER
        phase1_strokes = (cfg.max_step // 2) * 5  # 200
        phase2_strokes = (cfg.max_step // 2) * 5 * (cfg.divide ** 2)  # 5000
        assert phase1_strokes + phase2_strokes == 5200
