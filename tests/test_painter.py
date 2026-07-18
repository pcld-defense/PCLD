"""Tests for the painter's checkpoint arithmetic and determinism.

Covers the regression where ``PainterConsts`` shipped with ``MAX_STEP=40,
DIVIDE=1``: the stroke counter then stopped at 200, so only 3 of the 16 paint
checkpoints fired and the decisioner received a truncated trajectory. The
fixed defaults (80, 5) reach 5200 and fire every default checkpoint.

Runs on CPU with random-initialised painter weights. Skipped automatically if
torch is not installed.
"""

import pytest

torch = pytest.importorskip('torch')

from pcld.painter.painter import ActorResNet, RendererFCN
from pcld.painter.painter_utils import (expected_fired_checkpoints, paint,
                                        paint_images)

_DEFAULT_OUTPUT_EVERY = [50, 100, 200, 300, 400, 500, 600, 700,
                         950, 1200, 1700, 2200, 3200, 4200, 5200]

_SMALL_OUTPUT_EVERY = [5, 10, 14, 50]


def _tiny_painter():
    torch.manual_seed(1234)
    actor = ActorResNet().eval()
    renderer = RendererFCN().eval()
    return actor, renderer


def test_expected_fired_checkpoints_defaults():
    # Fixed defaults (80, 5): counter runs 1..200 then 225..5200 step 25,
    # hitting every default checkpoint.
    assert expected_fired_checkpoints(_DEFAULT_OUTPUT_EVERY,
                                      max_step=80, divide=5) == \
        sorted(_DEFAULT_OUTPUT_EVERY)
    # PainterConsts fallbacks are now (80, 5), so omitting the args is the same.
    assert expected_fired_checkpoints(_DEFAULT_OUTPUT_EVERY) == \
        sorted(_DEFAULT_OUTPUT_EVERY)
    # The old bug (40, 1): Phase 2 disabled, counter stops at 200 — only the
    # first three checkpoints ever fired.
    assert expected_fired_checkpoints(_DEFAULT_OUTPUT_EVERY,
                                      max_step=40, divide=1) == [50, 100, 200]


def test_expected_fired_checkpoints_small():
    # max_step=4 halves to 2 (divide > 1): Phase 1 counts 1..10, then Phase 2
    # adds divide**2 = 4 per stroke for 10 strokes: 14, 18, ..., 50.
    assert expected_fired_checkpoints(_SMALL_OUTPUT_EVERY,
                                      max_step=4, divide=2) == [5, 10, 14, 50]
    # A checkpoint the counter skips over (12 is between 10 and 14) never fires.
    assert expected_fired_checkpoints([12], max_step=4, divide=2) == []


def test_paint_fires_all_checkpoints_small():
    actor, renderer = _tiny_painter()
    x = torch.rand(1, 3, 32, 32)

    canvases = paint(x, _SMALL_OUTPUT_EVERY, 'cpu', actor, renderer,
                     max_step=4, divide=2)
    assert canvases.shape == (1, 4, 3, 32, 32)
    assert canvases.min() >= 0.0 and canvases.max() <= 1.0

    with_orig = paint_images(x, _SMALL_OUTPUT_EVERY, 'cpu', actor, renderer,
                             add_original=True, max_step=4, divide=2)
    assert with_orig.shape == (1, 5, 3, 32, 32)
    assert torch.equal(with_orig[:, -1], x)


def test_paint_determinism_and_batch_invariance():
    actor, renderer = _tiny_painter()
    torch.manual_seed(0)
    x = torch.rand(1, 3, 32, 32)

    first = paint(x.clone(), _SMALL_OUTPUT_EVERY, 'cpu', actor, renderer,
                  max_step=4, divide=2)
    second = paint(x.clone(), _SMALL_OUTPUT_EVERY, 'cpu', actor, renderer,
                   max_step=4, divide=2)
    # Hard requirement: repeated identical calls are bit-identical (eval-mode
    # BN uses running stats; no RNG is consumed during painting).
    assert torch.equal(first, second)

    batched = paint(x.repeat(2, 1, 1, 1), _SMALL_OUTPUT_EVERY, 'cpu', actor,
                    renderer, max_step=4, divide=2)
    assert batched.shape == (2, 4, 3, 32, 32)
    for i in range(2):
        assert torch.equal(batched[i], first[0])
