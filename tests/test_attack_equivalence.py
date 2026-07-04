"""Proves the registry-based attack dispatch is a pure pass-through.

``attack_batch`` was refactored from an if/elif chain to a registry lookup.
These tests assert the dispatched result is bit-identical to calling the
underlying attack primitive directly (i.e. the pre-refactor code path), so the
refactor is numerically inert.

Skipped automatically if torch / cleverhans are not installed.
"""

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('cleverhans')


def _toy_model():
    torch.manual_seed(123)
    m = torch.nn.Sequential(torch.nn.Flatten(),
                            torch.nn.Linear(3 * 8 * 8, 5))
    m.eval()
    return m


def _inputs():
    torch.manual_seed(7)
    x = torch.rand(4, 3, 8, 8)          # [0, 1]
    y = torch.tensor([0, 1, 2, 3])
    return x, y


def test_fgsm_dispatch_matches_direct_call():
    from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
    from pcld.attacks.attacks import attack_batch

    x, y = _inputs()
    eps = 8 / 255.0

    m = _toy_model()
    direct = fast_gradient_method(model_fn=m, x=x.clone(), eps=eps, norm=np.inf,
                                  y=y, targeted=False, clip_min=0, clip_max=1)
    m = _toy_model()
    dispatched = attack_batch(m, x.clone(), 'fgsm', eps, 1, False, y, norm='linf')

    assert torch.equal(direct, dispatched)


def test_pgd_dispatch_matches_direct_call():
    from pcld.attacks.attacks import attack_batch, pgd_with_multi_step_loss
    from pcld.attacks.pcld_bpda import BPDAPainterLayer

    x, y = _inputs()
    eps = 8 / 255.0

    m = _toy_model()
    BPDAPainterLayer.reset()
    torch.manual_seed(2024)
    direct = pgd_with_multi_step_loss(model=m, x=x.clone(), epsilon=eps, nb_iter=10,
                                      targeted=False, y=y, nb_restarts=2,
                                      use_apgd=True, norm='linf')
    m = _toy_model()
    torch.manual_seed(2024)
    dispatched = attack_batch(m, x.clone(), 'pgd', eps, 10, False, y, norm='linf',
                              nb_restarts=2, use_apgd=True)

    assert torch.equal(direct, dispatched)
