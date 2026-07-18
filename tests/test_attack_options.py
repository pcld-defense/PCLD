"""Tests for the WS4 attack-strength options.

Covers the straight-through BPDA surrogate (``AllIdentitySurrogate``) and the
EOT gradient averaging in ``pgd_with_multi_step_loss``:

* the surrogate broadcasts across steps and sums upstream gradients back to x,
  both directly and through ``BPDAPainterLayer``;
* ``eot_samples=1`` is bit-identical to the pre-EOT code path;
* ``eot_samples>1`` on a deterministic model reproduces the ``eot_samples=1``
  trajectory exactly (an averaged deterministic gradient has the same sign).

Skipped automatically if torch / cleverhans are not installed.
"""

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


def test_all_identity_surrogate_shape_and_grad():
    from pcld.painter.painter_surrogate import AllIdentitySurrogate

    num_steps = 4
    surrogate = AllIdentitySurrogate(num_steps)
    x = torch.rand(2, 3, 8, 8, requires_grad=True)

    out = surrogate(x)
    assert out.shape == (2, num_steps, 3, 8, 8)

    out.sum().backward()
    # Each of the num_steps identity copies contributes a gradient of ones.
    assert torch.equal(x.grad, torch.full_like(x, float(num_steps)))


def test_all_identity_surrogate_through_bpda_layer():
    from pcld.attacks.pcld_bpda import BPDAPainterLayer
    from pcld.painter.painter_surrogate import AllIdentitySurrogate

    num_steps = 4
    surrogate = AllIdentitySurrogate(num_steps)

    def dummy_painter(x, output_every, device, actor, renderer):
        """Non-differentiable stand-in for paint_images: tiles x across steps."""
        return x.detach().unsqueeze(1).expand(-1, num_steps, -1, -1, -1).contiguous()

    BPDAPainterLayer.reset()
    x = torch.rand(2, 3, 8, 8, requires_grad=True)
    out = BPDAPainterLayer.apply(x, dummy_painter, surrogate,
                                 [1, 2, 3], 'cpu', None, None, None, None)
    assert out.shape == (2, num_steps, 3, 8, 8)

    out.backward(torch.ones_like(out))
    # Straight-through BPDA: input grad is the sum of upstream step grads.
    assert torch.equal(x.grad, torch.full_like(x, float(num_steps)))


def test_eot_one_bit_identical():
    from pcld.attacks.attacks import pgd_with_multi_step_loss
    from pcld.attacks.pcld_bpda import BPDAPainterLayer

    x, y = _inputs()
    eps = 8 / 255.0

    m = _toy_model()
    BPDAPainterLayer.reset()
    torch.manual_seed(2024)
    default = pgd_with_multi_step_loss(model=m, x=x.clone(), epsilon=eps,
                                       nb_iter=10, targeted=False, y=y,
                                       nb_restarts=2, use_apgd=True, norm='linf')

    m = _toy_model()
    BPDAPainterLayer.reset()
    torch.manual_seed(2024)
    explicit = pgd_with_multi_step_loss(model=m, x=x.clone(), epsilon=eps,
                                        nb_iter=10, targeted=False, y=y,
                                        nb_restarts=2, use_apgd=True, norm='linf',
                                        eot_samples=1)

    assert torch.equal(default, explicit)


def test_eot_n_matches_on_deterministic_model():
    from pcld.attacks.attacks import pgd_with_multi_step_loss
    from pcld.attacks.pcld_bpda import BPDAPainterLayer

    x, y = _inputs()
    eps = 8 / 255.0

    m = _toy_model()
    BPDAPainterLayer.reset()
    torch.manual_seed(2024)
    single = pgd_with_multi_step_loss(model=m, x=x.clone(), epsilon=eps,
                                      nb_iter=10, targeted=False, y=y,
                                      nb_restarts=1, use_apgd=False, norm='linf',
                                      eot_samples=1)

    m = _toy_model()
    BPDAPainterLayer.reset()
    torch.manual_seed(2024)
    averaged = pgd_with_multi_step_loss(model=m, x=x.clone(), epsilon=eps,
                                        nb_iter=10, targeted=False, y=y,
                                        nb_restarts=1, use_apgd=False, norm='linf',
                                        eot_samples=3)

    # A deterministic gradient averaged 3x keeps its sign, and PGD only
    # consumes grad.sign(), so the trajectories are identical.
    assert torch.equal(single, averaged)
