"""Tests for the six-check gradient-validity battery (the R01 gate).

A well-behaved differentiable toy model must pass the checks, while a
constant-logit (gradient-masked) model must fail the gate. The epsilon
sweeps run the real PGD implementation through ``attack_batch``, so these
tests exercise the same attack path the experiments use.

Skipped automatically if torch is not installed.
"""

import argparse
import json
import os

import pytest

torch = pytest.importorskip('torch')

from pcld.eval.gradient_battery import (check_eps_to_zero,
                                        check_finite_difference,
                                        run_gradient_battery)


def _toy_model() -> torch.nn.Module:
    """Builds a deterministic linear toy classifier over 3x8x8 inputs."""
    torch.manual_seed(123)
    m = torch.nn.Sequential(torch.nn.Flatten(),
                            torch.nn.Linear(3 * 8 * 8, 5))
    m.eval()
    return m


class _MaskedModel(torch.nn.Module):
    """Constant-logit model: input-connected but with exactly-zero gradients."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns constant logits (class 0 wins) with a zeroed input path."""
        zero_path = x.reshape(x.shape[0], -1).sum(dim=1, keepdim=True) * 0.0
        base = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0], device=x.device)
        return zero_path + base


def _inputs(n: int = 16):
    torch.manual_seed(7)
    x = torch.rand(n, 3, 8, 8)          # [0, 1]
    y = torch.arange(n) % 5
    return x, y


def _loss_fn(model: torch.nn.Module):
    def loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(model(x), y, reduction='sum')
    return loss


def _pgd_attack_callable(nb_iter: int = 10):
    pytest.importorskip('cleverhans')
    from pcld.attacks.attacks import attack_batch

    def attack(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
               epsilon: float) -> torch.Tensor:
        return attack_batch(model, x, 'pgd', epsilon, nb_iter, False, y,
                            norm='linf')
    return attack


def _args(**overrides) -> argparse.Namespace:
    base = dict(num_samples=16, battery_epsilons=[8, 16, 32, 64],
                battery_fd_dirs=8, battery_fd_delta=1e-3,
                attack='pgd', attack_nb_iter=10, attack_norm='linf')
    base.update(overrides)
    return argparse.Namespace(**base)


def test_finite_difference_passes_on_linear_model():
    m = _toy_model()
    x, y = _inputs(4)
    torch.manual_seed(0)
    res = check_finite_difference(_loss_fn(m), x, y, n_dirs=8, delta=1e-3)
    assert res['name'] == 'finite_difference'
    assert res['passed'] is True
    assert res['details']['sign_agreement'] >= 0.75
    assert res['details']['zero_gradient'] is False


def test_finite_difference_fails_on_masked_model():
    m = _MaskedModel().eval()
    x, y = _inputs(4)
    torch.manual_seed(0)
    res = check_finite_difference(_loss_fn(m), x, y, n_dirs=8, delta=1e-3)
    assert res['passed'] is False
    assert res['details']['zero_gradient'] is True


def test_eps_to_zero_with_real_pgd_on_toy_model():
    m = _toy_model()
    x, y = _inputs(16)
    torch.manual_seed(11)
    res = check_eps_to_zero(m, x, y, _pgd_attack_callable(),
                            epsilons=(8, 16, 32, 64))
    sweep = res['details']['sweep']
    assert [row['epsilon'] for row in sweep] == [8, 16, 32, 64]
    # A linear model has exact gradients: a 64/255 PGD must zero accuracy.
    assert res['passed'] is True
    assert sweep[-1]['robust_accuracy'] <= 0.02


def test_masked_model_fails_gate(tmp_path):
    m = _MaskedModel().eval()
    x, y = _inputs(16)
    y = torch.zeros_like(y)  # constant model predicts class 0 -> accuracy 1.0
    torch.manual_seed(11)
    summary = run_gradient_battery(m, _loss_fn(m), (x, y), _args(), 'cpu',
                                   str(tmp_path),
                                   attack_callable=_pgd_attack_callable())
    assert summary['gate_passed'] is False
    by_name = {c['name']: c for c in summary['checks']}
    assert by_name['sign_test']['passed'] is False
    assert by_name['finite_difference']['passed'] is False
    assert by_name['eps_to_zero']['passed'] is False
    assert by_name['unbounded_eps']['passed'] is False


def test_runner_writes_artifacts_and_passes_on_toy_model(tmp_path):
    m = _toy_model()
    x, y = _inputs(16)
    torch.manual_seed(11)
    summary = run_gradient_battery(m, _loss_fn(m), (x, y), _args(), 'cpu',
                                   str(tmp_path),
                                   attack_callable=_pgd_attack_callable())

    for artifact in ('gradient_battery.json', 'loss_curve.csv',
                     'eps_sweep.csv'):
        assert os.path.isfile(os.path.join(str(tmp_path), artifact)), artifact

    with open(os.path.join(str(tmp_path), 'gradient_battery.json')) as f:
        saved = json.load(f)
    assert saved['n_checks'] == 6
    assert saved['gate_passed'] == summary['gate_passed']
    assert len(summary['checks']) == 6
    # An honest differentiable model must clear the gate.
    assert summary['gate_passed'] is True
