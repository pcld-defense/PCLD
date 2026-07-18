"""Gradient-validity battery: six checks proving the absence of gradient masking.

Robustness claims for a BPDA-style defence are only meaningful if the attack
gradients are real (Athalye et al. 2018; Carlini et al. 2019; Tramer et al.
NeurIPS 2020). This module implements the six-check "R01 gate":

1. **Sign test** — one FGSM step must increase the loss.
2. **Loss-vs-iteration curve** — PGD loss must trend upward over iterations.
3. **Monotonicity** — second-half mean loss >= first-half mean loss.
4. **Finite differences** — analytic directional derivatives must sign-agree
   with central finite differences over random directions.
5. **Epsilon sweep to zero** — robust accuracy must reach ~0 as epsilon grows.
6. **Unbounded epsilon** — an epsilon=255 attack must drive accuracy to ~0.

Each check returns ``{'name': str, 'passed': bool, 'details': dict}``.
``run_gradient_battery`` runs all six, writes ``gradient_battery.json``,
``loss_curve.csv`` and ``eps_sweep.csv`` artifacts, and reports the overall
gate verdict (pass = all six pass).
"""

import csv
import json
import os
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
AttackFn = Callable[[torch.nn.Module, torch.Tensor, torch.Tensor, float],
                    torch.Tensor]


def check_sign(loss_fn: LossFn, x: torch.Tensor, y: torch.Tensor,
               epsilon: float) -> dict:
    """Checks that a single FGSM step along the gradient increases the loss.

    If the gradient points uphill, ``L(x + eps * sign(grad L)) > L(x)`` must
    hold. A flat or decreasing loss after the step is a strong masking signal
    (zero, shattered, or misdirected gradients).

    Args:
        loss_fn: Differentiable loss callable ``loss_fn(x, y) -> scalar``.
        x: Clean input batch of shape (B, 3, H, W) in [0, 1].
        y: Label tensor matching what ``loss_fn`` expects.
        epsilon: L-inf step size in [0, 1].

    Returns:
        Dict with 'name', 'passed', and 'details' holding ``loss_clean`` and
        ``loss_after_fgsm``.
    """
    x_leaf = x.detach().requires_grad_(True)
    loss_clean = loss_fn(x_leaf, y)
    loss_clean.backward()
    with torch.no_grad():
        x_fgsm = torch.clamp(x + epsilon * x_leaf.grad.sign(), 0.0, 1.0)
        loss_after_fgsm = loss_fn(x_fgsm, y).item()
    loss_clean_val = loss_clean.item()
    passed = loss_after_fgsm > loss_clean_val

    print(f'[gradient battery] Sign test: clean_loss={loss_clean_val:.4f}, '
          f'fgsm_loss={loss_after_fgsm:.4f} -> '
          f'{"PASSED" if passed else "FAILED (gradient masking suspected)"}')

    return {
        'name': 'sign_test',
        'passed': bool(passed),
        'details': {
            'loss_clean': loss_clean_val,
            'loss_after_fgsm': loss_after_fgsm,
        },
    }


def check_loss_curve(loss_fn: LossFn, x: torch.Tensor, y: torch.Tensor,
                     epsilon: float, nb_iter: int = 50,
                     output_csv: Optional[str] = None) -> dict:
    """Runs fixed-step PGD and checks that the loss curve trends upward.

    Correct gradients let PGD climb the loss surface, so the final-iteration
    loss should exceed the first-iteration loss. The full per-iteration curve
    is returned for the monotonicity check and optional CSV export.

    Args:
        loss_fn: Differentiable loss callable ``loss_fn(x, y) -> scalar``.
        x: Clean input batch of shape (B, 3, H, W) in [0, 1].
        y: Label tensor matching what ``loss_fn`` expects.
        epsilon: L-inf perturbation budget in [0, 1].
        nb_iter: Number of PGD iterations (step size ``epsilon / nb_iter``).
        output_csv: If provided, saves the per-iteration loss curve to this
            CSV path.

    Returns:
        Dict with 'name', 'passed' (final loss > first loss), and 'details'
        holding ``loss_curve`` (one entry per iteration).
    """
    alpha = epsilon / nb_iter
    x_adv = torch.clamp(x + torch.zeros_like(x).uniform_(-epsilon, epsilon),
                        0.0, 1.0).detach()
    loss_curve: list[float] = []

    for _step in range(nb_iter):
        x_adv = x_adv.requires_grad_(True)
        step_loss = loss_fn(x_adv, y)
        step_loss.backward()
        with torch.no_grad():
            x_adv = x_adv.detach() + alpha * x_adv.grad.sign()
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        loss_curve.append(step_loss.item())

    passed = loss_curve[-1] > loss_curve[0]

    print(f'[gradient battery] Loss curve: first={loss_curve[0]:.4f}, '
          f'last={loss_curve[-1]:.4f} -> '
          f'{"PASSED" if passed else "FAILED (loss did not increase)"}')

    if output_csv is not None:
        os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'loss'])
            for i, l in enumerate(loss_curve):
                writer.writerow([i, l])
        print(f'[gradient battery] Saved loss curve to {output_csv}')

    return {
        'name': 'loss_curve',
        'passed': bool(passed),
        'details': {'loss_curve': loss_curve},
    }


def check_monotonicity(loss_curve: Sequence[float]) -> dict:
    """Checks that the second-half mean PGD loss is >= the first-half mean.

    A loss curve that peaks early and decays afterwards indicates the attack
    is following noise rather than a real gradient signal.

    Args:
        loss_curve: Per-iteration PGD loss values from ``check_loss_curve``.

    Returns:
        Dict with 'name', 'passed', and 'details' holding the two half means.
    """
    mid = len(loss_curve) // 2
    loss_first_half = float(np.mean(loss_curve[:mid])) if mid > 0 else 0.0
    loss_second_half = float(np.mean(loss_curve[mid:]))
    passed = loss_second_half >= loss_first_half

    print(f'[gradient battery] Monotonicity: first-half avg={loss_first_half:.4f}, '
          f'second-half avg={loss_second_half:.4f} -> '
          f'{"PASSED" if passed else "FAILED (gradient masking suspected)"}')

    return {
        'name': 'monotonicity',
        'passed': bool(passed),
        'details': {
            'loss_first_half': loss_first_half,
            'loss_second_half': loss_second_half,
        },
    }


def check_finite_difference(loss_fn: LossFn, x: torch.Tensor, y: torch.Tensor,
                            n_dirs: int = 8, delta: float = 1e-3,
                            agreement_threshold: float = 0.75) -> dict:
    """Compares analytic directional derivatives against finite differences.

    For ``n_dirs`` random unit directions ``v``, compares the analytic
    directional derivative ``<grad L(x), v>`` against the central difference
    ``(L(x + delta*v) - L(x - delta*v)) / (2*delta)``. Passes when the sign
    agreement rate reaches ``agreement_threshold`` — sign agreement rather
    than magnitude, because BPDA gradients are approximate by construction.
    An exactly-zero analytic gradient fails outright (masking by flatness).

    Args:
        loss_fn: Differentiable loss callable ``loss_fn(x, y) -> scalar``.
        x: Clean input batch of shape (B, 3, H, W) in [0, 1].
        y: Label tensor matching what ``loss_fn`` expects.
        n_dirs: Number of random unit directions to probe.
        delta: Central-difference step size.
        agreement_threshold: Minimum sign-agreement rate to pass.

    Returns:
        Dict with 'name', 'passed', and 'details' holding ``sign_agreement``,
        ``cosine`` (cosine between the analytic and finite-difference value
        vectors), the paired values, and a ``zero_gradient`` flag.
    """
    x_leaf = x.detach().requires_grad_(True)
    loss = loss_fn(x_leaf, y)
    loss.backward()
    grad = x_leaf.grad.detach()

    grad_norm = grad.norm().item()
    if grad_norm == 0.0:
        print('[gradient battery] Finite differences: analytic gradient is '
              'exactly zero -> FAILED (gradient masking suspected)')
        return {
            'name': 'finite_difference',
            'passed': False,
            'details': {
                'sign_agreement': 0.0,
                'cosine': 0.0,
                'zero_gradient': True,
                'analytic': [],
                'finite_diff': [],
            },
        }

    analytic_vals: list[float] = []
    fd_vals: list[float] = []
    with torch.no_grad():
        for _ in range(n_dirs):
            v = torch.randn_like(x)
            v = v / v.norm()
            analytic = (grad * v).sum().item()
            loss_plus = loss_fn(torch.clamp(x + delta * v, 0.0, 1.0), y).item()
            loss_minus = loss_fn(torch.clamp(x - delta * v, 0.0, 1.0), y).item()
            fd = (loss_plus - loss_minus) / (2.0 * delta)
            analytic_vals.append(analytic)
            fd_vals.append(fd)

    a = np.asarray(analytic_vals)
    f = np.asarray(fd_vals)
    sign_agreement = float(np.mean(np.sign(a) == np.sign(f)))
    denom = float(np.linalg.norm(a) * np.linalg.norm(f))
    cosine = float(np.dot(a, f) / denom) if denom > 0 else 0.0
    passed = sign_agreement >= agreement_threshold

    print(f'[gradient battery] Finite differences: sign_agreement='
          f'{sign_agreement:.2f} (threshold {agreement_threshold}), '
          f'cosine={cosine:.3f} -> '
          f'{"PASSED" if passed else "FAILED (gradient masking suspected)"}')

    return {
        'name': 'finite_difference',
        'passed': bool(passed),
        'details': {
            'sign_agreement': sign_agreement,
            'cosine': cosine,
            'zero_gradient': False,
            'analytic': analytic_vals,
            'finite_diff': fd_vals,
        },
    }


def _robust_accuracy(model: torch.nn.Module, x_adv: torch.Tensor,
                     y: torch.Tensor) -> float:
    """Computes classification accuracy of ``model`` on adversarial inputs."""
    with torch.no_grad():
        preds = torch.argmax(model(x_adv), dim=1)
    return float((preds[:y.shape[0]] == y).float().mean().item())


def check_eps_to_zero(model: torch.nn.Module, x: torch.Tensor,
                      y: torch.Tensor, attack_callable: AttackFn,
                      epsilons: Sequence[int] = (8, 16, 32, 64),
                      zero_threshold: float = 0.02) -> dict:
    """Checks that robust accuracy decays to ~0 as epsilon increases.

    A defence whose robust accuracy plateaus above zero for arbitrarily large
    perturbation budgets is masking gradients: with an unbounded budget any
    input can be turned into any other input, so accuracy must collapse.

    Args:
        model: The model under evaluation (used for accuracy measurement).
        x: Clean input batch of shape (B, 3, H, W) in [0, 1].
        y: True label tensor of shape (B,).
        attack_callable: ``attack_callable(model, x, y, epsilon) -> x_adv``
            with ``epsilon`` already normalised to [0, 1].
        epsilons: Integer pixel budgets to sweep (divided by 255 internally).
        zero_threshold: Maximum allowed robust accuracy at the largest epsilon.

    Returns:
        Dict with 'name', 'passed', and 'details' holding the ``sweep`` table
        (list of ``{'epsilon', 'robust_accuracy'}`` rows).
    """
    sweep: list[dict] = []
    for eps in epsilons:
        x_adv = attack_callable(model, x, y, eps / 255.0)
        acc = _robust_accuracy(model, x_adv, y)
        sweep.append({'epsilon': int(eps), 'robust_accuracy': acc})
        print(f'[gradient battery] eps={eps}/255: robust accuracy {acc:.4f}')

    final_acc = sweep[-1]['robust_accuracy']
    passed = final_acc <= zero_threshold

    print(f'[gradient battery] Eps-to-zero: accuracy at eps={epsilons[-1]} is '
          f'{final_acc:.4f} (threshold {zero_threshold}) -> '
          f'{"PASSED" if passed else "FAILED (gradient masking suspected)"}')

    return {
        'name': 'eps_to_zero',
        'passed': bool(passed),
        'details': {'sweep': sweep, 'zero_threshold': zero_threshold},
    }


def check_unbounded_eps(model: torch.nn.Module, x: torch.Tensor,
                        y: torch.Tensor, attack_callable: AttackFn,
                        zero_threshold: float = 0.02) -> dict:
    """Checks that an unbounded (epsilon=255) attack drives accuracy to ~0.

    With epsilon=255 (i.e. 1.0 after /255) the attacker can replace the input
    with any image in [0, 1]; any surviving accuracy means the attack — not
    the defence — is failing, i.e. gradients are masked.

    Args:
        model: The model under evaluation (used for accuracy measurement).
        x: Clean input batch of shape (B, 3, H, W) in [0, 1].
        y: True label tensor of shape (B,).
        attack_callable: ``attack_callable(model, x, y, epsilon) -> x_adv``
            with ``epsilon`` already normalised to [0, 1].
        zero_threshold: Maximum allowed robust accuracy at epsilon=255.

    Returns:
        Dict with 'name', 'passed', and 'details' holding
        ``robust_accuracy`` at epsilon=255.
    """
    x_adv = attack_callable(model, x, y, 255 / 255.0)
    acc = _robust_accuracy(model, x_adv, y)
    passed = acc <= zero_threshold

    print(f'[gradient battery] Unbounded eps: accuracy at eps=255 is '
          f'{acc:.4f} (threshold {zero_threshold}) -> '
          f'{"PASSED" if passed else "FAILED (gradient masking suspected)"}')

    return {
        'name': 'unbounded_eps',
        'passed': bool(passed),
        'details': {'robust_accuracy': acc, 'zero_threshold': zero_threshold},
    }


def _collect_samples(loader_or_batch: Union[Iterable, Tuple[torch.Tensor, torch.Tensor]],
                     max_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gathers up to ``max_samples`` (x, y) samples from a loader or batch.

    Args:
        loader_or_batch: Either an ``(x, y)`` tensor tuple or an iterable of
            batches yielding ``(x, y)`` or ``(x, y, paths)`` tuples.
        max_samples: Maximum number of samples to collect.

    Returns:
        Tuple ``(x, y)`` of stacked tensors capped at ``max_samples``.
    """
    if (isinstance(loader_or_batch, (tuple, list))
            and len(loader_or_batch) >= 2
            and torch.is_tensor(loader_or_batch[0])):
        x, y = loader_or_batch[0], loader_or_batch[1]
        return x[:max_samples], y[:max_samples]

    xs, ys, n = [], [], 0
    for batch in loader_or_batch:
        x_b, y_b = batch[0], batch[1]
        xs.append(x_b)
        ys.append(y_b)
        n += x_b.shape[0]
        if n >= max_samples:
            break
    return torch.cat(xs)[:max_samples], torch.cat(ys)[:max_samples]


def _default_attack_callable(args: object) -> AttackFn:
    """Builds an untargeted attack callable from ``attack_batch`` and args.

    Args:
        args: Namespace read for ``attack``, ``attack_nb_iter``, and
            ``attack_norm`` (all optional, with the usual defaults).

    Returns:
        Callable ``(model, x, y, epsilon) -> x_adv`` dispatching through
        ``attack_batch``.
    """
    # Imported lazily: pcld.attacks.attacks pulls cleverhans/autoattack, which
    # keeps this module importable in environments without attack libraries.
    from pcld.attacks.attacks import attack_batch

    attack = getattr(args, 'attack', 'pgd')
    nb_iter = int(getattr(args, 'attack_nb_iter', 10))
    norm = getattr(args, 'attack_norm', 'linf')

    def _attack(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
                epsilon: float) -> torch.Tensor:
        return attack_batch(model, x, attack, epsilon, nb_iter, False, y,
                            norm=norm)

    return _attack


def run_gradient_battery(model: torch.nn.Module, loss_fn: LossFn,
                         loader_or_batch: Union[Iterable, Tuple[torch.Tensor, torch.Tensor]],
                         args: object, device: str, out_dir: str,
                         attack_callable: Optional[AttackFn] = None) -> dict:
    """Runs all six gradient-validity checks and writes the gate artifacts.

    Collects up to ``args.num_samples`` (default 128) samples, runs the six
    checks, prints a PASS/FAIL line per check plus the overall gate verdict,
    and writes ``gradient_battery.json``, ``loss_curve.csv``, and
    ``eps_sweep.csv`` into ``out_dir``. The gate passes only when all six
    checks pass.

    Args:
        model: The model under test (e.g. the PCLD BPDA pipeline).
        loss_fn: Differentiable loss callable ``loss_fn(x, y) -> scalar``
            (differentiable through the model, e.g. via BPDA).
        loader_or_batch: DataLoader (yielding ``(x, y[, paths])`` batches) or
            a single ``(x, y)`` tensor tuple.
        args: Namespace read for ``num_samples``, ``battery_epsilons``,
            ``battery_fd_dirs``, ``battery_fd_delta``, ``attack_nb_iter``,
            and (for the default attack callable) ``attack``/``attack_norm``.
        device: Target device string.
        out_dir: Directory where the JSON/CSV artifacts are written.
        attack_callable: Optional ``(model, x, y, epsilon) -> x_adv`` override;
            when None, one is built from ``attack_batch`` and ``args``.

    Returns:
        Dict with 'checks' (the six per-check result dicts), 'n_passed',
        'n_checks', 'gate_passed', and 'num_samples'.
    """
    os.makedirs(out_dir, exist_ok=True)
    max_samples = getattr(args, 'num_samples', None) or 128
    x, y = _collect_samples(loader_or_batch, max_samples)
    x, y = x.to(device), y.to(device)
    model.eval()

    battery_eps = [int(e) for e in
                   getattr(args, 'battery_epsilons', None) or (8, 16, 32, 64)]
    fd_dirs = int(getattr(args, 'battery_fd_dirs', 8))
    fd_delta = float(getattr(args, 'battery_fd_delta', 1e-3))
    nb_iter = int(getattr(args, 'attack_nb_iter', 10))
    epsilon_real = battery_eps[0] / 255.0

    if attack_callable is None:
        attack_callable = _default_attack_callable(args)

    checks: list[dict] = []
    checks.append(check_sign(loss_fn, x, y, epsilon_real))
    curve_res = check_loss_curve(loss_fn, x, y, epsilon_real, nb_iter=nb_iter,
                                 output_csv=os.path.join(out_dir, 'loss_curve.csv'))
    checks.append(curve_res)
    checks.append(check_monotonicity(curve_res['details']['loss_curve']))
    checks.append(check_finite_difference(loss_fn, x, y, n_dirs=fd_dirs,
                                          delta=fd_delta))
    eps_res = check_eps_to_zero(model, x, y, attack_callable,
                                epsilons=battery_eps)
    checks.append(eps_res)
    unbounded_res = check_unbounded_eps(model, x, y, attack_callable)
    checks.append(unbounded_res)

    # Epsilon sweep table (including the unbounded eps=255 row).
    sweep_csv = os.path.join(out_dir, 'eps_sweep.csv')
    with open(sweep_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epsilon', 'robust_accuracy'])
        for row in eps_res['details']['sweep']:
            writer.writerow([row['epsilon'], row['robust_accuracy']])
        writer.writerow([255, unbounded_res['details']['robust_accuracy']])
    print(f'[gradient battery] Saved epsilon sweep to {sweep_csv}')

    n_passed = sum(1 for c in checks if c['passed'])
    gate_passed = n_passed == len(checks)
    for c in checks:
        print(f"[gradient battery] {c['name']}: "
              f"{'PASS' if c['passed'] else 'FAIL'}")
    print(f"GATE: {'PASS' if gate_passed else 'FAIL'} "
          f"({n_passed}/{len(checks)})")

    summary = {
        'checks': checks,
        'n_passed': n_passed,
        'n_checks': len(checks),
        'gate_passed': gate_passed,
        'num_samples': int(x.shape[0]),
    }
    json_path = os.path.join(out_dir, 'gradient_battery.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[gradient battery] Saved battery results to {json_path}')

    return summary
