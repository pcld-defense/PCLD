"""Attack registry: one uniform interface over FGSM, PGD, and AutoAttack.

Every attack is a callable with the signature::

    fn(model, x, y, *, epsilon, targeted, norm, nb_iter,
       loss_fn, nb_restarts, use_apgd, eot_samples, aa_version) -> torch.Tensor

Only the arguments an attack needs are consumed; the rest are accepted and
ignored so ``attack_batch`` can call any of them uniformly.  The numerical
behaviour of each attack is identical to the original ``attack_batch`` branch;
this module only reorganises the dispatch so new attacks can be added with a
decorator instead of another ``elif``.
"""

import numpy as np
import torch
from autoattack import AutoAttack
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method

from pcld.attacks.pcld_bpda import BPDAPainterLayer
from pcld.utils.registry import Registry

ATTACKS: Registry = Registry('attack')


def _norm_val(norm: str):
    """Maps the string norm to the numeric norm cleverhans/AutoAttack expect."""
    return np.inf if norm == 'linf' else 2


@ATTACKS.register('fgsm')
def _fgsm(model, x, y, *, epsilon, targeted, norm='linf', **_) -> torch.Tensor:
    """Single-step FGSM via CleverHans (unchanged from the legacy path)."""
    return fast_gradient_method(
        model_fn=model, x=x, eps=epsilon, norm=_norm_val(norm),
        y=y, targeted=targeted, clip_min=0, clip_max=1)


@ATTACKS.register('pgd')
def _pgd(model, x, y, *, epsilon, targeted, norm='linf', nb_iter=10,
         loss_fn=None, nb_restarts=1, use_apgd=False, eot_samples=1,
         **_) -> torch.Tensor:
    """Custom PGD loop (APGD schedule, restarts, multi-step loss, EOT)."""
    # Import here to avoid a circular import (attacks.py imports this module).
    from pcld.attacks.attacks import pgd_with_multi_step_loss

    BPDAPainterLayer.reset()
    return pgd_with_multi_step_loss(
        model=model, x=x, epsilon=epsilon, nb_iter=nb_iter,
        targeted=targeted, y=y, loss_fn=loss_fn,
        nb_restarts=nb_restarts, use_apgd=use_apgd, norm=norm,
        eot_samples=eot_samples)


@ATTACKS.register('aa')
def _autoattack(model, x, y, *, epsilon, norm='linf', targeted=False,
                aa_version='standard', **_) -> torch.Tensor:
    """AutoAttack ensemble ('standard' or 'rand'), optionally targeted."""
    aa_norm = 'Linf' if norm == 'linf' else 'L2'
    adv_attack = AutoAttack(model, norm=aa_norm, eps=epsilon, version=aa_version)
    if targeted and aa_version == 'standard':
        # Restrict the 'standard' ensemble to its targeted subset. 'rand'
        # defines its own EOT-based attacks_to_run (apgd-ce, apgd-dlr) with no
        # targeted variants, so it is left untouched. The constructor rejects a
        # non-empty attacks_to_run for named versions, so override after init.
        adv_attack.attacks_to_run = ['apgd-t', 'fab-t']
    return adv_attack.run_standard_evaluation_individual(x, y)
