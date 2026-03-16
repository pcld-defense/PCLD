import os
import random
import time
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from autoattack import AutoAttack
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method

from model.pcld_bpda import BPDAPainterLayer


def pgd_with_multi_step_loss(
        model: torch.nn.Module,
        x: torch.Tensor,
        epsilon: float,
        nb_iter: int,
        targeted: bool,
        y: torch.Tensor,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        nb_restarts: int = 1,
        use_apgd: bool = False,
        norm: str = 'linf',
) -> torch.Tensor:
    """PGD attack with optional APGD step schedule, random restarts, and custom loss.

    Replaces the CleverHans PGD implementation for the ``'pgd'`` attack path.
    Key improvements over vanilla PGD:

    * **Custom loss function** — accepts any ``loss_fn(x_adv, y) -> scalar``
      callable, enabling multi-step intermediate losses for PCLD without
      changing model forward signatures.
    * **APGD step schedule** — when ``use_apgd=True``, halves the step size
      whenever the loss does not improve for ``checkfreq = max(nb_iter//10, 10)``
      consecutive iterations, matching Croce & Hein (ICML 2020).
    * **Random restarts** — when ``nb_restarts > 1``, runs PGD from independent
      random starting points and keeps the adversarial example with the highest
      loss (evaluated with ``model.eval()``).
    * **State reset** — calls ``BPDAPainterLayer.reset()`` before every restart
      and at the start of each new batch to prevent stale canvas contamination.

    Args:
        model: The model to attack; must accept ``(B, 3, H, W)`` input.
        x: Clean input batch of shape ``(B, 3, H, W)`` in ``[0, 1]``.
        epsilon: Perturbation budget in ``[0, 1]`` (already divided by 255).
        nb_iter: Number of PGD iterations per restart.
        targeted: If True, minimises the loss toward the target; if False,
            maximises the loss away from the true class.
        y: Label tensor of shape ``(B,)``; target classes for targeted attacks
            or true classes for untargeted attacks.
        loss_fn: Optional callable ``loss_fn(x_adv, y) -> scalar``.  When None,
            falls back to standard cross-entropy via ``model(x_adv)`` logits.
        nb_restarts: Number of random restarts. The adversarial example with
            the highest untargeted loss (or lowest targeted loss) is returned.
        use_apgd: If True, applies the APGD adaptive step-size schedule.
        norm: Perturbation norm; only ``'linf'`` is currently supported.

    Returns:
        Best adversarial example batch of shape ``(B, 3, H, W)`` in ``[0, 1]``.
    """
    model.eval()

    # Initial step size: 2 * epsilon / nb_iter (Croce & Hein 2020 convention).
    alpha_init = 2.0 * epsilon / nb_iter
    sign = -1.0 if targeted else 1.0

    best_x_adv = x.clone()
    best_loss = torch.full((x.shape[0],), -float('inf'), device=x.device)

    for _restart in range(nb_restarts):
        # Reset BPDA state so stale canvases from the previous restart don't
        # contaminate this restart's first forward pass.
        BPDAPainterLayer.reset()

        # Random initialisation within the ε-ball.
        delta = torch.zeros_like(x).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x + delta, 0.0, 1.0).detach()

        alpha = alpha_init
        checkfreq = max(nb_iter // 10, 10)
        no_improve_count = 0
        prev_best_loss_val = -float('inf')

        for step in range(nb_iter):
            x_adv = x_adv.requires_grad_(True)

            if loss_fn is not None:
                loss = loss_fn(x_adv, y)
            else:
                logits = model(x_adv)
                loss = F.cross_entropy(logits, y, reduction='sum')

            # Maximise loss for untargeted; minimise for targeted.
            (sign * loss).backward()

            with torch.no_grad():
                grad = x_adv.grad.data
                x_adv = x_adv.detach() + alpha * grad.sign()
                # Project back into ε-ball and image range.
                x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)

            # APGD step-size adaptation.
            if use_apgd and (step + 1) % checkfreq == 0:
                current_loss = loss.item()
                if current_loss <= prev_best_loss_val:
                    no_improve_count += 1
                    alpha = max(alpha / 2.0, epsilon / nb_iter)
                else:
                    no_improve_count = 0
                    prev_best_loss_val = current_loss

        # Evaluate this restart's final adversarial example to pick the best.
        # Always reduce to per-IMAGE loss (x.shape[0] == B) so that the
        # comparison and where() call work correctly even when y has been
        # repeated to (B*Steps,) for PCL's paints_inference path.
        B = x.shape[0]
        with torch.no_grad():
            if loss_fn is not None:
                raw_loss = loss_fn(x_adv, y)
                restart_loss = raw_loss.expand(B) if raw_loss.dim() == 0 else raw_loss[:B]
            else:
                logits = model(x_adv)
                per_sample = F.cross_entropy(logits, y, reduction='none')
                if per_sample.shape[0] > B:
                    # PCL: (B*Steps,) → mean over steps → (B,)
                    steps = per_sample.shape[0] // B
                    restart_loss = per_sample.view(B, steps).mean(dim=1)
                else:
                    restart_loss = per_sample  # (B,)

        # For targeted attacks the loss is minimised, so invert for comparison.
        cmp_loss = restart_loss if not targeted else -restart_loss

        improved = cmp_loss > best_loss
        best_x_adv = torch.where(
            improved.view(-1, 1, 1, 1).expand_as(x_adv),
            x_adv, best_x_adv)
        best_loss = torch.where(improved, cmp_loss, best_loss)

    return best_x_adv.detach()


def attack_batch(model: torch.nn.Module, x: torch.Tensor, attack: str,
                 epsilon: float, attack_nb_iter: int, targeted: bool,
                 y_classes_targeted: torch.Tensor,
                 norm: str = 'linf',
                 loss_fn: Optional[Callable] = None,
                 nb_restarts: int = 1,
                 use_apgd: bool = False) -> torch.Tensor:
    """Generates adversarial examples for a single batch using the chosen attack.

    When epsilon is 0 the input is returned unchanged. Supported attacks:
    'fgsm' (single-step via CleverHans), 'pgd' (multi-step, custom loop),
    and 'aa' (AutoAttack).

    Args:
        model: The model to attack; must accept (B, 3, H, W) input.
        x: Clean input batch of shape (B, 3, H, W) in [0, 1].
        attack: Attack name; one of 'fgsm', 'pgd', or 'aa'.
        epsilon: Perturbation budget in [0, 1] (already normalised by /255).
        attack_nb_iter: Number of PGD iterations (ignored for FGSM/AA).
        targeted: If True the attack minimises the loss toward the target class;
            if False it maximises the loss away from the true class.
        y_classes_targeted: Label tensor of shape (B,) with target class indices
            for targeted attacks or true class indices for untargeted attacks.
        norm: Perturbation norm; 'linf' or 'l2'.
        loss_fn: Optional custom loss callable passed to ``pgd_with_multi_step_loss``.
            Only used when ``attack='pgd'``. When None, cross-entropy on the
            model's direct output is used.
        nb_restarts: Number of random restarts for PGD. Only used when
            ``attack='pgd'``.
        use_apgd: If True, apply the APGD step-size adaptation schedule for
            PGD. Only used when ``attack='pgd'``.

    Returns:
        Adversarial example batch of shape (B, 3, H, W) in [0, 1].
    """
    if epsilon == 0:
        return x

    norm_val = np.inf if norm == 'linf' else 2

    if attack == 'fgsm':
        x_adv = fast_gradient_method(model_fn=model,
                                     x=x,
                                     eps=epsilon,
                                     norm=norm_val,
                                     y=y_classes_targeted,
                                     targeted=targeted,
                                     clip_min=0,
                                     clip_max=1)
    elif attack == 'pgd':
        # Reset BPDA state at the start of each new batch.
        BPDAPainterLayer.reset()
        x_adv = pgd_with_multi_step_loss(
            model=model,
            x=x,
            epsilon=epsilon,
            nb_iter=attack_nb_iter,
            targeted=targeted,
            y=y_classes_targeted,
            loss_fn=loss_fn,
            nb_restarts=nb_restarts,
            use_apgd=use_apgd,
            norm=norm,
        )
    elif attack == 'aa':
        aa_norm = 'Linf' if norm == 'linf' else 'L2'
        adv_attack = AutoAttack(model, norm=aa_norm, eps=epsilon, version='standard')
        x_adv = adv_attack.run_standard_evaluation_individual(x, y_classes_targeted)

    return x_adv


def _make_results_dict() -> dict:
    """Creates an empty results dictionary with all required columns.

    The `probs` column stores the full softmax vector as a list per row,
    keeping the output self-contained for decisioner training.

    Returns:
        Dictionary with empty lists for every metadata and probability column.
    """
    return {
        'experiment': [], 'dataset': [], 'image': [], 't': [], 'phase': [],
        'attacked_model': [], 'defense_model': [],
        'attack': [], 'targeted': [], 'targeted_jumps_allowed': [],
        'targeted_label': [], 'norm': [], 'epsilon': [], 'nb_iter': [],
        'actual': [], 'actual_class': [], 'pred': [], 'pred_class': [],
        'attack_time_sec_avg': [], 'defense_time_sec_avg': [],
        'probs': [],
    }


def _append_batch_results(results_dict: dict, probs: list[list[float]],
                           decisions: list[int], classes: list[str],
                           attack_type: str, experiment: str, dataset: str,
                           phase: str, attack: str, targeted: bool,
                           targeted_jumps_allowed: int, epsilon: int,
                           attack_nb_iter: int, img_names: list[str],
                           y_classes: list[int], y_classes_targeted: list[int],
                           output_every_expanded: list[int],
                           attack_time: float, defense_time: float,
                           norm: str = 'linf') -> None:
    """Appends one attack-type's results for a single batch in-place.

    Writes all metadata fields and the full softmax vector (as a list in the
    `probs` column) for either the 'naive' or 'adaptive' attacked_model rows.

    Args:
        results_dict: The accumulator dict to extend (modified in place).
        probs: List of per-sample softmax vectors, shape (n_rows, n_classes).
        decisions: List of predicted class indices, length n_rows.
        classes: Sorted list of class name strings (used for actual_class /
            pred_class labels only).
        attack_type: Label stored in the 'attacked_model' column ('naive' or
            'adaptive').
        experiment: Experiment name.
        dataset: Dataset name.
        phase: Split label (e.g. 'train', 'val', 'test').
        attack: Attack algorithm name.
        targeted: Whether the attack was targeted.
        targeted_jumps_allowed: Number of class jumps allowed for targeted attack.
        epsilon: Perturbation budget as an integer in pixel space.
        attack_nb_iter: Number of attack iterations.
        img_names: Image base-name strings for the batch.
        y_classes: True class indices for each image.
        y_classes_targeted: Target class indices for each image.
        output_every_expanded: Full list of paint-step labels (including t=∞
            placeholder 999999).
        attack_time: Wall-clock seconds spent running the attack.
        defense_time: Wall-clock seconds spent running defence inference.
        norm: Perturbation norm; 'linf' or 'l2'.
    """
    batch_size = len(img_names)
    paint_steps = len(output_every_expanded)
    n = len(decisions)
    avg_attack_time = attack_time / batch_size
    avg_defense_time = defense_time / batch_size

    results_dict['experiment'].extend([experiment] * n)
    results_dict['dataset'].extend([dataset] * n)
    results_dict['image'].extend(np.repeat(img_names, paint_steps).tolist())
    results_dict['t'].extend(output_every_expanded * batch_size)
    results_dict['phase'].extend([phase] * n)
    results_dict['attacked_model'].extend([attack_type] * n)
    results_dict['defense_model'].extend(['adaptive'] * n)
    results_dict['attack'].extend([attack] * n)
    results_dict['targeted'].extend([targeted] * n)
    results_dict['targeted_jumps_allowed'].extend([int(targeted_jumps_allowed)] * n)
    results_dict['targeted_label'].extend(np.repeat(y_classes_targeted, paint_steps).tolist())
    results_dict['norm'].extend([norm] * n)
    results_dict['epsilon'].extend([int(epsilon)] * n)
    results_dict['nb_iter'].extend([attack_nb_iter] * n)
    results_dict['actual'].extend(np.repeat(y_classes, paint_steps).tolist())
    results_dict['actual_class'].extend(
        np.repeat([classes[c] for c in y_classes], paint_steps).tolist())
    results_dict['pred'].extend(decisions)
    results_dict['pred_class'].extend([classes[d] for d in decisions])
    results_dict['attack_time_sec_avg'].extend([avg_attack_time] * n)
    results_dict['defense_time_sec_avg'].extend([avg_defense_time] * n)

    results_dict['probs'].extend(probs)


def attacker(experiment: str, dataset: str, attack: str,
             adaptive_model: torch.nn.Module, naive_model: torch.nn.Module,
             run_naive_attack: int, loader: torch.utils.data.DataLoader,
             phase: str, epsilon: int, targeted: bool, output_every: list[int],
             classes: list[str], attack_nb_iter: int, device: str,
             output_dir: str, output_type: str = 'final_decision',
             norm: str = 'linf', save_parquet: bool = True,
             targeted_jumps_allowed: int = 6,
             loss_fn: Optional[Callable] = None,
             nb_restarts: int = 1,
             use_apgd: bool = False) -> pd.DataFrame:
    """Runs adaptive and (optionally) naïve attacks over an entire data loader.

    For each batch:
    1. Optionally attacks with a naive model (no BPDA surrogate).
    2. Attacks with the adaptive BPDA model.
    3. Runs defence inference (using adaptive_model) on both adversarial inputs.
    4. Accumulates per-(image × paint-step) results rows.

    Saves results to disk as Parquet (with softmax vectors) or lightweight CSV.

    Args:
        experiment: Experiment name written into each results row.
        dataset: Dataset name written into each results row.
        attack: Attack algorithm; one of 'fgsm', 'pgd', or 'aa'.
        adaptive_model: BPDA-wrapped model used as the attack target and for
            defence inference.
        naive_model: Unwrapped model used for the naïve baseline attack
            (only used when run_naive_attack=1).
        run_naive_attack: If non-zero, also runs a naïve attack with
            naive_model and records those rows.
        loader: DataLoader returning (x, y, img_paths) tuples.
        phase: Split label written into each row (e.g. 'train', 'val').
        epsilon: Perturbation budget as an integer in pixel space (divided
            by 255 internally).
        targeted: If True, a random target class is chosen per image.
        output_every: Stroke-count checkpoints for canvas snapshots; the
            original image is appended as an extra step (t=999999).
        classes: Sorted list of class name strings.
        attack_nb_iter: Number of PGD iterations (ignored for FGSM/AA).
        device: Target device string.
        output_dir: Directory where the output file is written.
        output_type: Controls the granularity of output rows.
            - 'paints_inference': one row per (image × paint step), including
              the original image as a final step (t=999999). Used by
              attack_pcl to generate decisioner training data — the full
              confidence trajectory across paint steps is required.
            - 'final_decision': one row per image (only t=999999). Used by
              attack_pcld where the decisioner has already collapsed the
              trajectory into a single prediction and only the final outcome
              matters.
        norm: Perturbation norm passed to attack_batch; 'linf' or 'l2'.
        save_parquet: If True, saves full results including softmax vectors
            as a Parquet file. If False, saves a lightweight CSV without the
            'probs' column.
        targeted_jumps_allowed: Maximum random class offset when generating
            targeted labels. The target is sampled as
            (true_label + randint(1, targeted_jumps_allowed+1)) % n_classes.
            For untargeted attacks this value is unused. Default is 6, which
            is reasonable for ImageNet (1000 classes); use a smaller value
            (e.g. 1) for CIFAR-10 (10 classes).
        loss_fn: Optional custom loss callable passed to attack_batch for the
            PGD attack path. When None, standard cross-entropy on the model's
            direct output is used. Use this to inject multi-step intermediate
            losses for the PCLD pipeline.
        nb_restarts: Number of random restarts for PGD. Default 1 (no restarts).
        use_apgd: If True, apply the APGD step-size adaptation schedule. Only
            effective when ``attack='pgd'``.

    Returns:
        DataFrame with one row per (image × paint step × attack type).
    """
    n_classes = len(classes)
    epsilon_real = epsilon / 255.0
    if output_type == 'paints_inference':
        output_every_expanded = output_every + [999999]
    else:
        output_every_expanded = [999999]
    paint_steps = len(output_every_expanded)

    results_dict = _make_results_dict()

    for batch in loader:
        x, y, img_paths = batch
        x = x.to(device)
        y = y.to(device)
        img_names = [os.path.basename(p) for p in img_paths]
        y_classes = y.cpu().numpy().tolist()

        if targeted:
            jumps = torch.randint(1, targeted_jumps_allowed + 1, y.shape).to(device)
            y_target = (y + jumps) % n_classes
        else:
            y_target = y
        y_classes_targeted = y_target.cpu().numpy().tolist()

        # For PCL (paints_inference) the model outputs (B*Steps, n_classes);
        # repeat labels to match. Naive model (clf) always outputs (B, n_classes).
        if output_type == 'paints_inference':
            y_adaptive_attack = y_target.repeat_interleave(paint_steps)
        else:
            y_adaptive_attack = y_target
        y_naive_attack = y_target

        # --- Naïve attack ---
        x_adv_naive = x
        naive_attack_time = 0.0
        if run_naive_attack:
            t0 = time.time()
            x_adv_naive = attack_batch(naive_model, x, attack, epsilon_real,
                                       attack_nb_iter, targeted, y_naive_attack,
                                       norm=norm)
            naive_attack_time = time.time() - t0

        # --- Adaptive BPDA attack ---
        t0 = time.time()
        x_adv_adaptive = attack_batch(adaptive_model, x, attack, epsilon_real,
                                      attack_nb_iter, targeted, y_adaptive_attack,
                                      norm=norm, loss_fn=loss_fn,
                                      nb_restarts=nb_restarts,
                                      use_apgd=use_apgd)
        adaptive_attack_time = time.time() - t0

        # --- Defence inference (always with adaptive_model) ---
        with torch.no_grad():
            t0 = time.time()
            adaptive_logits = adaptive_model(x_adv_adaptive)
            adaptive_defense_time = time.time() - t0
            adaptive_probs_t = torch.softmax(adaptive_logits, dim=1)
            adaptive_probs = adaptive_probs_t.cpu().numpy().tolist()
            adaptive_decisions = torch.argmax(adaptive_probs_t, dim=1).cpu().numpy().tolist()

            naive_probs, naive_decisions, naive_defense_time = [], [], 0.0
            if run_naive_attack:
                t0 = time.time()
                naive_logits = adaptive_model(x_adv_naive)
                naive_defense_time = time.time() - t0
                naive_probs_t = torch.softmax(naive_logits, dim=1)
                naive_probs = naive_probs_t.cpu().numpy().tolist()
                naive_decisions = torch.argmax(naive_probs_t, dim=1).cpu().numpy().tolist()

        shared = dict(
            classes=classes, experiment=experiment, dataset=dataset,
            phase=phase, attack=attack, targeted=targeted,
            targeted_jumps_allowed=targeted_jumps_allowed,
            epsilon=epsilon, attack_nb_iter=attack_nb_iter,
            img_names=img_names, y_classes=y_classes,
            y_classes_targeted=y_classes_targeted,
            output_every_expanded=output_every_expanded,
            norm=norm,
        )

        if run_naive_attack:
            _append_batch_results(results_dict, naive_probs, naive_decisions,
                                   attack_type='naive',
                                   attack_time=naive_attack_time,
                                   defense_time=naive_defense_time,
                                   **shared)

        _append_batch_results(results_dict, adaptive_probs, adaptive_decisions,
                               attack_type='adaptive',
                               attack_time=adaptive_attack_time,
                               defense_time=adaptive_defense_time,
                               **shared)

    results_df = pd.DataFrame(results_dict)

    os.makedirs(output_dir, exist_ok=True)
    stem = f'{phase}_eps{epsilon}_{norm}_results'
    if save_parquet:
        parquet_path = os.path.join(output_dir, f'{stem}.parquet')
        results_df.to_parquet(parquet_path, compression='snappy', index=False)
        print(f'Saved {len(results_df)} rows to {parquet_path}')
    else:
        csv_path = os.path.join(output_dir, f'{stem}.csv')
        results_df.drop(columns=['probs'], errors='ignore').to_csv(csv_path, index=False)
        print(f'Saved {len(results_df)} rows to {csv_path}')
    print(f'Finished attacking {phase}!')

    return results_df
