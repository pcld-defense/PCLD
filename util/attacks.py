import os
import random
import time

import numpy as np
import pandas as pd
import torch
from autoattack import AutoAttack
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent


def attack_batch(model: torch.nn.Module, x: torch.Tensor, attack: str,
                 epsilon: float, attack_nb_iter: int, targeted: bool,
                 y_classes_targeted: torch.Tensor) -> torch.Tensor:
    """Generates adversarial examples for a single batch using the chosen attack.

    When epsilon is 0 the input is returned unchanged. Supported attacks:
    'fgsm' (single-step L-inf), 'pgd' (multi-step L-inf), and 'aa'
    (AutoAttack standard evaluation).

    Args:
        model: The model to attack; must accept (B, 3, H, W) input.
        x: Clean input batch of shape (B, 3, H, W) in [0, 1].
        attack: Attack name; one of 'fgsm', 'pgd', or 'aa'.
        epsilon: L-inf perturbation budget in [0, 1] (already normalised).
        attack_nb_iter: Number of PGD iterations (ignored for FGSM/AA).
        targeted: If True the attack minimises the loss toward the target class;
            if False it maximises the loss away from the true class.
        y_classes_targeted: Label tensor of shape (B,) with target class indices
            for targeted attacks or true class indices for untargeted attacks.

    Returns:
        Adversarial example batch of shape (B, 3, H, W) in [0, 1].
    """
    if epsilon == 0:
        return x

    if attack == 'fgsm':
        x_adv = fast_gradient_method(model_fn=model,
                                     x=x,
                                     eps=epsilon,
                                     norm=np.inf,
                                     y=y_classes_targeted,
                                     targeted=targeted,
                                     clip_min=0,
                                     clip_max=1)
    elif attack == 'pgd':
        x_adv = projected_gradient_descent(model_fn=model,
                                           x=x,
                                           eps=epsilon,
                                           eps_iter=epsilon / attack_nb_iter,
                                           nb_iter=attack_nb_iter,
                                           norm=np.inf,
                                           y=y_classes_targeted,
                                           targeted=targeted,
                                           rand_init=False,
                                           sanity_checks=False,
                                           clip_min=0,
                                           clip_max=1)
    elif attack == 'aa':
        adv_attack = AutoAttack(model, norm='Linf', eps=epsilon, version='standard')
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
                           attack_time: float, defense_time: float) -> None:
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
        epsilon: L-inf budget as an integer in pixel space.
        attack_nb_iter: Number of attack iterations.
        img_names: Image base-name strings for the batch.
        y_classes: True class indices for each image.
        y_classes_targeted: Target class indices for each image.
        output_every_expanded: Full list of paint-step labels (including t=∞
            placeholder 999999).
        attack_time: Wall-clock seconds spent running the attack.
        defense_time: Wall-clock seconds spent running defence inference.
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
    results_dict['norm'].extend(['linf'] * n)
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
             output_dir: str, output_type: str = 'final_decision') -> pd.DataFrame:
    """Runs adaptive and (optionally) naïve attacks over an entire data loader.

    For each batch:
    1. Generates adversarial examples via the adaptive BPDA model (and the
       naïve CLD model when run_naive_attack is set).
    2. Passes both adversarial inputs through the adaptive model to get softmax
       probabilities.
    3. Accumulates all metadata and softmax values into a unified results dict.

    After all batches, the complete results are saved as a single Parquet file
    to `output_dir`. Softmax probabilities are stored as inline `prob_<class>`
    columns alongside all metadata, making the file self-contained for
    decisioner training (no separate HDF5 required).

    Output file naming: `<phase>_eps<epsilon>_results.parquet`

    Args:
        experiment: Experiment name stored in every result row.
        dataset: Dataset name stored in every result row.
        attack: Attack algorithm; one of 'fgsm', 'pgd', or 'aa'.
        adaptive_model: The PCLD/BPDA model used for the adaptive attack and
            defence inference.
        naive_model: The CLD model used for the naïve baseline attack (only
            queried when run_naive_attack is non-zero).
        run_naive_attack: When non-zero, also generates naïve adversarial
            examples and records their defence results for comparison.
        loader: DataLoader yielding (images, labels, paths) tuples.
        phase: Dataset split label (e.g. 'train', 'val', 'test').
        epsilon: L-inf budget as an integer in pixel space (divided by 255
            internally before calling attack_batch).
        targeted: If True, attacks toward a randomly chosen incorrect class.
        output_every: Ordered stroke-count checkpoints used to label the 't'
            column (one entry per paint step).
        classes: Sorted list of class name strings.
        attack_nb_iter: Number of PGD iterations.
        device: Target device string.
        output_dir: Directory where the output Parquet file will be written.
        output_type: Controls the model output shape. 'final_decision' expects
            (B, n_classes) from a PCLD model; 'paints_inference' expects
            (B * Steps, n_classes) from a PCL model and records one row per
            (image × paint step).

    Returns:
        DataFrame with one row per (image × paint step × attack type) containing
        all metadata and per-class softmax probabilities.
    """
    print(f'run attacks on {phase}...')
    epsilon_real = epsilon / 255.0

    # In paints_inference mode each paint step is a separate row;
    # in final_decision mode there is a single row per image (t=-1 placeholder).
    output_every_expanded = output_every + [999999] if output_type == 'paints_inference' else [-1]
    paint_steps = len(output_every_expanded)

    targeted_jumps_allowed = 6 if targeted else 1
    results_dict = _make_results_dict()

    for i, data in enumerate(loader):
        print(f'batch {i} attack...')
        x, y, paths = data[0].to(device), data[1].to(device), data[2]
        img_names = [p.split('/')[-1].split('.')[0] for p in paths]
        y_classes = [yi.item() for yi in y]
        y_classes_targeted = [
            int((yi.item() + random.randint(1, targeted_jumps_allowed)) % len(classes))
            for yi in y
        ]

        y_target_tensor = torch.tensor(y_classes_targeted, dtype=torch.long, device=device)
        y_true_tensor = torch.tensor(y_classes, dtype=torch.long, device=device)

        # In paints_inference mode the PCL model returns B*Steps rows, so labels
        # must be repeated to match.
        if output_type == 'paints_inference':
            y_target_tensor = y_target_tensor.repeat_interleave(paint_steps)
            y_true_tensor = y_true_tensor.repeat_interleave(paint_steps)

        y_attack_labels = y_target_tensor if targeted else y_true_tensor

        # --- Naïve baseline attack ---
        t0 = time.time()
        x_adv_naive = x
        if run_naive_attack:
            x_adv_naive = attack_batch(naive_model, x, attack, epsilon_real,
                                       attack_nb_iter, targeted, y_attack_labels)
        naive_attack_time = time.time() - t0

        # --- Adaptive BPDA attack ---
        t0 = time.time()
        x_adv_adaptive = attack_batch(adaptive_model, x, attack, epsilon_real,
                                      attack_nb_iter, targeted, y_attack_labels)
        adaptive_attack_time = time.time() - t0

        # --- Defence inference ---
        t0 = time.time()
        with torch.no_grad():
            naive_probs = torch.softmax(adaptive_model(x_adv_naive), dim=1).tolist()
            adaptive_probs = torch.softmax(adaptive_model(x_adv_adaptive), dim=1).tolist()
        defense_time = (time.time() - t0) / 2  # averaged over the two forward passes

        naive_decisions = np.argmax(naive_probs, axis=1).tolist()
        adaptive_decisions = np.argmax(adaptive_probs, axis=1).tolist()

        # --- Accumulate results ---
        shared = dict(experiment=experiment, dataset=dataset, phase=phase,
                      attack=attack, targeted=targeted,
                      targeted_jumps_allowed=targeted_jumps_allowed,
                      epsilon=epsilon, attack_nb_iter=attack_nb_iter,
                      img_names=img_names, y_classes=y_classes,
                      y_classes_targeted=y_classes_targeted,
                      output_every_expanded=output_every_expanded,
                      classes=classes)

        if run_naive_attack:
            _append_batch_results(results_dict, naive_probs, naive_decisions,
                                  attack_type='naive',
                                  attack_time=naive_attack_time,
                                  defense_time=defense_time, **shared)

        _append_batch_results(results_dict, adaptive_probs, adaptive_decisions,
                              attack_type='adaptive',
                              attack_time=adaptive_attack_time,
                              defense_time=defense_time, **shared)

        print(f'finished attacking batch {i}!')

    results_df = pd.DataFrame(results_dict)

    os.makedirs(output_dir, exist_ok=True)
    parquet_path = os.path.join(output_dir, f'{phase}_eps{epsilon}_results.parquet')
    results_df.to_parquet(parquet_path, compression='snappy', index=False)
    print(f'Saved {len(results_df)} rows to {parquet_path}')
    print(f'Finished attacking {phase}!')

    return results_df

