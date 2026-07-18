"""Gradient-validity battery experiment (the research plan's R01 gate).

Runs the six gradient-masking checks in :mod:`pcld.eval.gradient_battery`
against the full PCLD BPDA pipeline and writes the gate artifacts
(``gradient_battery.json``, ``loss_curve.csv``, ``eps_sweep.csv``) to the
experiment's results directory. A passing gate is the precondition for all
cluster robustness runs: DiffPure-style defences have lost 25+ points of
claimed robustness once their gradients were corrected, so reviewers expect
this audit as an appendix.

Heavy (torch) imports are deferred into ``main_gradient_battery`` so the
module stays importable in torch-free environments, matching the
attack_classifier idiom.
"""

import argparse
import os


def main_gradient_battery(args: argparse.Namespace, device: str) -> None:
    """Entry point for the gradient_battery experiment.

    Loads the evaluation data (folder pipeline or the fixed RobustBench
    test-set prefix, per ``args.data_source``), assembles the PCLD pipeline
    via ``build_pcld``, and runs the six-check gradient-validity battery.
    The loss under test is the sum-reduced cross-entropy over the PCLD
    decisioner logits (gradients flow through the BPDA painter); the attack
    used for the epsilon sweeps is the configured attack dispatched through
    ``attack_batch`` (untargeted).

    Artifacts are written to
    ``resources/results/<experiment_name>/`` — ``gradient_battery.json``
    (per-check results + gate verdict), ``loss_curve.csv`` (PGD
    loss-vs-iteration), and ``eps_sweep.csv`` (robust accuracy per epsilon,
    including the unbounded epsilon=255 row).

    Args:
        args: Parsed argument namespace. Reads: dataset, dataset_type, splits,
            experiment_name, batch_size, data_source, num_samples,
            output_every, classifier_experiment, decisioner_experiment,
            surrogate_type, joint_surrogate, attack, attack_nb_iter,
            attack_norm, battery_epsilons, battery_fd_dirs, battery_fd_delta.
        device: Target device string resolved by the runner.
    """
    import torch
    import torch.nn.functional as F

    from pcld.attacks.attacks import attack_batch
    from pcld.data.datasets import build_eval_loaders
    from pcld.eval.gradient_battery import run_gradient_battery
    from pcld.experiments.attack_pcld import build_pcld
    from pcld.utils.consts import NUM_OF_HYPHENS, RESOURCES_RESULTS_DIR
    from pcld.utils.integrative import save_args_json

    run_dir = os.path.join(RESOURCES_RESULTS_DIR, args.experiment_name)
    os.makedirs(run_dir, exist_ok=True)
    save_args_json(args, run_dir)

    loaders = build_eval_loaders(args, args.batch_size, run_dir=run_dir)
    split = 'test' if 'test' in loaders else args.splits[0]
    first_ds, loader = loaders[split]
    # Class names in label-index order (works for both ImageFolder's
    # alphabetical mapping and RBPrefixDataset's enumeration mapping).
    classes = [c for c, _ in sorted(first_ds.class_to_idx.items(),
                                    key=lambda kv: kv[1])]
    n_classes = len(classes)

    built = build_pcld(args, device, n_classes)
    pcld = built['pcld']

    def _pcld_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Sum-reduced cross-entropy over the PCLD decisioner logits."""
        return F.cross_entropy(pcld(x), y, reduction='sum')

    def _attack(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
                epsilon: float) -> torch.Tensor:
        """Untargeted attack via the registry dispatch at the given epsilon."""
        return attack_batch(model, x, args.attack, epsilon,
                            args.attack_nb_iter, False, y,
                            norm=args.attack_norm)

    print('-' * NUM_OF_HYPHENS)
    print(f'Running gradient-validity battery on split {split!r}...')
    summary = run_gradient_battery(pcld, _pcld_loss, loader, args, device,
                                   run_dir, attack_callable=_attack)
    print(f"Gradient battery finished: gate "
          f"{'PASSED' if summary['gate_passed'] else 'FAILED'} "
          f"({summary['n_passed']}/{summary['n_checks']} checks)")
