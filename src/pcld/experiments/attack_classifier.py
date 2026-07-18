"""Attack a standalone classifier (no painter, no decisioner).

Runs the real adversarial attack path (FGSM/PGD/AA via ``attacker()``)
directly against a single classifier over an evaluation set, producing the
same per-epsilon result parquet files the PCL/PCLD attack experiments write.
This is the per-model workhorse behind the RobustBench comparison sweep
(``scripts/sweep.py``): each sweep entry dispatches here with a different
``model_type`` / ``attack_norm``.

Heavy (torch) imports are deferred into ``main_attack_classifier`` so the
pure helpers in this module stay importable in torch-free environments.
"""

import argparse
import os


def build_epsilon_list(epsilons: list) -> list:
    """Builds the epsilon schedule, guaranteeing a clean (epsilon 0) pass.

    ``attack_batch`` returns the input unchanged when ``epsilon == 0``, so the
    epsilon-0 run yields the clean-accuracy rows the comparison table needs.

    Args:
        epsilons: Configured perturbation budgets (ints for linf, absolute
            floats allowed for l2).

    Returns:
        The epsilon list with ``0`` prepended exactly once if it was missing.
    """
    return list(epsilons) if 0 in epsilons else [0] + list(epsilons)


def main_attack_classifier(args: argparse.Namespace, device: str) -> None:
    """Entry point for the attack_classifier experiment.

    Loads the evaluation data (folder pipeline or the fixed RobustBench
    test-set prefix, per ``args.data_source``), builds the classifier, and
    runs the configured attack for every epsilon. RobustBench-family models
    normalize internally and are used raw; every other family is wrapped in
    ``NormalizedModel`` so attacks operate in [0, 1] pixel space either way.

    Results are saved per (split, epsilon) to
    ``resources/results/<experiment_name>/<split>_eps<eps>_<norm>_results.parquet``
    with the schema ``pcld.eval.metrics.robust_accuracy`` consumes.

    Args:
        args: Parsed argument namespace. Reads: dataset, dataset_type, splits,
            experiment_name, model_type, batch_size, data_source, num_samples,
            attack, attack_direction, attack_nb_iter, attack_norm, epsilons,
            output_every, targeted_jumps_allowed, attack_nb_restarts,
            use_apgd, eot_samples, aa_version.
        device: Target device string resolved by the runner.
    """
    import torch

    from pcld.attacks.attacks import attacker
    from pcld.data.datasets import build_eval_loaders
    from pcld.models.classifier import get_net
    from pcld.models.normalized_model import NormalizedModel
    from pcld.models.registry import CLASSIFIER_REGISTRY
    from pcld.utils.consts import (NUM_OF_HYPHENS, RESOURCES_RESULTS_DIR,
                                   CIFAR10Consts, CIFAR100Consts,
                                   IMAGENETConsts)
    from pcld.utils.integrative import save_args_json

    run_dir = os.path.join(RESOURCES_RESULTS_DIR, args.experiment_name)
    os.makedirs(run_dir, exist_ok=True)
    save_args_json(args, run_dir)

    loaders = build_eval_loaders(args, args.batch_size, run_dir=run_dir)

    first_ds = next(iter(loaders.values()))[0]
    # Class names in label-index order (works for both ImageFolder's
    # alphabetical mapping and RBPrefixDataset's enumeration mapping).
    classes = [c for c, _ in sorted(first_ds.class_to_idx.items(),
                                    key=lambda kv: kv[1])]
    n_classes = len(classes)

    print('-' * NUM_OF_HYPHENS)
    print(f'Load classifier model (model_type={args.model_type!r})...')
    net = get_net(args.dataset_type, device, args.model_type, weights=None,
                  n_classes=n_classes)

    cfg = CLASSIFIER_REGISTRY.get(args.model_type)
    if cfg is not None and cfg.family == 'robustbench':
        # RobustBench models normalize internally: feed raw [0, 1] inputs.
        model = net
    else:
        consts = {'cifar10': CIFAR10Consts,
                  'cifar100': CIFAR100Consts}.get(args.dataset_type,
                                                  IMAGENETConsts)
        model = NormalizedModel(net, consts.MEAN, consts.STD)
    model = model.to(device).eval()

    if torch.cuda.device_count() > 1:
        print("Parallelization: There are ", torch.cuda.device_count(), " GPUs!")
        model = torch.nn.DataParallel(model)

    targeted = args.attack_direction == 'targeted'
    eps_list = build_epsilon_list(args.epsilons)

    for epsilon in eps_list:
        print(f'attack with epsilon {epsilon} ({args.attack_norm})...')
        for split, (_, loader) in loaders.items():
            attacker(args.experiment_name, args.dataset, args.attack,
                     model, model, 0, loader, split, epsilon, targeted,
                     args.output_every, classes=classes,
                     attack_nb_iter=args.attack_nb_iter, device=device,
                     output_dir=run_dir, output_type='final_decision',
                     norm=args.attack_norm, save_parquet=True,
                     targeted_jumps_allowed=args.targeted_jumps_allowed,
                     loss_fn=None,
                     nb_restarts=getattr(args, 'attack_nb_restarts', 1),
                     use_apgd=bool(getattr(args, 'use_apgd', 0)),
                     eot_samples=int(getattr(args, 'eot_samples', 1)),
                     aa_version=getattr(args, 'aa_version', 'standard'))
        print(f'finished attack with epsilon {epsilon}!')
