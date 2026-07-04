import argparse
import os
from typing import Union

import pandas as pd
import torch.nn as nn

from pcld.models.classifier import SUPPORTED_MODELS, get_net
from pcld.models.registry import CLASSIFIER_REGISTRY
from pcld.utils.consts import RESOURCES_MODELS_DIR, RESOURCES_RESULTS_DIR
from pcld.data.datasets import transform_dataset, get_loaders
from pcld.models.train_utils import process_epoch_clf


def _resolve_weights(args: argparse.Namespace, model_type: str) -> Union[str, None]:
    """Determines the checkpoint path (or None) for eval_classifier.

    Priority:
    1. ``args.pretrained_weights`` — explicit local ``.pth`` path.
    2. ``args.classifier_experiment`` — resolves to
       ``RESOURCES_MODELS_DIR/<classifier_experiment>/best_model.pth``.
    3. ``None`` — allowed only when the model auto-loads weights (robustbench
       or timm with ``timm_pretrained=True``).

    Args:
        args: Parsed argument namespace.
        model_type: Key in ``CLASSIFIER_REGISTRY``.

    Returns:
        Resolved checkpoint path, or ``None`` for auto-loading models.

    Raises:
        ValueError: If no weights source is provided and the model does not
            support auto-loading.
    """
    if getattr(args, 'pretrained_weights', None):
        return args.pretrained_weights

    if getattr(args, 'classifier_experiment', None):
        return os.path.join(RESOURCES_MODELS_DIR,
                            args.classifier_experiment, 'best_model.pth')

    cfg = CLASSIFIER_REGISTRY.get(model_type)
    auto_loads = cfg is not None and (
        cfg.family == 'robustbench'
        or (cfg.family == 'timm' and cfg.timm_pretrained)
    )
    if not auto_loads:
        raise ValueError(
            "No weights source found. Provide one of:\n"
            "  --pretrained_weights <path/to/model.pth>\n"
            "  --classifier_experiment <experiment_folder_name>\n"
            "Or use a model that auto-loads weights "
            f"(e.g. 'wrn-28-10-standard', 'xcit-m12'). "
            f"Supported models: {SUPPORTED_MODELS}"
        )
    return None


def main_eval_classifier(args: argparse.Namespace, device: str) -> None:
    """Entry point for the eval_classifier experiment.

    Evaluates a pretrained classifier on one or more dataset splits without
    any gradient updates. Supports three weight sources (in priority order):

    1. ``--pretrained_weights`` — direct path to a ``.pth`` checkpoint.
    2. ``--classifier_experiment`` — loads
       ``resources/models/<name>/best_model.pth``.
    3. Auto-loading models (e.g. ``wrn-28-10-standard`` via RobustBench,
       ``xcit-m12`` via timm hub) — no path argument needed.

    Example CLI usage::

        # From a trained experiment folder
        python main.py --experiment_type eval_classifier \\
            --experiment_suff my_eval \\
            --dataset subset_of_imagenet \\
            --dataset_type imagenet \\
            --splits val test \\
            --classifier_experiment train_classifier_bp \\
            --batch_size 32

        # From a direct checkpoint path
        python main.py --experiment_type eval_classifier \\
            --experiment_suff my_eval \\
            --dataset cifar10_data \\
            --dataset_type cifar10 \\
            --splits test \\
            --model_type wrn-34-10 \\
            --pretrained_weights /path/to/model.pth \\
            --batch_size 64

        # Using RobustBench Standard (non-AT) model — no path needed
        python main.py --experiment_type eval_classifier \\
            --experiment_suff rb_standard \\
            --dataset cifar10_data \\
            --dataset_type cifar10 \\
            --splits test \\
            --model_type wrn-28-10-standard \\
            --batch_size 64

    Args:
        args: Parsed argument namespace. Reads: dataset, dataset_type, splits,
            experiment_name, model_type, batch_size, preprocessing,
            pretrained_weights (optional), classifier_experiment (optional).
        device: Target device string resolved by main.py.
    """
    dataset, dataset_type, splits, experiment_name, model_type, batch_size = (
        args.dataset, args.dataset_type, args.splits, args.experiment_name,
        args.model_type, args.batch_size,
    )

    transform = transform_dataset(dataset_type=dataset_type,
                                  preprocessing=args.preprocessing)
    loaders = get_loaders(dataset, splits, {s: transform for s in splits}, batch_size)

    first_ds = loaders[splits[0]][0]
    classes = sorted(first_ds.class_to_idx.keys())
    n_classes = len(classes)

    weights_path = _resolve_weights(args, model_type)
    print(f'Loading classifier (model_type={model_type!r}, weights={weights_path!r})')
    net = get_net(dataset_type, device, model_type, weights_path, n_classes=n_classes)
    net.eval()

    criterion = nn.CrossEntropyLoss()
    results_df = pd.DataFrame()

    for split in splits:
        print(f'Evaluating on {split}...')
        _, loader = loaders[split]
        results_df = process_epoch_clf(
            experiment=experiment_name,
            device=device,
            epoch=0,
            net=net,
            loader=loader,
            loader_name=split,
            criterion=criterion,
            optimizer=None,
            results_df=results_df,
            n_classes=n_classes,
            classes=classes,
            is_train=False,
            phase=split,
            save_model=False,
        )

    results_dir = os.path.join(RESOURCES_RESULTS_DIR, experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'eval_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f'Evaluation complete. Results saved to {csv_path}')
