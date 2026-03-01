import argparse
import os

import pandas as pd
import torch.nn as nn

from model.classifier import get_net
from util.consts import RESOURCES_MODELS_DIR, RESOURCES_RESULTS_DIR
from util.datasets import transform_dataset, get_loaders
from util.models import process_epoch_clf


def main_eval_classifier(args: argparse.Namespace, device: str) -> None:
    """Entry point for the eval_classifier experiment.

    Evaluates a pretrained classifier on one or more dataset splits without
    any gradient updates. Loads the best checkpoint from the specified
    classifier experiment folder, runs inference over every requested split,
    and saves the per-split accuracy metrics to a CSV file.

    Example CLI usage::

        python main.py --experiment_type eval_classifier \\
            --experiment_suff my_eval \\
            --dataset subset_of_imagenet \\
            --dataset_type imagenet \\
            --splits val test \\
            --classifier_experiment train_classifier_bp \\
            --batch_size 32

    Args:
        args: Parsed argument namespace. Reads: dataset, dataset_type, splits,
            experiment_name, model_type, batch_size, classifier_experiment.
        device: Target device string resolved by main.py.
    """
    dataset, dataset_type, splits, experiment_name, model_type, batch_size, classifier_experiment = (
        args.dataset, args.dataset_type, args.splits, args.experiment_name,
        args.model_type, args.batch_size, args.classifier_experiment
    )

    transform = transform_dataset(dataset_type=dataset_type,
                                  preprocessing=args.preprocessing)
    loaders = get_loaders(dataset, splits, {s: transform for s in splits}, batch_size)

    first_ds = loaders[splits[0]][0]
    classes = sorted(first_ds.class_to_idx.keys())
    n_classes = len(classes)

    clf_path = os.path.join(RESOURCES_MODELS_DIR, classifier_experiment, 'best_model.pth')
    print(f'Loading classifier from {clf_path}')
    net = get_net(dataset_type, device, model_type, clf_path)
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
