import argparse

from pcld.experiments.paint_dataset import main_paint_dataset
from pcld.experiments.train_classifier import main_train_classifier
from pcld.experiments.eval_classifier import main_eval_classifier
from pcld.experiments.attack_pcl import main_attack_pcl
from pcld.experiments.train_decisioner import main_train_decisioner
from pcld.experiments.attack_pcld import main_attack_pcld
from pcld.experiments.train_surrogate_painter import main_train_surrogate_painter


def apply_experiment(args: argparse.Namespace, device: str) -> None:
    """Dispatches execution to the correct experiment entry-point.

    Reads `args.experiment_type` and calls the matching main function,
    passing the full argument namespace and the resolved device string.
    Supported experiment types:

    - 'paint_dataset'             – Paint a raw dataset and save canvases to disk.
    - 'train_classifier'          – Train a ResNet classifier on painted images.
    - 'eval_classifier'           – Evaluate a pretrained classifier on one or more
                                    splits without any training; saves per-split
                                    accuracy metrics to a CSV.
    - 'attack_pcl'                – Attack the Painter–Classifier (PCL) pipeline
                                    with BPDA and collect per-step confidence
                                    trajectories for decisioner training.
    - 'train_decisioner'          – Train the decisioner on PCL attack outputs.
    - 'attack_pcld'               – Attack the full PCLD pipeline with an adaptive
                                    BPDA attack and evaluate robustness.
    - 'train_surrogate_painter'   – Train per-step PainterSurrogate_ models used
                                    for BPDA gradient approximation during attacks.
                                    One surrogate is trained and saved per step in
                                    --output_every.

    Args:
        args: Parsed argument namespace from `parse_args()`.
        device: Device string selected by main.py (e.g. 'cuda' or 'cpu').
    """
    params = {'args': args, 'device': device}

    if args.experiment_type == 'paint_dataset':
        main_paint_dataset(**params)
    elif args.experiment_type == 'train_classifier':
        main_train_classifier(**params)
    elif args.experiment_type == 'eval_classifier':
        main_eval_classifier(**params)
    elif args.experiment_type == 'attack_pcl':
        main_attack_pcl(**params)
    elif args.experiment_type == 'attack_pcld':
        main_attack_pcld(**params)
    elif args.experiment_type == 'train_decisioner':
        main_train_decisioner(**params)
    elif args.experiment_type == 'train_surrogate_painter':
        main_train_surrogate_painter(**params)
