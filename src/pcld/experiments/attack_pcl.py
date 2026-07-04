import argparse
import os

import torch

from pcld.models.classifier import get_net
from pcld.models.normalized_model import NormalizedModel
from pcld.attacks.pcld_bpda import BPDAPainter, PCL
from pcld.painter.painter_surrogate import (IdentitySurrogate_, PainterSurrogate,
                                        load_painter_surrogate, load_joint_surrogate)
from pcld.painter.painter_utils import load_painter, paint_images
from pcld.attacks.attacks import attacker
from pcld.utils.consts import NUM_OF_HYPHENS, RESOURCES_RESULTS_DIR, RESOURCES_MODELS_DIR, \
    ACTOR_WEIGHTS_PATH, RENDERER_WEIGHTS_PATH, CIFAR10Consts, IMAGENETConsts
from pcld.data.datasets import transform_dataset, get_loaders
from pcld.utils.integrative import save_args_json


def main_attack_pcl(args: argparse.Namespace, device: str) -> None:
    """Entry point for the attack_pcl experiment.

    Constructs the Painter–Classifier (PCL) pipeline with BPDA gradient
    approximation, then runs the chosen adversarial attack (FGSM/PGD) for
    each epsilon value. The output is a CSV of per-step confidence vectors for
    each image, which is later used as the decisioner's training dataset.

    Model loading order:
    1. Pretrained painter (actor + renderer).
    2. Per-step painter surrogates from the train_surrogate_painter folder.
    3. Pretrained classifier specified by --classifier_experiment.
    4. Full PCL BPDA model assembled from the above components.

    Results are saved per-epsilon to
    resources/results/<experiment_name>/<epsilon>_results.csv.

    Args:
        args: Parsed argument namespace. Reads: dataset, dataset_type, splits,
            experiment_name, model_type, batch_size, output_every,
            classifier_experiment, attack, attack_direction, attack_nb_iter,
            run_naive_attack, epsilons.
        device: Target device string resolved by main.py.
    """
    dataset, dataset_type, splits, experiment_name, model_type, batch_size, output_every, classifier_experiment, \
        attack, attack_direction, attack_nb_iter, run_naive_attack, epsilons = \
        (args.dataset, args.dataset_type, args.splits, args.experiment_name, args.model_type, args.batch_size,
         args.output_every, args.classifier_experiment, args.attack, args.attack_direction, args.attack_nb_iter,
         args.run_naive_attack, args.epsilons)

    # RobustBench convention: DataLoader returns [0, 1] (no normalization).
    # The classifier normalizes internally via NormalizedModel wrapper.
    # This ensures attack clip bounds (0, 1) match the actual input range.
    split_transform = transform_dataset(dataset_type=dataset_type,
                                        preprocessing='ToTensorOnly')
    transform_dict = {split: split_transform for split in splits}

    loaders = get_loaders(dataset, splits, transform_dict, batch_size)

    first_ds = loaders[splits[0]][0]
    classes = sorted(first_ds.class_to_idx.keys())

    actor, renderer = load_painter(ACTOR_WEIGHTS_PATH, RENDERER_WEIGHTS_PATH, device)

    print('-' * NUM_OF_HYPHENS)
    print(f'Load pre-trained painter-surrogates models...')
    surr_local_folder = os.path.join(RESOURCES_MODELS_DIR, 'train_surrogate_painter')
    joint_path = getattr(args, 'joint_surrogate', None) or None
    if joint_path:
        # JointWithIdentity outputs (B, Steps+1, 3, H, W) — same interface
        # as PainterSurrogate with separate models + IdentitySurrogate_.
        painter_surrogate = load_joint_surrogate(joint_path, device,
                                                 num_steps=len(output_every))
        print(f'  Using JointPainterSurrogate from {joint_path}')
    else:
        painter_surrogates_list = load_painter_surrogate(surr_local_folder, device,
                                                         output_every=output_every)
        painter_surrogates_list.append(IdentitySurrogate_().to(device))
        painter_surrogate = PainterSurrogate(painter_surrogates_list)
        print(f'  Using {len(painter_surrogates_list)} separate PainterSurrogate_ models')
    painter_surrogate.to(device).eval()

    print('-' * NUM_OF_HYPHENS)
    print(f'Load pre-trained classifier model...')
    clf_local_path = os.path.join(RESOURCES_MODELS_DIR, classifier_experiment, 'best_model.pth')
    clf = get_net(dataset_type, device, model_type, clf_local_path)
    clf.eval()

    # Wrap classifier with internal normalization (RobustBench convention).
    # The classifier was trained with transforms.Normalize in the DataLoader,
    # so its weights expect normalised input.  NormalizedModel moves the
    # normalization inside the model so attacks operate in [0, 1] pixel space.
    consts = CIFAR10Consts if dataset_type == 'cifar10' else IMAGENETConsts
    clf = NormalizedModel(clf, consts.MEAN, consts.STD).to(device).eval()

    print('-' * NUM_OF_HYPHENS)
    print(f'Creating PCL BPDA model...')
    # mean=None: input is already [0, 1], painter works in [0, 1],
    # NormalizedModel(clf) handles normalisation of canvases internally.
    painter = BPDAPainter(paint_images, painter_surrogate, output_every, device, actor, renderer,
                          mean=None, std=None).to(device).eval()
    pcl = PCL(painter, clf).to(device).eval()

    if torch.cuda.device_count() > 1:
        print("Parallelization: There are ", torch.cuda.device_count(), " GPUs!")
        pcl = torch.nn.DataParallel(pcl)

    results_local_dir = os.path.join(RESOURCES_RESULTS_DIR, experiment_name)
    os.makedirs(results_local_dir, exist_ok=True)
    save_args_json(args, results_local_dir)
    attack_direction_bool = attack_direction == 'targeted'
    save_parquet = bool(args.save_parquet)
    nb_restarts: int = getattr(args, 'attack_nb_restarts', 1)
    use_apgd: bool = bool(getattr(args, 'use_apgd', 0))
    # PCL already outputs (B*Steps, n_classes); CE across all steps is implicit.
    # Multi-step loss weight is not applied here — pass loss_fn=None.
    for epsilon in args.epsilons:
        print(f'attack with epsilon {epsilon}/255...')
        for split in splits:
            attacker(experiment_name, dataset, attack, pcl, clf, run_naive_attack,
                     loaders[split][1], split, epsilon, attack_direction_bool,
                     output_every, classes=classes, attack_nb_iter=attack_nb_iter,
                     device=device, output_dir=results_local_dir,
                     output_type='paints_inference', norm=args.attack_norm,
                     save_parquet=save_parquet,
                     targeted_jumps_allowed=args.targeted_jumps_allowed,
                     loss_fn=None, nb_restarts=nb_restarts,
                     use_apgd=use_apgd)
        print(f'finished attack with epsilon {epsilon}/255!')
