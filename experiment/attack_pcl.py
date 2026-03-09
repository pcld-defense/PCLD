import argparse
import os

import torch

from model.classifier import get_net
from model.pcld_bpda import BPDAPainter, PCL
from painter.painter_surrogate import IdentitySurrogate_, PainterSurrogate, load_painter_surrogate
from painter.painter_utils import load_painter, paint_images
from util.attacks import attacker
from util.consts import NUM_OF_HYPHENS, RESOURCES_RESULTS_DIR, RESOURCES_MODELS_DIR, \
    ACTOR_WEIGHTS_PATH, RENDERER_WEIGHTS_PATH
from util.datasets import transform_dataset, get_loaders


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

    train_transform = transform_dataset(dataset_type=dataset_type,
                                        preprocessing=args.preprocessing)
    val_transform = transform_dataset(dataset_type=dataset_type,
                                      preprocessing=args.preprocessing)
    transform_dict = {"train": train_transform, "val": val_transform}

    loaders = get_loaders(dataset, splits, transform_dict, batch_size)

    train_ds, train_loader = loaders["train"]
    classes = sorted(train_ds.class_to_idx.keys())

    actor, renderer = load_painter(ACTOR_WEIGHTS_PATH, RENDERER_WEIGHTS_PATH, device)

    print('-' * NUM_OF_HYPHENS)
    print(f'Load pre-trained painter-surrogates models...')
    surr_local_folder = os.path.join(RESOURCES_MODELS_DIR, 'train_surrogate_painter')
    painter_surrogates_list = load_painter_surrogate(surr_local_folder, device, output_every=output_every)
    painter_surrogates_list.append(IdentitySurrogate_().to(device))
    painter_surrogate = PainterSurrogate(painter_surrogates_list)
    painter_surrogate.to(device).eval()

    print('-' * NUM_OF_HYPHENS)
    print(f'Load pre-trained classifier model...')
    clf_local_path = os.path.join(RESOURCES_MODELS_DIR, classifier_experiment, 'best_model.pth')
    clf = get_net(dataset_type, device, model_type, clf_local_path)
    clf.eval()

    print('-' * NUM_OF_HYPHENS)
    print(f'Creating PCL BPDA model...')
    painter = BPDAPainter(paint_images, painter_surrogate, output_every, device, actor, renderer).to(device).eval()
    pcl = PCL(painter, clf).to(device).eval()

    if torch.cuda.device_count() > 1:
        print("Parallelization: There are ", torch.cuda.device_count(), " GPUs!")
        pcl = torch.nn.DataParallel(pcl)

    results_local_dir = os.path.join(RESOURCES_RESULTS_DIR, experiment_name)
    os.makedirs(results_local_dir, exist_ok=True)
    attack_direction_bool = attack_direction == 'targeted'
    save_parquet = bool(args.save_parquet)
    for epsilon in args.epsilons:
        print(f'attack with epsilon {epsilon}/255...')
        for split in splits:
            attacker(experiment_name, dataset, attack, pcl, clf, run_naive_attack,
                     loaders[split][1], split, epsilon, attack_direction_bool,
                     output_every, classes=classes, attack_nb_iter=attack_nb_iter,
                     device=device, output_dir=results_local_dir,
                     output_type='paints_inference', norm=args.attack_norm,
                     save_parquet=save_parquet)
        print(f'finished attack with epsilon {epsilon}/255!')
