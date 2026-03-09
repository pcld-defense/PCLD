import argparse
import os

import torch
from torchvision import models
import torch.nn as nn

from model.decisioner import Decisioner1DConv, DecisionerFC
from painter.painter_surrogate import IdentitySurrogate_, PainterSurrogate, load_painter_surrogate
from painter.painter_utils import load_painter, paint_images
from model.pcld_bpda import BPDAPainter, CLD, PCLD
from util.attacks import attacker
from util.consts import NUM_OF_HYPHENS, IMAGENET_2012_LABELS, RESOURCES_RESULTS_DIR, \
    RESOURCES_MODELS_DIR, CIFAR10Consts, IMAGENETConsts
from util.datasets import transform_dataset, get_loaders
from util.models import load_model


def main_attack_pcld(args: argparse.Namespace, device: str) -> None:
    """Entry point for the attack_pcld experiment.

    Assembles the full Painter–Classifier–Decisioner (PCLD) pipeline and runs
    an adaptive white-box attack (BPDA) against it. A naïve baseline attack
    (no painter in the loop) can optionally be run for comparison.

    Model loading order:
    1. Pretrained painter (actor + renderer).
    2. Per-step painter surrogates from train_surrogate_painter.
    3. Pretrained classifier from --classifier_experiment.
    4. Pretrained decisioner from --decisioner_experiment.
    5. CLD (no painter) assembled for the naïve baseline.
    6. PCLD with BPDAPainter for the adaptive attack.

    For each epsilon value, attacks the test split (and train/val if
    --attack_train is set). Results are saved as a running CSV to
    resources/results/<experiment_name>/results.csv, updated after each epsilon.

    Args:
        args: Parsed argument namespace. Reads: dataset, experiment_name,
            batch_size, output_every, classifier_experiment,
            decisioner_experiment, attack, attack_direction, attack_nb_iter,
            run_naive_attack, attack_train, epsilons.
        device: Target device string resolved by main.py.
    """
    dataset, experiment_name, batch_size, output_every, classifier_experiment, decisioner_experiment, \
        attack, attack_direction, attack_nb_iter, run_naive_attack, attack_train, epsilons = \
        (args.dataset, args.experiment_name, args.batch_size, args.output_every, args.classifier_experiment,
         args.decisioner_experiment, args.attack, args.attack_direction, args.attack_nb_iter, args.run_naive_attack,
         args.attack_train, args.epsilons)

    n_classes = len(IMAGENET_2012_LABELS.keys())
    classes = sorted(IMAGENET_2012_LABELS.values())
    train_transform = transform_dataset(dataset_type=args.dataset_type,
                                        preprocessing=args.preprocessing)
    test_transform = transform_dataset(dataset_type=args.dataset_type,
                                       preprocessing=args.preprocessing)
    loaders = get_loaders(dataset, train_transform, test_transform, batch_size)

    actor, renderer = load_painter(device)

    print('-' * NUM_OF_HYPHENS)
    print(f'Load pre-trained painter-surrogates models...')
    surr_local_folder = os.path.join(RESOURCES_MODELS_DIR, 'train_surrogate_painter')
    painter_surrogates_list = load_painter_surrogate(surr_local_folder, device, output_every=output_every)
    painter_surrogates_list.append(IdentitySurrogate_().to(device))
    num_paint_steps = len(painter_surrogates_list)
    painter_surrogate = PainterSurrogate(painter_surrogates_list)
    painter_surrogate.to(device).eval()

    print('-' * NUM_OF_HYPHENS)
    print(f'Load pre-trained classifier model...')
    clf_local_path = os.path.join(RESOURCES_MODELS_DIR, classifier_experiment, 'model.pth')
    clf = models.resnet18()
    clf.fc = nn.Linear(clf.fc.in_features, n_classes)
    clf = load_model(clf, clf_local_path, device)
    clf.eval()

    print('-' * NUM_OF_HYPHENS)
    print(f'Load pre-trained decisioner model...')
    decisioner_local_path = os.path.join(RESOURCES_MODELS_DIR, decisioner_experiment, 'model.pth')
    decisioner_architecture = 'conv'
    if 'conv' in decisioner_experiment:
        decisioner = Decisioner1DConv(n_classes, num_paint_steps, num_filters=32)
    else:
        decisioner = DecisionerFC(n_classes, num_paint_steps)
        decisioner_architecture = 'fc'
    decisioner = load_model(decisioner, decisioner_local_path, device)
    decisioner = decisioner.to(device)
    decisioner.eval()

    print('-' * NUM_OF_HYPHENS)
    print(f'Creating classifier-decisioner (CLD) model for Naïve attacks...')
    cld = CLD(clf, decisioner, num_paint_steps, decisioner_architecture)
    cld = cld.to(device)
    cld.eval()

    consts = CIFAR10Consts if args.dataset_type == 'cifar10' else IMAGENETConsts
    norm_mean = torch.tensor(consts.MEAN)
    norm_std  = torch.tensor(consts.STD)

    print(f'Creating PCLD BPDA model...')
    bpda_painter = BPDAPainter(paint_images, painter_surrogate, output_every, device, actor, renderer,
                               mean=norm_mean, std=norm_std).to(device).eval()
    pcld = PCLD(bpda_painter, clf, decisioner, num_paint_steps, decisioner_architecture).to(device).eval()
    print(f'finished creating PCLD BPDA model!')

    if torch.cuda.device_count() > 1:
        print("Parallelization: There are ", torch.cuda.device_count(), " GPUs!")
        cld = torch.nn.DataParallel(cld)
        pcld = torch.nn.DataParallel(pcld)

    results_local_dir = os.path.join(RESOURCES_RESULTS_DIR, experiment_name)
    os.makedirs(results_local_dir, exist_ok=True)
    targeted = attack_direction == 'targeted'

    for epsilon in args.epsilons:
        print(f'attack with epsilon {epsilon}/255...')
        if attack_train:
            attacker(experiment_name, dataset, attack, pcld, cld, run_naive_attack,
                     loaders['train'][1], 'train', epsilon, targeted, output_every,
                     classes, attack_nb_iter, device, output_dir=results_local_dir,
                     targeted_jumps_allowed=args.targeted_jumps_allowed)
            attacker(experiment_name, dataset, attack, pcld, cld, run_naive_attack,
                     loaders['val'][1], 'val', epsilon, targeted, output_every,
                     classes, attack_nb_iter, device, output_dir=results_local_dir,
                     targeted_jumps_allowed=args.targeted_jumps_allowed)
        attacker(experiment_name, dataset, attack, pcld, cld, run_naive_attack,
                 loaders['test'][1], 'test', epsilon, targeted, output_every,
                 classes, attack_nb_iter, device, output_dir=results_local_dir)
        print(f'finished attack with epsilon {epsilon}/255!')
