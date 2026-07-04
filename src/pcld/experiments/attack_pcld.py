import argparse
import os
from typing import Optional

import torch
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn

from pcld.models.decisioner import Decisioner1DConv, DecisionerFC
from pcld.models.normalized_model import NormalizedModel
from pcld.painter.painter_surrogate import (IdentitySurrogate_, PainterSurrogate,
                                        load_painter_surrogate, load_joint_surrogate)
from pcld.painter.painter_utils import load_painter, paint_images
from pcld.attacks.pcld_bpda import BPDAPainter, CLD, PCLD
from pcld.attacks.attacks import attacker
from pcld.utils.consts import NUM_OF_HYPHENS, IMAGENET_2012_LABELS, RESOURCES_RESULTS_DIR, \
    RESOURCES_MODELS_DIR, CIFAR10Consts, IMAGENETConsts, ACTOR_WEIGHTS_PATH, RENDERER_WEIGHTS_PATH
from pcld.data.datasets import transform_dataset, get_loaders
from pcld.utils.integrative import save_args_json
from pcld.models.train_utils import load_model


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
        attack, attack_direction, attack_nb_iter, run_naive_attack, epsilons = \
        (args.dataset, args.experiment_name, args.batch_size, args.output_every, args.classifier_experiment,
         args.decisioner_experiment, args.attack, args.attack_direction, args.attack_nb_iter, args.run_naive_attack,
         args.epsilons)

    # RobustBench convention: DataLoader returns [0, 1] (no normalization).
    # The classifier normalizes internally via NormalizedModel wrapper.
    split_transform = transform_dataset(dataset_type=args.dataset_type,
                                        preprocessing='ToTensorOnly')
    transform_dict = {split: split_transform for split in args.splits}
    loaders = get_loaders(dataset, args.splits, transform_dict, batch_size)

    # Derive the class set from the dataset so the pipeline matches the trained
    # classifier/decisioner class count. The paper's classifier is trained on a
    # 7-class ImageNet subset, so hardcoding all 1000 ImageNet labels would not
    # match the shipped checkpoints. This mirrors attack_pcl's behaviour.
    first_ds = loaders[args.splits[0]][0]
    classes = sorted(first_ds.class_to_idx.keys())
    n_classes = len(classes)

    actor, renderer = load_painter(ACTOR_WEIGHTS_PATH, RENDERER_WEIGHTS_PATH, device)

    print('-' * NUM_OF_HYPHENS)
    print(f'Load pre-trained painter-surrogates models...')
    surr_local_folder = os.path.join(RESOURCES_MODELS_DIR, 'train_surrogate_painter')
    joint_path: str | None = getattr(args, 'joint_surrogate', None) or None
    if joint_path:
        painter_surrogate = load_joint_surrogate(joint_path, device,
                                                 num_steps=len(output_every))
        num_paint_steps = len(output_every) + 1  # steps + identity
        print(f'  Using JointPainterSurrogate from {joint_path}')
    else:
        painter_surrogates_list = load_painter_surrogate(surr_local_folder, device,
                                                         output_every=output_every)
        painter_surrogates_list.append(IdentitySurrogate_().to(device))
        num_paint_steps = len(painter_surrogates_list)
        painter_surrogate = PainterSurrogate(painter_surrogates_list)
        print(f'  Using {num_paint_steps} separate PainterSurrogate_ models')
    painter_surrogate.to(device).eval()

    print('-' * NUM_OF_HYPHENS)
    print(f'Load pre-trained classifier model...')
    clf_local_path = os.path.join(RESOURCES_MODELS_DIR, classifier_experiment, 'model.pth')
    clf = models.resnet18()
    clf.fc = nn.Linear(clf.fc.in_features, n_classes)
    clf = load_model(clf, clf_local_path, device)
    clf.eval()

    # Wrap classifier with internal normalization (RobustBench convention).
    consts = CIFAR10Consts if args.dataset_type == 'cifar10' else IMAGENETConsts
    clf = NormalizedModel(clf, consts.MEAN, consts.STD).to(device).eval()

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

    print(f'Creating PCLD BPDA model...')
    # mean=None: input is [0, 1], NormalizedModel(clf) handles normalization.
    bpda_painter = BPDAPainter(paint_images, painter_surrogate, output_every, device, actor, renderer,
                               mean=None, std=None).to(device).eval()
    pcld = PCLD(bpda_painter, clf, decisioner, num_paint_steps, decisioner_architecture).to(device).eval()
    print(f'finished creating PCLD BPDA model!')

    # Build the PCLD multi-step loss closure BEFORE DataParallel wrapping so
    # the closure captures the unwrapped submodule references directly.
    # The closure computes:
    #   L = CE(decisioner_output, y) + weight * CE(all_canvas_logits, y_repeated)
    # This provides a direct gradient path to each paint step, counteracting
    # vanishing gradients across 15 paint steps.
    multi_step_weight: float = getattr(args, 'multi_step_loss_weight', 0.0)
    if multi_step_weight > 0.0:
        _painter_ref = bpda_painter
        _clf_ref = clf
        _decisioner_ref = decisioner
        _num_steps = num_paint_steps
        _dec_arch = decisioner_architecture

        def _pcld_loss_fn(x_adv: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Multi-step PCLD loss: decisioner CE + weighted intermediate CE."""
            B = x_adv.shape[0]
            canvases = _painter_ref(x_adv)            # (B, Steps, 3, H, W)
            flat = canvases.view(-1, *canvases.shape[2:])  # (B*Steps, 3, H, W)
            logits_all = _clf_ref(flat)                # (B*Steps, n_classes)
            probs = torch.softmax(logits_all, dim=1)
            if _dec_arch == 'conv':
                dec_input = probs.view(B, _num_steps, -1)
            else:
                dec_input = probs.reshape(B, -1)
            logits_final = _decisioner_ref(dec_input)  # (B, n_classes)
            L_final = F.cross_entropy(logits_final, y)
            L_inter = F.cross_entropy(logits_all, y.repeat_interleave(_num_steps))
            return L_final + multi_step_weight * L_inter

        loss_fn: Optional[callable] = _pcld_loss_fn
        print(f'Using PCLD multi-step loss with weight={multi_step_weight}')
    else:
        loss_fn = None

    nb_restarts: int = getattr(args, 'attack_nb_restarts', 1)
    use_apgd: bool = bool(getattr(args, 'use_apgd', 0))

    if torch.cuda.device_count() > 1:
        print("Parallelization: There are ", torch.cuda.device_count(), " GPUs!")
        cld = torch.nn.DataParallel(cld)
        pcld = torch.nn.DataParallel(pcld)

    results_local_dir = os.path.join(RESOURCES_RESULTS_DIR, experiment_name)
    os.makedirs(results_local_dir, exist_ok=True)
    save_args_json(args, results_local_dir)
    targeted = attack_direction == 'targeted'
    save_parquet = bool(args.save_parquet)

    for epsilon in args.epsilons:
        print(f'attack with epsilon {epsilon}/255...')
        for split in args.splits:
            attacker(experiment_name, dataset, attack, pcld, cld, run_naive_attack,
                     loaders[split][1], split, epsilon, targeted, output_every,
                     classes, attack_nb_iter, device, output_dir=results_local_dir,
                     norm=args.attack_norm, save_parquet=save_parquet,
                     targeted_jumps_allowed=args.targeted_jumps_allowed,
                     loss_fn=loss_fn, nb_restarts=nb_restarts,
                     use_apgd=use_apgd)
        print(f'finished attack with epsilon {epsilon}/255!')
