import os
import pandas as pd
import torch
from torchvision import models
import torch.nn as nn

from model.decisioner import Decisioner1DConv, DecisionerFC
from model.model_utils import load_painter_surrogate
from model.painter_surrogate import IdentitySurrogate_, PainterSurrogate
from model.painter_utils import load_painter, paint_images
from model.pcld_bpda import BPDAPainter, PCL, CLD, PCLD
from util.attacks import attacker
from util.consts import NUM_OF_HYPHENS, IMAGENET_7_LABELS, RESOURCES_RESULTS_DIR, \
    RESOURCES_MODELS_DIR
from util.datasets import transform_dataset, get_loaders
from util.models import load_model


def main_attack_pcld(args, device):
    # Fail fast
    dataset, experiment_name, batch_size, output_every, classifier_experiment, decisioner_experiment, \
        attack, attack_direction, attack_nb_iter, run_naive_attack, attack_train, epsilons = \
        (args.dataset, args.experiment_name, args.batch_size, args.output_every, args.classifier_experiment,
         args.decisioner_experiment, args.attack, args.attack_direction, args.attack_nb_iter, args.run_naive_attack,
         args.attack_train, args.epsilons)

    # =================== Load the dataset =================== #
    n_classes = len(IMAGENET_7_LABELS.keys())
    classes = sorted(IMAGENET_7_LABELS.values())
    train_transform = transform_dataset(augmentations=False, to_integers=False)
    test_transform = transform_dataset(augmentations=False, to_integers=False)
    loaders = get_loaders(dataset, train_transform, test_transform, batch_size)

    # =================== Load painter models =================== #
    actor, renderer = load_painter(device)

    # =================== Load painter's surrogate model =================== #
    print('-' * NUM_OF_HYPHENS)
    print(f'Load pre-trained painter-surrogates models...')
    surr_local_folder = os.path.join(RESOURCES_MODELS_DIR, 'train_surrogate_painter')
    painter_surrogates_list = load_painter_surrogate(surr_local_folder, device, output_every=output_every)
    painter_surrogates_list.append(IdentitySurrogate_().to(device))  # add the image itself (t=∞) surrogate
    num_paint_steps = len(painter_surrogates_list)
    painter_surrogate = PainterSurrogate(painter_surrogates_list)
    painter_surrogate.to(device).eval()

    # =================== Load classifier model =================== #
    print('-' * NUM_OF_HYPHENS)
    print(f'Load pre-trained classifier model...')
    clf_local_path = os.path.join(RESOURCES_MODELS_DIR, classifier_experiment, 'model.pth')
    clf = models.resnet18()
    clf.fc = nn.Linear(clf.fc.in_features, n_classes)
    clf = load_model(clf, clf_local_path, device)
    clf.eval()

    # =================== Load decisioner model =================== #
    print('-' * NUM_OF_HYPHENS)
    print(f'Load pre-trained decisioner model...')
    decisioner_local_path = os.path.join(RESOURCES_MODELS_DIR, decisioner_experiment, 'model.pth')
    decisioner_architecture = 'conv'
    if 'conv' in decisioner_experiment:
        decisioner = Decisioner1DConv(n_classes, num_paint_steps, num_filters=32)
    else:  # fc
        decisioner = DecisionerFC(n_classes, num_paint_steps)
        decisioner_architecture = 'fc'
    decisioner = load_model(decisioner, decisioner_local_path, device)
    decisioner = decisioner.to(device)
    decisioner.eval()

    # =================== Creating Classifier-Decisioner (CLD) model for Naïve attacks =================== #
    print('-' * NUM_OF_HYPHENS)
    print(f'Creating classifier-decisioner (CLD) model for Naïve attacks...')
    cld = CLD(clf, decisioner, num_paint_steps, decisioner_architecture)
    cld = cld.to(device)
    cld.eval()

    # =================== Creating PCLD BPDA =================== #
    print(f'Creating PCLD BPDA model...')
    bpda_painter = BPDAPainter(paint_images, painter_surrogate, output_every, device, actor, renderer).to(device).eval()
    pcld = PCLD(bpda_painter, clf, decisioner, num_paint_steps, decisioner_architecture).to(device).eval()
    print(f'finished creating PCLD BPDA model!')

    # =================== Parallelize models =================== #
    if torch.cuda.device_count() > 1:
        print("Parallelization: There are ", torch.cuda.device_count(), " GPUs!")
        cld = torch.nn.DataParallel(cld)
        pcld = torch.nn.DataParallel(pcld)

    # =================== Attack =================== #
    results_local_dir = os.path.join(RESOURCES_RESULTS_DIR, experiment_name)
    os.makedirs(results_local_dir, exist_ok=True)
    results_local_path = os.path.join(results_local_dir, f'results.csv')
    res_train = pd.DataFrame()
    res_val = pd.DataFrame()
    res_test = pd.DataFrame()
    res_epsilon = pd.DataFrame()
    targeted = attack_direction == 'targeted'
    for epsilon in args.epsilons:
        print(f'attack with epsilon {epsilon}/255...')
        if attack_train:
            # attack train images
            res_train = attacker(experiment_name, dataset, attack, pcld, cld, run_naive_attack,
                                 loaders['train'][1], 'train', epsilon, targeted, output_every, n_classes,
                                 classes, attack_nb_iter, device)
            # attack validation images
            res_val = attacker(experiment_name, dataset, attack, pcld, cld, run_naive_attack,
                               loaders['val'][1], 'val', epsilon, targeted, output_every, n_classes,
                               classes, attack_nb_iter, device)
        # attack test images
        res_test = attacker(experiment_name, dataset, attack, pcld, cld, run_naive_attack,
                            loaders['test'][1], 'test', epsilon, targeted, output_every, n_classes,
                            classes, attack_nb_iter, device)
        res_epsilon = pd.concat([res_epsilon, res_train, res_val, res_test], ignore_index=True, axis=0)
        print(f'save results...')
        res_epsilon.to_csv(results_local_path, index=False)
        print(f'finished attack with epsilon {epsilon}/255!')
