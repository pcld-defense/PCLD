import time

import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from art.attacks.evasion import CarliniLInfMethod, AutoAttack
from art.estimators.classification import PyTorchClassifier


def attack_batch(model, x, attack, epsilon, attack_nb_iter, targeted, y_classes_targeted,
                 n_classes, device):
    if epsilon == 0:
        return x
    if attack == 'fgsm':
        x_adv = fast_gradient_method(model_fn=model,
                                     x=x,
                                     eps=epsilon,
                                     norm=np.inf,
                                     y=y_classes_targeted,
                                     targeted=targeted,
                                     clip_min=0,
                                     clip_max=1)
    elif attack == 'pgd':
        x_adv = projected_gradient_descent(model_fn=model,
                                           x=x,
                                           eps=epsilon,
                                           eps_iter=epsilon/attack_nb_iter,
                                           nb_iter=attack_nb_iter,
                                           norm=np.inf,
                                           y=y_classes_targeted,
                                           targeted=targeted,
                                           rand_init=False,
                                           sanity_checks=False,
                                           clip_min=0,
                                           clip_max=1)
    elif attack in ('aa', 'cw'):
        # Define the loss function and the optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # Wrap the model with ART's PyTorchClassifier
        classifier = PyTorchClassifier(
            model=model,
            loss=criterion,
            optimizer=optimizer,
            input_shape=x.shape[1:],
            nb_classes=7,
            clip_values=(0.0, 1.0)
        )

        # Initialize AutoAttack
        if attack == 'aa':
            attack = AutoAttack(estimator=classifier,
                                eps=epsilon,
                                norm=np.inf,
                                parallel=False)

        elif attack == 'cw':
            attack = CarliniLInfMethod(classifier=classifier,
                                       targeted=targeted,
                                       max_iter=attack_nb_iter,
                                       batch_size=x.shape[0],
                                       confidence=0.0,
                                       #  verbose=False
                                       )

        # Convert PyTorch tensor to NumPy array
        x_np = x.detach().cpu().numpy()
        # Perform the attack
        x_adv = attack.generate(x=x_np)
        x_adv = torch.tensor(x_adv, device=device)

    return x_adv


def attacker(experiment, dataset, attack, adaptive_model, naive_model, run_naive_attack, loader, phase,
             epsilon, targeted, output_every, n_classes, classes, attack_nb_iter, device,
             output_type='final_decision'):
    print(f'run attacks on {phase}...')
    epsilon_real = epsilon / 255.
    output_every_expanded = output_every + [999999]  # add the original image paint step = ∞
    if output_type == 'final_decision':
        output_every_expanded = [-1]
    paint_steps = len(output_every_expanded)
    results_dict = {
        'experiment': [],
        'dataset': [], 'image': [], 't': [], 'phase': [], 'attacked_model': [], 'defense_model': [],
        'attack': [], 'targeted': [], 'targeted_jumps_allowed': [], 'targeted_label': [], 'norm': [], 'epsilon': [],
        'nb_iter': [], 'actual': [], 'actual_class': [], 'pred': [], 'pred_class': [],
        f'prob_{classes[0]}': [],
        f'prob_{classes[1]}': [],
        f'prob_{classes[2]}': [],
        f'prob_{classes[3]}': [],
        f'prob_{classes[4]}': [],
        f'prob_{classes[5]}': [],
        f'prob_{classes[6]}': [],
        'attack_time_sec_avg': [],
        'defense_time_sec_avg': []
    }
    targeted_jumps_allowed = 6 if targeted else 1  # make sure to attack to all direction for each image

    for i, data in enumerate(loader, 0):
        print(f'batch {i} attack...')
        x, y, paths = data[0].to(device), data[1].to(device), data[2]
        img_names = [p.split('/')[-1].split('.')[0] for p in paths]
        y_classes = [yi.item() for yi in y]
        # generate targeted labels for adv attack
        y_classes_targeted = [int((yi.item()+random.randint(1, targeted_jumps_allowed)) % len(classes)) for yi in y]
        # y_classes_targeted = [int((yi.item()+ 1) % len(classes)) for yi in y]
        y_classes_targeted_adaptive = y_classes_targeted_naive = torch.Tensor(y_classes_targeted).to(torch.long).to(device)
        y_classes_adaptive = y_classes_naive = torch.Tensor(y_classes).to(torch.long).to(device)

        if output_type == 'paints_inference':
            y_classes_targeted_adaptive = y_classes_targeted_naive.repeat_interleave(len(output_every_expanded))
            y_classes_adaptive = y_classes_naive.repeat_interleave(len(output_every_expanded))

        print(f'attack naïve attack for BPDA validation')
        start_time = time.time()
        x_adv_naive = x
        if run_naive_attack:
            x_adv_naive = attack_batch(naive_model, x, attack, epsilon_real,
                                       attack_nb_iter, targeted,
                                       y_classes_targeted_naive if targeted else y_classes_naive,
                                       n_classes, device)
        end_time = time.time()
        attack_naive_time = end_time - start_time
        print(f'attack adaptive BPDA model with surrogate model')
        start_time = time.time()
        x_adv_adaptive = attack_batch(adaptive_model, x, attack, epsilon_real,
                                      attack_nb_iter, targeted,
                                      y_classes_targeted_adaptive if targeted else y_classes_adaptive,
                                      n_classes, device)


        end_time = time.time()
        attack_adaptive_time = end_time - start_time

        print(f'defend against both attacks using the model')
        start_time = time.time()
        adv_naive_probs = torch.softmax(adaptive_model(x_adv_naive), dim=1).tolist()
        adv_adaptive_probs = torch.softmax(adaptive_model(x_adv_adaptive), dim=1).tolist()
        end_time = time.time()
        defense_adaptive_time = (end_time - start_time) / 2
        adv_naive_decisions = np.argmax(adv_naive_probs, axis=1).tolist()
        adv_adaptive_decisions = np.argmax(adv_adaptive_probs, axis=1).tolist()

        print(f'saving batch {i} results')
        batch_size = len(img_names)
        n = len(adv_naive_decisions)
        y_classes_names = [classes[y_classes[c]] for c in range(len(y_classes))]
        y_pred_classes_names_naive = [classes[adv_naive_decisions[c]] for c in range(len(adv_naive_decisions))]
        y_pred_classes_names_adaptive = [classes[adv_adaptive_decisions[c]] for c in range(len(adv_adaptive_decisions))]
        results_dict['experiment'].extend([experiment] * n * 2)
        results_dict['dataset'].extend([dataset] * n * 2)
        results_dict['image'].extend(np.repeat(img_names, paint_steps).tolist() * 2)
        results_dict['t'].extend(output_every_expanded * batch_size * 2)
        results_dict['phase'].extend([phase] * n * 2)
        results_dict['attacked_model'].extend((['naive'] * n) + (['adaptive'] * n))
        results_dict['defense_model'].extend(['adaptive'] * n * 2)
        results_dict['attack'].extend([attack] * n * 2)
        results_dict['targeted'].extend([int(targeted)] * n * 2)
        results_dict['targeted_jumps_allowed'].extend([int(targeted_jumps_allowed)] * n * 2)
        results_dict['targeted_label'].extend(np.repeat(y_classes_targeted, paint_steps).tolist() * 2)
        results_dict['norm'].extend(['linf'] * n * 2)
        results_dict['epsilon'].extend([int(epsilon)] * n * 2)
        results_dict['nb_iter'].extend([attack_nb_iter] * n * 2)
        results_dict['actual'].extend(np.repeat(y_classes, paint_steps).tolist() * 2)
        results_dict['actual_class'].extend(np.repeat(y_classes_names, paint_steps).tolist() * 2)
        results_dict['pred'].extend(adv_naive_decisions + adv_adaptive_decisions)
        results_dict['pred_class'].extend(y_pred_classes_names_naive + y_pred_classes_names_adaptive)
        for pi in range(len(classes)):
            results_dict[f'prob_{classes[pi]}'].extend(np.array(adv_naive_probs)[:, pi].tolist() + \
                                                       np.array(adv_adaptive_probs)[:, pi].tolist())
        results_dict['attack_time_sec_avg'].extend(([attack_naive_time/batch_size] * n) + ([attack_adaptive_time/batch_size] * n))
        results_dict['defense_time_sec_avg'].extend(([defense_adaptive_time/batch_size] * n) + ([defense_adaptive_time/batch_size] * n))

        # Validate lengths
        len_validate = len(results_dict['experiment'])
        for k in results_dict.keys():
            if len(results_dict[k]) != len_validate:
                raise Exception(f'ValueError: All arrays must be of the same length!!!!!')
                exit(1)
        print(f'finished attacking batch {i}!')

    results_df = pd.DataFrame(results_dict)
    print(f'finished attacking {phase}!')
    return results_df

