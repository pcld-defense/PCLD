import os

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import torch

from model.decisioner import Decisioner1DConv, DecisionerFC
from util.consts import RESOURCES_MODELS_DIR
from util.evaluations import deep_evaluation_training, evaluate_print, evaluate_print_decisioner


def load_model(model, path, device):
    state_dict = torch.load(path, map_location=torch.device(device))
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    return model


def get_best_epoch(res_df, epoch):
    if len(res_df) == 0:
        return epoch, 0, 0
    val_res_df = res_df[res_df['ds_type'] == 'validation']
    val_res_df = val_res_df.sort_values(by='epoch').reset_index()
    best_iter_idx = val_res_df['avg_loss'].idxmin()
    best_epoch = val_res_df.loc[best_iter_idx, 'epoch']
    best_loss = val_res_df.loc[best_iter_idx, 'avg_loss']
    best_acc = val_res_df.loc[best_iter_idx, 'accuracy']
    return best_epoch, best_loss, best_acc


def process_epoch_clf(experiment, device, epoch, net, loader, loader_name, n_batches,
                      criterion, optimizer,
                      results_df, n_classes, classes, is_train=True,
                      phase='train', save_model=True,
                      deep_evaluate=False, results_deep_df=None, scheduler=None):
    total_loss = 0.0
    if is_train:
        net.train()
    else:
        net.eval()
    class_correct = list(0. for i in range(n_classes))
    class_total = list(0. for i in range(n_classes))

    for i, data in enumerate(loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        if 'cuda' in device:
            inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        images_paths = data[2]
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        if is_train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        # compare predictions to true label
        correct_tensor = pred.eq(labels.data.view_as(pred))
        correct = correct_tensor.numpy() if device == 'cpu' else correct_tensor.cpu().numpy()
        for j in range(len(labels.data)):
            label = labels.data[j]
            class_correct[label] += correct[j].item()
            class_total[label] += 1

        if deep_evaluate:
            results_deep_df = deep_evaluation_training(experiment,
                                                       results_deep_df,
                                                       epoch,
                                                       loader_name,
                                                       phase,
                                                       loader_name,
                                                       n_classes,
                                                       classes,
                                                       images_paths,
                                                       outputs,
                                                       labels,
                                                       criterion)

    if is_train and scheduler:
        scheduler.step()

    results_df = evaluate_print(experiment=experiment,
                                res_df=results_df,
                                class_correct=class_correct,
                                class_total=class_total,
                                loss=total_loss,
                                epoch=epoch,
                                ds_type=phase,
                                dataset_size=n_batches,
                                loader_name=loader_name,
                                n_classes=n_classes,
                                classes=classes
                                )

    if save_model:
        # remove all prev models in this dir
        save_dir = os.path.join(RESOURCES_MODELS_DIR, experiment)
        os.makedirs(save_dir, exist_ok=True)
        # [os.remove(os.path.join(save_dir, d)) for d in os.listdir(save_dir)]
        # save the updated model
        save_path = f'{save_dir}/model.pth'
        torch.save(net.state_dict(), save_path)

    return results_df, results_deep_df


def prepare_torch_ds_decisioner(df, p_steps, prob_cols, target_col, architecture,
                     epsilons_weights, device):
    bys = ['experiment', 'targeted', 'image', 'attack', 'epsilon']
    df['idx'] = df[bys].astype(str).agg('-'.join, axis=1)
    df['sample_idx'], _ = pd.factorize(df['idx'])
    df.drop('idx', axis=1, inplace=True)
    x = torch.tensor(df[prob_cols].values, device=device, dtype=torch.float32)
    if architecture == 'conv':
        x = x.view(-1, p_steps, len(prob_cols))
    else:  # fc
        x = x.reshape(-1, p_steps*len(prob_cols))
    df_target = df.groupby('sample_idx')[target_col].max().reset_index()
    y = torch.tensor(df_target[target_col].values, device=device,
                     dtype=torch.long)
    indices = torch.tensor(df_target['sample_idx'].values,
                           device=device, dtype=torch.long)
    epsilons = df.groupby(bys[:-1])['epsilon'].unique().explode().tolist()
    sample_weights = torch.tensor([epsilons_weights[ep] for ep in epsilons], device=device)
    epsilons = torch.tensor(epsilons, device=device, dtype=torch.long)
    return x, y, indices, epsilons, sample_weights, df



def process_epoch_decisioner(model, epoch, loader, dataset_size, n_classes, names_classes, device,
                             optimizer, criterion, is_train,
                             epsilons_all):
    total_loss = 0
    n_epsilons = len(epsilons_all)
    epsilon_stats = {epsilon: [0, 0] for epsilon in epsilons_all}
    class_correct = list(0. for i in range(n_classes))
    class_total = list(0. for i in range(n_classes))
    indices = []
    y_pred = []
    y_prob = []
    y_actual = []
    if is_train:
        model.train()
    else:
        model.eval()
    # run traing epoch
    for i, data in enumerate(loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        indices.extend(list(data[2].cpu().numpy()))
        epsilons_by_sample = list(data[3].cpu().numpy())
        if 'cuda' in device:
            inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        if is_train:
            optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        if is_train:
            sample_weights = data[4]
            loss = (loss * sample_weights).mean()
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        _, pred = torch.max(outputs, 1)

        y_actual.extend(list(labels.data.view_as(pred).detach().cpu().numpy()))
        y_pred.extend(list(pred.detach().cpu().numpy()))
        y_prob.extend(list(torch.softmax(outputs, dim=-1).detach().cpu().numpy()))

        correct_tensor = pred.eq(labels.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if device == 'cpu' \
            else np.squeeze(correct_tensor.cpu().numpy())
        for j in range(len(labels.data)):
            label = labels.data[j]
            epsilon_stats[epsilons_by_sample[j]][0] += correct[j].item()
            epsilon_stats[epsilons_by_sample[j]][1] += 1
            class_correct[label] += correct[j].item()
            class_total[label] += 1
    evaluate_print_decisioner(class_correct, class_total, total_loss, epoch,
                              dataset_size, n_classes, names_classes,
                              epsilon_stats)

    return y_actual, y_pred, y_prob, indices


def trainer_decisioner(decisioner_architechture, batch_size, max_epochs, find_best_epoch,
                       df_train, df_val, df_train_full, df_test, paint_steps, prob_cols, epsilons_weights,
                       n_classes, names_classes, epsilons, device):

    print('prepare dataset for training')
    bys = ['experiment', 'targeted', 'image', 'epsilon', 't']
    df_train = df_train.sort_values(by=bys)
    df_val = df_val.sort_values(by=bys)
    df_train_full = df_train_full.sort_values(by=bys)
    df_test = df_test.sort_values(by=bys)

    x_train, y_train, indices_train, epsilons_train, sample_weights_train, df_train = \
        prepare_torch_ds_decisioner(df_train, paint_steps, prob_cols, 'actual',
                         decisioner_architechture, epsilons_weights, device)
    x_val, y_val, indices_val, epsilons_val, sample_weights_val, df_val = \
        prepare_torch_ds_decisioner(df_val, paint_steps, prob_cols, 'actual',
                         decisioner_architechture, epsilons_weights, device)
    x_train_full, y_train_full, indices_train_full, epsilons_train_full, sample_weights_train_full, df_train_full = \
        prepare_torch_ds_decisioner(df_train_full, paint_steps, prob_cols, 'actual',
                         decisioner_architechture, epsilons_weights, device)
    x_test, y_test, indices_test, epsilons_test, sample_weights_test, df_test = \
        prepare_torch_ds_decisioner(df_test, paint_steps, prob_cols, 'actual',
                         decisioner_architechture, epsilons_weights, device)

    train_dataset = TensorDataset(x_train, y_train, indices_train, epsilons_train, sample_weights_train)
    val_dataset = TensorDataset(x_val, y_val, indices_val, epsilons_val, sample_weights_val)
    train_full_dataset = TensorDataset(x_train_full, y_train_full, indices_train_full, epsilons_train_full,
                                       sample_weights_train_full)
    test_dataset = TensorDataset(x_test, y_test, indices_test, epsilons_test, sample_weights_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    train_full_loader = DataLoader(train_full_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    best_epoch = max_epochs
    if find_best_epoch:
        print(f'\nrun train_validate phase to find the best epoch\n')
        print(f'load the decisioner model')
        decisioner = Decisioner1DConv(n_classes, paint_steps, 32).to(device)
        if decisioner_architechture == 'fc':
            decisioner = DecisionerFC(n_classes, paint_steps).to(device)
        criterion_train = nn.CrossEntropyLoss(reduction='none')  # for sample weight
        criterion_test = nn.CrossEntropyLoss()
        optimizer = optim.SGD(decisioner.parameters(), lr=0.01, weight_decay=0.001)
        best_acc_val = 0
        for epoch in range(0, max_epochs):
            print(f'train_validate: start epoch {epoch}')
            # Train
            print(f'process epoch {epoch} on train')
            y_actual_train, y_pred_train, y_prob_train, indices_train = \
                process_epoch_decisioner(decisioner, epoch,
                                         train_loader, len(train_loader),
                                         n_classes, names_classes, device, optimizer, criterion_train, True,
                                         epsilons
                                         )
            # Validation
            print(f'process epoch {epoch} on validation')
            with torch.no_grad():
                y_actual_val, y_pred_val, y_prob_val, indices_val = \
                    process_epoch_decisioner(decisioner, epoch,
                                             val_loader, len(val_loader),
                                             n_classes, names_classes, device, optimizer, criterion_test, False,
                                             epsilons
                                             )

            is_correct_val = [int(a == b) for a, b in zip(y_actual_val, y_pred_val)]
            acc_val = sum(is_correct_val)/len(is_correct_val)
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                best_epoch = epoch
            elif epoch - best_epoch >= 20:
                print(f'Training had been stopped by OD. Best epoch {best_epoch} ' +
                      f'Best validation accuracy: {best_acc_val} ')
                break

    print(f'\nrun train_full_test phase\n')
    print(f'load the decisioner model')
    decisioner = Decisioner1DConv(n_classes, paint_steps, 32).to(device)
    if decisioner_architechture == 'fc':
        decisioner = DecisionerFC(n_classes, paint_steps).to(device)
    criterion_train = nn.CrossEntropyLoss(reduction='none')  # for sample weight
    criterion_test = nn.CrossEntropyLoss()
    optimizer = optim.SGD(decisioner.parameters(), lr=0.01, weight_decay=0.001)
    for epoch in range(0, best_epoch):
        # Train Full
        print(f'process epoch {epoch} on train_full')
        y_actual_train_full, y_pred_train_full, y_prob_train_full, indices_train_full = \
            process_epoch_decisioner(decisioner, epoch,
                                     train_full_loader, len(train_full_loader),
                                     n_classes, names_classes, device, optimizer, criterion_train, True,
                                     epsilons
                                     )
        # Test
        print(f'process epoch {epoch} on test')
        with torch.no_grad():
            y_actual_test, y_pred_test, y_prob_test, indices_test = \
                process_epoch_decisioner(decisioner, epoch,
                                         test_loader, len(test_loader),
                                         n_classes, names_classes, device, optimizer, criterion_test, False,
                                         epsilons
                                         )

    print(f'training {decisioner_architechture} model finished!')

    return df_train_full, df_test, decisioner, \
        y_actual_train_full, y_pred_train_full, y_prob_train_full, indices_train_full, \
        y_actual_test, y_pred_test, y_prob_test, indices_test


def get_predictions_df_decisioner(actuals, preds, probs, indices, names_classes):
    predictions = {'sample_idx': indices,
                   'actual': actuals,
                   'pred': preds,
                   'pred_class': [names_classes[p] for p in preds]
                   }
    prob_matrix = np.array(probs)
    for idx, label in enumerate(names_classes):
        predictions[f"prob_{label}"] = prob_matrix[:, idx].tolist()
    predictions = pd.DataFrame(predictions)
    return predictions


def arange_results_decisioner(df_preds_train_full, df_preds_test, df_tfl, df_tst, experiment):
    merge_cols = ['sample_idx', 'image', 'experiment', 'attacked_model', 'defense_model', 'attack', 'targeted',
                  'targeted_jumps_allowed', 'targeted_label', 'norm', 'epsilon', 'nb_iter', 'actual_class'
                  ]

    df_preds_train_full = df_preds_train_full.merge(df_tfl[merge_cols].drop_duplicates(), on='sample_idx', how='left')
    df_preds_test = df_preds_test.merge(df_tst[merge_cols].drop_duplicates(), on='sample_idx', how='left')
    df_preds_train_full['phase'] = 'train_full'
    df_preds_test['phase'] = 'test'
    df_res = pd.concat([df_preds_train_full, df_preds_test], axis=0, ignore_index=True)

    df_res['t'] = 'Decisioner'
    df_res.rename(columns={'experiment': 'dataset'}, inplace=True)
    df_res['experiment'] = experiment
    df_res['attack_time_sec_avg'] = 0
    df_res['defense_time_sec_avg'] = 0
    cols_ordered = [c for c in df_tfl.columns if c in df_res.columns]
    df_res = df_res[cols_ordered]  # arrange in the same order
    return df_res