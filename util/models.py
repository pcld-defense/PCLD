import os
from typing import Optional

import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import torch

from model.decisioner import Decisioner1DConv, DecisionerFC, DecisionerStepAttention
from util.consts import RESOURCES_MODELS_DIR
from util.evaluations import evaluate_print, evaluate_print_decisioner, plot_loss_and_acc, plot_pcl_accuracy_vs_epsilon


def load_model(model: torch.nn.Module, path: str, device: str) -> torch.nn.Module:
    """Loads a checkpoint into a model and moves it to the target device.

    Strips the 'module.' prefix from state-dict keys that is added by
    torch.nn.DataParallel, so checkpoints saved with or without DataParallel
    can be loaded transparently.

    Args:
        model: Model architecture whose weights will be replaced.
        path: Path to the saved .pth or .pt checkpoint file.
        device: Target device string (e.g. 'cpu' or 'cuda').

    Returns:
        The model instance with loaded weights moved to `device`.
    """
    state_dict = torch.load(path, map_location=torch.device(device))
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)

    return model


def save_best_cls_model(net: torch.nn.Module, experiment: str, epoch: int,
                        val_loss: float, best_val_loss: float) -> float:
    """Saves the model if the current validation loss is the best seen so far.

    Writes the model state dict to resources/models/<experiment>/best_model.pth
    only when val_loss improves upon best_val_loss.

    Args:
        net: The classifier model whose state dict should be saved.
        experiment: Experiment name used as the subdirectory in RESOURCES_MODELS_DIR.
        epoch: Current epoch index (logged in the success message).
        val_loss: Validation loss for the current epoch.
        best_val_loss: Best validation loss observed across all previous epochs.

    Returns:
        The new best validation loss (val_loss if improved, best_val_loss otherwise).
    """
    save_dir = os.path.join(RESOURCES_MODELS_DIR, experiment)
    os.makedirs(save_dir, exist_ok=True)

    # Save as "best_model.pth" if validation loss improved
    if val_loss < best_val_loss:
        save_path = f'{save_dir}/best_model.pth'
        torch.save(net.state_dict(), save_path)
        print(f"--- Validation loss improved. Best model saved at epoch {epoch} ---")
        return val_loss  # New best_val_loss
    return best_val_loss


def get_best_epoch(res_df: pd.DataFrame, epoch: int) -> tuple:
    """Finds the validation epoch with the lowest average loss.

    Filters the results DataFrame to validation rows and returns the epoch
    index, loss, and accuracy of the best checkpoint. Falls back to the
    current epoch with zero loss/accuracy when the DataFrame is empty.

    Args:
        res_df: Results DataFrame containing at least 'ds_type', 'epoch',
            'avg_loss', and 'accuracy' columns.
        epoch: Current epoch index used as a fallback when res_df is empty.

    Returns:
        Tuple of (best_epoch, best_loss, best_accuracy).
    """
    if len(res_df) == 0:
        return epoch, 0, 0

    val_res_df = res_df[res_df['ds_type'] == 'validation']
    val_res_df = val_res_df.sort_values(by='epoch').reset_index()
    best_iter_idx = val_res_df['avg_loss'].idxmin()
    best_epoch = val_res_df.loc[best_iter_idx, 'epoch']
    best_loss = val_res_df.loc[best_iter_idx, 'avg_loss']
    best_acc = val_res_df.loc[best_iter_idx, 'accuracy']

    return best_epoch, best_loss, best_acc


def _update_class_stats(class_correct: list, class_total: list,
                        pred: torch.Tensor, labels: torch.Tensor) -> None:
    """Accumulates per-class correct-prediction and sample counts in-place.

    Args:
        class_correct: Running correct-prediction counts per class.
        class_total: Running sample counts per class.
        pred: Predicted class indices of shape (B,).
        labels: Ground-truth class indices of shape (B,).
    """
    correct = pred.eq(labels).cpu()
    for label, c in zip(labels.cpu().tolist(), correct.tolist()):
        class_correct[label] += int(c)
        class_total[label] += 1


def process_epoch_clf(experiment: str, device: str, epoch: int,
                      net: torch.nn.Module, loader: torch.utils.data.DataLoader,
                      loader_name: str,
                      criterion: torch.nn.Module,
                      optimizer: Optional[torch.optim.Optimizer],
                      results_df: pd.DataFrame, n_classes: int, classes: list[str],
                      is_train: bool = True, phase: str = 'train',
                      save_model: bool = True,
                      scheduler=None) -> pd.DataFrame:
    """Runs one training or evaluation epoch for the classifier.

    Iterates over all batches in `loader`, computes predictions, accumulates
    per-class correct/total counts, optionally backpropagates, and saves the
    model checkpoint after each epoch. Appends the epoch metrics to
    `results_df` via evaluate_print.

    Args:
        experiment: Experiment name forwarded to evaluate_print and used as
            the checkpoint subdirectory.
        device: Device string for moving data (e.g. 'cuda').
        epoch: Current epoch index.
        net: The classifier model.
        loader: DataLoader yielding (inputs, labels) or (inputs, labels, paths).
        loader_name: Human-readable split name used in log messages.
        criterion: Classification loss function.
        optimizer: Optimiser updated when is_train=True; may be None when
            is_train=False since it is never accessed during eval.
        results_df: Existing results DataFrame to append to.
        n_classes: Number of output classes.
        classes: Sorted list of class name strings.
        is_train: If True, runs the forward+backward+step training loop;
            otherwise runs in eval mode without gradient updates.
        phase: Split label stored in the results row (e.g. 'train' or
            'validation').
        save_model: If True, saves the current model state dict after the
            epoch completes.
        scheduler: Optional LR scheduler stepped at the end of each training
            epoch.

    Returns:
        Updated results DataFrame with the current epoch metrics appended.
    """
    total_loss = 0.0
    class_correct = [0] * n_classes
    class_total = [0] * n_classes

    net.train() if is_train else net.eval()

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for data in loader:
            inputs = data[0].to(device, non_blocking=True)
            labels = data[1].to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            _update_class_stats(class_correct, class_total, pred, labels)

    if is_train and scheduler:
        scheduler.step()

    results_df = evaluate_print(experiment=experiment,
                                res_df=results_df,
                                class_correct=class_correct,
                                class_total=class_total,
                                loss=total_loss,
                                epoch=epoch,
                                ds_type=phase,
                                dataset_size=len(loader),
                                loader_name=loader_name,
                                n_classes=n_classes,
                                classes=classes)

    if save_model:
        save_dir = os.path.join(RESOURCES_MODELS_DIR, experiment)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(net.state_dict(), f'{save_dir}/model.pth')

    return results_df


def prepare_torch_ds_decisioner(df: pd.DataFrame, p_steps: int,
                                target_col: str, architecture: str,
                                epsilons_weights: dict, device: str) -> tuple:
    """Converts a decisioner results DataFrame into tensors for PyTorch training.

    Groups rows by unique (experiment, targeted, image, attack, epsilon) tuples
    to assign a sample index, stacks the `probs` array column into an input
    tensor shaped for either 1D-conv or FC architecture, and computes per-sample
    epsilon weights.

    Args:
        df: DataFrame with one row per (image × paint step × attack). Must
            contain a `probs` list column, the grouping columns, and `target_col`.
        p_steps: Number of paint steps (temporal sequence length).
        target_col: Name of the column holding the ground-truth class index.
        architecture: 'conv' to keep (B, p_steps, n_classes) shape, 'fc' to
            flatten to (B, p_steps * n_classes).
        epsilons_weights: Dict mapping epsilon int → sample weight scalar.
        device: Target device string for tensor creation.

    Returns:
        Tuple of (x, y, indices, epsilons, sample_weights, df) where:
            x: Feature tensor of shape (B, p_steps, n_classes) or
               (B, p_steps * n_classes).
            y: Label tensor of shape (B,).
            indices: Sample-index tensor of shape (B,).
            epsilons: Epsilon-per-sample tensor of shape (B,).
            sample_weights: Weight tensor of shape (B,).
            df: Input DataFrame with added 'sample_idx' column.
    """
    bys = ['experiment', 'targeted', 'image', 'attack', 'epsilon']
    df['idx'] = df[bys].astype(str).agg('-'.join, axis=1)
    df['sample_idx'], _ = pd.factorize(df['idx'])
    df.drop('idx', axis=1, inplace=True)
    x = torch.tensor(np.stack(df['probs'].values), device=device, dtype=torch.float32)
    n_classes = x.shape[-1]
    if architecture == 'conv':
        x = x.view(-1, p_steps, n_classes)
    else:  # fc
        x = x.reshape(-1, p_steps * n_classes)
    df_target = df.groupby('sample_idx')[target_col].max().reset_index()
    y = torch.tensor(df_target[target_col].values, device=device,
                     dtype=torch.long)
    indices = torch.tensor(df_target['sample_idx'].values,
                           device=device, dtype=torch.long)
    epsilons = df.groupby(bys[:-1])['epsilon'].unique().explode().tolist()
    sample_weights = torch.tensor([epsilons_weights[ep] for ep in epsilons], device=device)
    epsilons = torch.tensor(epsilons, device=device, dtype=torch.long)
    return x, y, indices, epsilons, sample_weights, df


def process_epoch_decisioner(model: torch.nn.Module, epoch: int,
                             loader: torch.utils.data.DataLoader,
                             dataset_size: int, n_classes: int,
                             names_classes: list[str], device: str,
                             optimizer: torch.optim.Optimizer,
                             criterion: torch.nn.Module, is_train: bool,
                             epsilons_all: list[int]) -> tuple:
    """Runs one training or evaluation epoch for the decisioner.

    Iterates over batches of (confidence trajectories, labels, sample indices,
    epsilons, sample weights), computes the decisioner output, and optionally
    backpropagates with per-sample loss weighting. Per-epsilon accuracy stats
    are tracked alongside global class accuracy.

    Args:
        model: Decisioner1DConv or DecisionerFC model.
        epoch: Current epoch index (passed to evaluate_print_decisioner).
        loader: DataLoader yielding 5-element tuples:
            (x, y, sample_indices, epsilons_per_sample, sample_weights).
        dataset_size: Number of batches (used as loss denominator).
        n_classes: Number of output classes.
        names_classes: Sorted list of class name strings.
        device: Device string for moving data.
        optimizer: Optimiser updated during training epochs.
        criterion: Loss function; should use reduction='none' for training
            (to apply per-sample weights) and default reduction for eval.
        is_train: If True, backpropagation and optimiser step are performed.
        epsilons_all: Sorted list of all unique epsilon values, used to
            initialise the per-epsilon accuracy tracking dict.

    Returns:
        Tuple of (y_actual, y_pred, y_prob, indices, avg_loss, accuracy) where
        the first four are per-sample lists and the last two are scalar epoch
        metrics used for training-curve tracking.
    """
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
    avg_loss, accuracy = evaluate_print_decisioner(class_correct, class_total, total_loss, epoch,
                                                   dataset_size, n_classes, names_classes,
                                                   epsilon_stats)

    return y_actual, y_pred, y_prob, indices, avg_loss, accuracy


def trainer_decisioner(decisioner_architechture: str, batch_size: int,
                       max_epochs: int, find_best_epoch: int,
                       df_train: pd.DataFrame, df_val: pd.DataFrame,
                       df_train_full: pd.DataFrame, df_test: pd.DataFrame,
                       paint_steps: int, epsilons_weights: dict, n_classes: int,
                       names_classes: list[str], epsilons: list[int],
                       device: str, output_dir: str) -> tuple:
    """Full decisioner training pipeline with optional early-stopping epoch search.

    Two-phase training strategy:
    1. (Optional, when find_best_epoch=1) Train on df_train / validate on
       df_val with early stopping (patience=20) to find the best epoch count.
    2. Re-train from scratch on df_train_full (train + val) for best_epoch
       epochs and evaluate on df_test.

    Loss and accuracy curves for each phase are saved as PNG files to
    `output_dir` (phase1_average_loss.png, phase2_accuracy_per_epoch.png, etc.).

    Args:
        decisioner_architechture: 'conv' for Decisioner1DConv or 'fc' for
            DecisionerFC.
        batch_size: Mini-batch size for all loaders.
        max_epochs: Maximum number of training epochs.
        find_best_epoch: When non-zero, Phase 1 is executed to discover the
            optimal epoch count before Phase 2 training.
        df_train: Training split DataFrame.
        df_val: Validation split DataFrame.
        df_train_full: Combined train + val DataFrame used in Phase 2.
        df_test: Test split DataFrame.
        paint_steps: Number of paint steps (temporal sequence length).
        epsilons_weights: Dict mapping epsilon → sample weight for weighted loss.
        n_classes: Number of output classes.
        names_classes: Sorted list of class name strings.
        epsilons: Sorted list of all unique epsilon values.
        device: Target device string.
        output_dir: Directory where training-curve plots are saved.

    Returns:
        Tuple of (df_train_full, df_test, decisioner,
                  y_actual_train_full, y_pred_train_full, y_prob_train_full,
                  indices_train_full,
                  y_actual_test, y_pred_test, y_prob_test, indices_test).
    """
    print('prepare dataset for training')
    bys = ['experiment', 'targeted', 'image', 'epsilon', 't']
    df_train = df_train.sort_values(by=bys)
    df_val = df_val.sort_values(by=bys)
    df_train_full = df_train_full.sort_values(by=bys)
    df_test = df_test.sort_values(by=bys)

    x_train, y_train, indices_train, epsilons_train, sample_weights_train, df_train = \
        prepare_torch_ds_decisioner(df_train, paint_steps, 'actual',
                                    decisioner_architechture, epsilons_weights, device)
    x_val, y_val, indices_val, epsilons_val, sample_weights_val, df_val = \
        prepare_torch_ds_decisioner(df_val, paint_steps, 'actual',
                                    decisioner_architechture, epsilons_weights, device)
    x_train_full, y_train_full, indices_train_full, epsilons_train_full, sample_weights_train_full, df_train_full = \
        prepare_torch_ds_decisioner(df_train_full, paint_steps, 'actual',
                                    decisioner_architechture, epsilons_weights, device)
    x_test, y_test, indices_test, epsilons_test, sample_weights_test, df_test = \
        prepare_torch_ds_decisioner(df_test, paint_steps, 'actual',
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

    ckpt_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    best_epoch = max_epochs
    if find_best_epoch:
        print(f'\nrun train_validate phase to find the best epoch\n')
        print(f'load the decisioner model')
        decisioner = Decisioner1DConv(n_classes, paint_steps, 32).to(device)
        if decisioner_architechture == 'fc':
            decisioner = DecisionerFC(n_classes, paint_steps).to(device)
        elif decisioner_architechture == 'attn':
            decisioner = DecisionerStepAttention(n_classes, paint_steps, 32).to(device)
        criterion_train = nn.CrossEntropyLoss(reduction='none')  # for sample weight
        criterion_test = nn.CrossEntropyLoss()
        optimizer = optim.Adam(decisioner.parameters(), lr=1e-3, weight_decay=1e-4)
        best_acc_val = 0
        phase1_history: list[dict] = []
        for epoch in range(0, max_epochs):
            print(f'train_validate: start epoch {epoch}')
            # Train
            print(f'process epoch {epoch} on train')
            y_actual_train, y_pred_train, y_prob_train, indices_train, train_loss, train_acc = \
                process_epoch_decisioner(decisioner, epoch,
                                         train_loader, len(train_loader),
                                         n_classes, names_classes, device, optimizer, criterion_train, True,
                                         epsilons
                                         )
            phase1_history.append({'ds_type': 'train', 'epoch': epoch,
                                    'avg_loss': train_loss, 'accuracy': train_acc})
            # Validation
            print(f'process epoch {epoch} on validation')
            with torch.no_grad():
                y_actual_val, y_pred_val, y_prob_val, indices_val, val_loss, val_acc = \
                    process_epoch_decisioner(decisioner, epoch,
                                             val_loader, len(val_loader),
                                             n_classes, names_classes, device, optimizer, criterion_test, False,
                                             epsilons
                                             )
            phase1_history.append({'ds_type': 'validation', 'epoch': epoch,
                                    'avg_loss': val_loss, 'accuracy': val_acc})

            torch.save(decisioner.state_dict(),
                       os.path.join(ckpt_dir, f'phase1_epoch_{epoch}.pth'))

            is_correct_val = [int(a == b) for a, b in zip(y_actual_val, y_pred_val)]
            acc_val = sum(is_correct_val) / len(is_correct_val)
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                best_epoch = epoch
            elif epoch - best_epoch >= 20:
                print(f'Training had been stopped by OD. Best epoch {best_epoch} ' +
                      f'Best validation accuracy: {best_acc_val} ')
                break

        os.makedirs(output_dir, exist_ok=True)
        plot_loss_and_acc(pd.DataFrame(phase1_history), output_dir, prefix='phase1_')

    print(f'\nrun train_full_test phase\n')
    print(f'load the decisioner model')
    decisioner = Decisioner1DConv(n_classes, paint_steps, 32).to(device)
    if decisioner_architechture == 'fc':
        decisioner = DecisionerFC(n_classes, paint_steps).to(device)
    elif decisioner_architechture == 'attn':
        decisioner = DecisionerStepAttention(n_classes, paint_steps, 32).to(device)
    criterion_train = nn.CrossEntropyLoss(reduction='none')  # for sample weight
    criterion_test = nn.CrossEntropyLoss()
    optimizer = optim.Adam(decisioner.parameters(), lr=1e-3, weight_decay=1e-4)
    phase2_history: list[dict] = []
    for epoch in range(0, best_epoch):
        # Train Full
        print(f'process epoch {epoch} on train_full')
        y_actual_train_full, y_pred_train_full, y_prob_train_full, indices_train_full, tfl_loss, tfl_acc = \
            process_epoch_decisioner(decisioner, epoch,
                                     train_full_loader, len(train_full_loader),
                                     n_classes, names_classes, device, optimizer, criterion_train, True,
                                     epsilons
                                     )
        phase2_history.append({'ds_type': 'train', 'epoch': epoch,
                                'avg_loss': tfl_loss, 'accuracy': tfl_acc})
        # Test
        print(f'process epoch {epoch} on test')
        with torch.no_grad():
            y_actual_test, y_pred_test, y_prob_test, indices_test, test_loss, test_acc = \
                process_epoch_decisioner(decisioner, epoch,
                                         test_loader, len(test_loader),
                                         n_classes, names_classes, device, optimizer, criterion_test, False,
                                         epsilons
                                         )
        phase2_history.append({'ds_type': 'validation', 'epoch': epoch,
                                'avg_loss': test_loss, 'accuracy': test_acc})

        torch.save(decisioner.state_dict(),
                   os.path.join(ckpt_dir, f'phase2_epoch_{epoch}.pth'))

        if (epoch + 1) % 5 == 0:
            eps_numpy = epsilons_test.cpu().numpy()
            y_actual_arr = np.array(y_actual_test)
            y_pred_arr = np.array(y_pred_test)
            decisioner_rows = []
            for eps in sorted(set(eps_numpy.tolist())):
                mask = eps_numpy == eps
                acc = float((y_actual_arr[mask] == y_pred_arr[mask]).mean())
                decisioner_rows.append({'epsilon': eps, 't': 'Decisioner', 'accuracy': acc})
            decisioner_df = pd.DataFrame(decisioner_rows)
            pcl_acc = df_test.groupby(['epsilon', 't']).apply(
                lambda g: pd.Series({'accuracy': (g['actual'] == g['pred']).mean()})
            ).reset_index()
            combined = pd.concat([pcl_acc, decisioner_df], ignore_index=True)
            plot_pcl_accuracy_vs_epsilon(
                {'adaptive': combined},
                title=f'PCL + Decisioner Accuracy vs Epsilon — Epoch {epoch + 1}',
                save_path=os.path.join(output_dir, f'decisioner_acc_vs_eps_epoch_{epoch + 1}.png'),
            )

    os.makedirs(output_dir, exist_ok=True)
    if phase2_history:
        plot_loss_and_acc(pd.DataFrame(phase2_history), output_dir, prefix='phase2_')
    print(f'training {decisioner_architechture} model finished!')

    return df_train_full, df_test, decisioner, \
        y_actual_train_full, y_pred_train_full, y_prob_train_full, indices_train_full, \
        y_actual_test, y_pred_test, y_prob_test, indices_test


def get_predictions_df_decisioner(actuals: list, preds: list, probs: list,
                                  indices: list, names_classes: list[str]) -> pd.DataFrame:
    """Builds a predictions DataFrame from the raw decisioner epoch output.

    Combines sample indices, true labels, predicted labels, and per-class
    probabilities into a structured DataFrame suitable for merging with
    the metadata DataFrame.

    Args:
        actuals: List of ground-truth class indices (one per sample).
        preds: List of predicted class indices (one per sample).
        probs: List of softmax probability arrays (one per sample).
        indices: List of sample index integers matching the dataset order.
        names_classes: Sorted list of class name strings; each class gets its
            own 'prob_<class>' column.

    Returns:
        DataFrame with columns: sample_idx, actual, pred, pred_class,
        and one prob_<class> column per class.
    """
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


def arange_results_decisioner(df_preds_train_full: pd.DataFrame,
                              df_preds_test: pd.DataFrame,
                              df_tfl: pd.DataFrame, df_tst: pd.DataFrame,
                              experiment: str) -> pd.DataFrame:
    """Merges decisioner predictions with attack metadata into the final results DataFrame.

    Joins the predictions DataFrame (produced by get_predictions_df_decisioner)
    with the full metadata DataFrame (from the attack_pcl output) on sample_idx.
    Tags each row with a phase label and the current experiment name, then
    reorders columns to match the standard results schema.

    Args:
        df_preds_train_full: Predictions DataFrame for the train_full split.
        df_preds_test: Predictions DataFrame for the test split.
        df_tfl: Full metadata DataFrame for train_full (from trainer_decisioner).
        df_tst: Full metadata DataFrame for test (from trainer_decisioner).
        experiment: Current experiment name stored in the 'experiment' column.

    Returns:
        Combined DataFrame with columns ordered to match the attack_pcl output
        schema, suitable for direct CSV export.
    """
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
