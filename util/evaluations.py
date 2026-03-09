import glob
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def evaluate_paint_similarity(paints: torch.Tensor, originals: torch.Tensor,
                              step_labels: Optional[list] = None,
                              print_results: bool = True) -> pd.DataFrame:
    """Computes pixel-level distance and similarity metrics between painted canvases and originals.

    For each paint step, computes MSE, RMSE, L-inf distance, PSNR, and SSIM
    (averaged over the batch and channels), then prints a summary table.

    Args:
        paints: Painted canvas batch of shape (B, Steps, 3, H, W) in [0, 1].
        originals: Original image batch of shape (B, 3, H, W) in [0, 1].
        step_labels: Optional list of length Steps with human-readable step names
            (e.g. the output_every list). Defaults to 0-indexed integers.
        print_results: If True, prints the metrics table to stdout.

    Returns:
        DataFrame with columns ['step', 'mse', 'rmse', 'linf', 'psnr', 'ssim'],
        one row per paint step, sorted by step order.
    """
    B, Steps, C, H, W = paints.shape
    orig_np = originals.cpu().numpy()  # (B, C, H, W) in [0, 1]

    if step_labels is None:
        step_labels = list(range(Steps))

    rows = []
    for s in range(Steps):
        canvas_np = paints[:, s].cpu().numpy()  # (B, C, H, W)
        diff = canvas_np - orig_np

        mse = float(np.mean(diff ** 2))
        rmse = float(np.sqrt(mse))
        linf = float(np.abs(diff).max())
        psnr = float(np.mean([
            peak_signal_noise_ratio(orig_np[i].transpose(1, 2, 0),
                                    canvas_np[i].transpose(1, 2, 0),
                                    data_range=1.0)
            for i in range(B)
        ]))
        ssim = float(np.mean([
            structural_similarity(orig_np[i].transpose(1, 2, 0),
                                  canvas_np[i].transpose(1, 2, 0),
                                  data_range=1.0, channel_axis=2)
            for i in range(B)
        ]))

        rows.append({'step': step_labels[s], 'mse': mse, 'rmse': rmse,
                     'linf': linf, 'psnr': psnr, 'ssim': ssim})

    df = pd.DataFrame(rows)

    if print_results:
        print(f'\nPaint similarity to originals (B={B}, {H}x{W})')
        print(df.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

    return df


def evaluate_pcl_accuracy_from_csv(results_dir: str,
                                   epsilons: Optional[list[int]] = None) -> pd.DataFrame:
    """Evaluates PCL classifier accuracy per epsilon and per paint step from attack CSV files.

    Loads all (or filtered) per-epsilon CSVs from an attack_pcl results directory,
    computes accuracy as correct / total for every (epsilon, paint_step) combination,
    and prints a pivot table with paint steps as rows and epsilons as columns.

    The row with t=999999 represents the original unperturbed image (add_original step).

    Args:
        results_dir: Path to the attack_pcl results directory containing
            files named ``val_eps<N>_linf_results.csv``.
        epsilons: Optional list of integer epsilon values to restrict evaluation to.
            When None, all CSVs found in the directory are loaded.

    Returns:
        DataFrame with columns ['epsilon', 't', 'accuracy', 'correct', 'total'],
        sorted by epsilon then t.
    """
    pattern = os.path.join(results_dir, 'val_eps*_results.csv')
    csv_files = sorted(glob.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f'No result CSVs found in {results_dir}')

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        if epsilons is not None and df['epsilon'].iloc[0] not in epsilons:
            continue
        dfs.append(df)

    if not dfs:
        raise ValueError(f'No CSVs matched epsilons={epsilons}')

    data = pd.concat(dfs, ignore_index=True)

    grouped = data.groupby(['epsilon', 't']).apply(
        lambda g: pd.Series({
            'correct': (g['actual'] == g['pred']).sum(),
            'total': len(g),
            'accuracy': (g['actual'] == g['pred']).mean(),
        })
    ).reset_index()

    grouped = grouped.sort_values(['epsilon', 't']).reset_index(drop=True)

    pivot = grouped.pivot(index='t', columns='epsilon', values='accuracy')
    pivot.index = pivot.index.map(lambda t: 'original' if t == 999999 else t)
    pivot.columns.name = 'epsilon'
    pivot.index.name = 'paint_step'

    print(f'\nPCL Accuracy per paint step and epsilon (n={grouped["total"].iloc[0]} images per step)')
    print(pivot.applymap(lambda x: f'{x:.3f}').to_string())

    return grouped


def evaluate_print_decisioner(class_correct: list, class_total: list,
                              loss: float, epoch: int, dataset_size: int,
                              n_classes: int, classes: list[str],
                              epsilon_stats: dict) -> None:
    """Prints decisioner training/evaluation metrics to stdout.

    Computes overall accuracy and per-epsilon accuracy from the running
    counters and prints a formatted summary for the current epoch.

    Args:
        class_correct: Per-class count of correct predictions, length n_classes.
        class_total: Per-class count of total samples, length n_classes.
        loss: Accumulated (summed) loss over the entire epoch.
        epoch: Current epoch index.
        dataset_size: Number of batches in the loader (used to compute avg loss).
        n_classes: Number of output classes.
        classes: Sorted list of class name strings.
        epsilon_stats: Dict mapping epsilon value → [correct_count, total_count].
    """
    avg_loss = loss / dataset_size
    correct_sum = np.sum(class_correct)
    sample_size = np.sum(class_total)
    accuracy = correct_sum / sample_size

    print(f'Epoch %d, loss: %.8f Accuracy (Overall): %2d%% (%2d/%2d)' % (
        epoch, avg_loss, 100. * accuracy, correct_sum, sample_size))
    print(f'Accuracy by epsilon:')
    for eps in epsilon_stats.keys():
        eps_correct = epsilon_stats[eps][0]
        eps_count = epsilon_stats[eps][1]
        acc = eps_correct / eps_count
        print(f'eps {eps}: {acc} ({eps_correct} / {eps_count})')


def plot_loss_and_acc(df: pd.DataFrame, output_path: str) -> None:
    """Saves training and validation loss/accuracy plots to disk.

    Generates two PNG files in `output_path`:
    - `average_loss.png`: train vs. validation average loss per epoch.
    - `accuracy_per_epoch.png`: train vs. validation accuracy per epoch.

    Args:
        df: Results DataFrame containing columns 'ds_type', 'epoch',
            'avg_loss', and 'accuracy'. Rows with ds_type='train' and
            ds_type='validation' are plotted separately.
        output_path: Directory path where the PNG files will be saved.
    """
    train_df = df[df['ds_type'] == 'train']
    val_df = df[df['ds_type'] == 'validation']

    plt.figure(figsize=(10, 6))
    plt.plot(train_df['epoch'], train_df['avg_loss'], label='Train Loss', marker='o')
    plt.plot(val_df['epoch'], val_df['avg_loss'], label='Validation Loss', marker='o')
    plt.title('Average Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_path}/average_loss.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(train_df['epoch'], train_df['accuracy'], label='Train Accuracy', marker='o')
    plt.plot(val_df['epoch'], val_df['accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_path}/accuracy_per_epoch.png')
    plt.close()

    print("Plots generated: average_loss.png, accuracy_per_epoch.png")


def evaluate_print(experiment: str, res_df: pd.DataFrame,
                   class_correct: list, class_total: list, loss: float,
                   epoch: int, ds_type: str, dataset_size: int,
                   loader_name: str, n_classes: int,
                   classes: list[str]) -> pd.DataFrame:
    """Prints classifier metrics and appends the epoch results to a DataFrame.

    Computes overall and per-class accuracy, prints a formatted summary, and
    concatenates the new metrics row to `res_df`.

    Args:
        experiment: Experiment name stored in the results row.
        res_df: Existing results DataFrame to append to.
        class_correct: Per-class correct-prediction counts, length n_classes.
        class_total: Per-class total sample counts, length n_classes.
        loss: Accumulated loss over the epoch.
        epoch: Current epoch index.
        ds_type: Split label (e.g. 'train' or 'validation').
        dataset_size: Number of batches (denominator for avg loss).
        loader_name: Human-readable loader name printed in the summary.
        n_classes: Number of output classes.
        classes: Sorted list of class name strings.

    Returns:
        Updated results DataFrame with the current epoch metrics appended.
    """
    avg_loss = round(loss / dataset_size, 3)
    correct_sum = np.sum(class_correct)
    sample_size = np.sum(class_total)
    accuracy = correct_sum / sample_size

    res_dict = {
        'experiment': [experiment],
        'epoch': [epoch],
        'ds_name': [loader_name],
        'ds_type': [ds_type],
        'avg_loss': [avg_loss],
        'accuracy': [accuracy]
    }
    print(
        f'Epoch %d, loss: %.8f \t{loader_name} Accuracy (Overall): %2d%% (%2d/%2d)' % (
            epoch, avg_loss, 100. * accuracy, correct_sum, sample_size))
    for i in range(n_classes):
        class_correct_i = np.sum(class_correct[i])
        class_size_i = np.sum(class_total[i])
        class_acc_i = class_correct[i] / class_total[i]
        res_dict['accuracy_' + classes[i]] = [class_acc_i]
        print(f'{ds_type} Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_acc_i, class_correct_i, class_size_i))
    res_df = pd.concat([res_df, pd.DataFrame(res_dict)], axis=0, ignore_index=True)
    return res_df
