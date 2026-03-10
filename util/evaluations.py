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


def _print_accuracy_pivot(grouped: pd.DataFrame, label: str) -> None:
    """Prints a pivot table of accuracy by paint step and epsilon.

    Args:
        grouped: DataFrame with columns ['epsilon', 't', 'accuracy', 'total'].
        label: Label string printed in the header (e.g. 'naive' or 'adaptive').
    """
    pivot = grouped.pivot(index='t', columns='epsilon', values='accuracy')
    pivot.index = pivot.index.map(lambda t: 'original' if t == 999999 else t)
    pivot.columns.name = 'epsilon'
    pivot.index.name = 'paint_step'
    n = grouped['total'].iloc[0]
    print(f'\nPCL Accuracy [{label}] per paint step and epsilon (n={n} images per step)')
    print(pivot.applymap(lambda x: f'{x:.3f}').to_string())


def evaluate_pcl_accuracy_from_results(results_dir: str,
                                       epsilons: Optional[list[int]] = None,
                                       use_parquet: bool = False) -> dict[str, pd.DataFrame]:
    """Evaluates PCL classifier accuracy per epsilon and per paint step from attack result files.

    Loads all (or filtered) per-epsilon CSV or Parquet files from an attack_pcl results
    directory, computes accuracy for every (attacked_model, epsilon, paint_step) combination,
    and prints one pivot table per attacked_model value (e.g. 'naive', 'adaptive').

    The row with t=999999 represents the original unperturbed image (add_original step).

    Args:
        results_dir: Path to the attack_pcl results directory containing
            files named ``val_eps<N>_linf_results.csv`` or ``val_eps<N>_linf_results.parquet``.
        epsilons: Optional list of integer epsilon values to restrict evaluation to.
            When None, all files found in the directory are loaded.
        use_parquet: If True, load ``.parquet`` files instead of ``.csv`` files.

    Returns:
        Dict mapping each attacked_model value (e.g. 'naive', 'adaptive') to a DataFrame
        with columns ['epsilon', 't', 'accuracy', 'correct', 'total'], sorted by epsilon then t.
    """
    ext = 'parquet' if use_parquet else 'csv'
    pattern = os.path.join(results_dir, f'val_eps*_results.{ext}')
    result_files = sorted(glob.glob(pattern))
    if not result_files:
        raise FileNotFoundError(f'No result {ext.upper()} files found in {results_dir}')

    read_fn = pd.read_parquet if use_parquet else pd.read_csv

    dfs = []
    for f in result_files:
        df = read_fn(f)
        if epsilons is not None and df['epsilon'].iloc[0] not in epsilons:
            continue
        dfs.append(df)

    if not dfs:
        raise ValueError(f'No {ext.upper()} files matched epsilons={epsilons}')

    data = pd.concat(dfs, ignore_index=True)

    group_cols = ['attacked_model', 'epsilon', 't'] if 'attacked_model' in data.columns else ['epsilon', 't']

    grouped_all = data.groupby(group_cols).apply(
        lambda g: pd.Series({
            'correct': (g['actual'] == g['pred']).sum(),
            'total': len(g),
            'accuracy': (g['actual'] == g['pred']).mean(),
        })
    ).reset_index()
    grouped_all = grouped_all.sort_values(group_cols).reset_index(drop=True)

    if 'attacked_model' in grouped_all.columns:
        results = {}
        for model_type, subset in grouped_all.groupby('attacked_model'):
            subset = subset.drop(columns='attacked_model').reset_index(drop=True)
            _print_accuracy_pivot(subset, label=model_type)
            results[model_type] = subset
        return results
    else:
        _print_accuracy_pivot(grouped_all, label='all')
        return {'all': grouped_all}


def plot_pcl_accuracy_vs_epsilon(results: dict[str, pd.DataFrame],
                                 title: str = '',
                                 save_path: Optional[str] = None) -> None:
    """Plots accuracy vs epsilon with one line per paint step, split by attacked_model.

    Produces a figure with one subplot per attacked_model key present in ``results``
    (up to two: 'naive' and 'adaptive'), placed side by side. Each line represents a
    single paint step; t=999999 is labelled 'original'.

    Args:
        results: Dict mapping attacked_model name to a DataFrame with columns
            ['epsilon', 't', 'accuracy'], as returned by
            ``evaluate_pcl_accuracy_from_results``.
        title: Optional super-title for the figure.
        save_path: If provided, saves the figure to this path instead of showing it.
    """
    attack_types = [k for k in ['naive', 'adaptive'] if k in results]
    if not attack_types:
        attack_types = list(results.keys())

    n_plots = len(attack_types)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5), sharey=True)
    if n_plots == 1:
        axes = [axes]

    for ax, model_type in zip(axes, attack_types):
        df = results[model_type]
        steps = sorted(df['t'].unique())
        cmap = plt.cm.viridis(np.linspace(0, 1, len(steps)))

        for color, step in zip(cmap, steps):
            subset = df[df['t'] == step].sort_values('epsilon')
            label = 'original' if step == 999999 else str(step)
            ax.plot(subset['epsilon'], subset['accuracy'], marker='o', markersize=3,
                    label=label, color=color)

        ax.set_title(model_type)
        ax.set_xlabel('epsilon')
        ax.set_ylabel('accuracy')
        ax.grid(True, alpha=0.3)
        ax.legend(title='paint step', bbox_to_anchor=(1.01, 1), loc='upper left',
                  fontsize=7, ncol=1)

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f'Saved plot to {save_path}')
    else:
        plt.show()
    plt.close()


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
