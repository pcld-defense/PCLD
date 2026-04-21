import argparse
import math
import os
import torch

from util.consts import NUM_OF_HYPHENS, RESOURCES_RESULTS_DIR, RESOURCES_MODELS_DIR
from util.datasets import concat_to_one_decisioner_dataset
from util.integrative import save_args_json
from util.models import trainer_decisioner, get_predictions_df_decisioner, arange_results_decisioner


def main_train_decisioner(args: argparse.Namespace, device: str) -> None:
    """Entry point for the train_decisioner experiment.

    Loads the attack-PCL result CSVs, trains the decisioner on per-step
    confidence trajectories, saves the trained model to
    resources/models/<experiment_name>/model.pth, and writes the evaluation
    results CSV to resources/results/<experiment_name>/results.csv.

    The decisioner is trained on 'train' split data, validated on 'val', and
    finally re-trained on 'train'+'val' (train_full) for the reported number
    of epochs before evaluating on 'test'. Sample weights are assigned per
    epsilon so that smaller (harder) perturbations receive higher weight.

    Args:
        args: Parsed argument namespace. Reads: dataset, experiment_name,
            batch_size, max_epochs, find_best_epoch, decisioner_architechture.
        device: Target device string resolved by main.py.
    """
    dataset, experiment_name, batch_size, max_epochs, find_best_epoch, decisioner_architechture = \
        args.dataset, args.experiment_name, args.batch_size, args.max_epochs, args.find_best_epoch, \
        args.decisioner_architechture
    max_train_epsilon = getattr(args, 'max_train_epsilon', -1)

    print('-' * NUM_OF_HYPHENS)
    print(f'Load the dataset...')
    ds_local_dir = os.path.join(RESOURCES_RESULTS_DIR, dataset)
    df_dataset = concat_to_one_decisioner_dataset(ds_local_dir)

    if max_train_epsilon >= 0:
        df_dataset = df_dataset[df_dataset['epsilon'] <= max_train_epsilon].reset_index(drop=True)
        print(f'Filtered dataset to epsilon <= {max_train_epsilon}: {len(df_dataset)} rows remaining')

    # Infer classes and class count directly from data
    classes = sorted(df_dataset['actual_class'].unique())
    n_classes = len(classes)
    epsilons = list(sorted(df_dataset.epsilon.unique()))
    max_eps = max(epsilons) if epsilons else 1
    epsilons_weights = {
        eps: max(1, round(10 * math.exp(-3 * eps / max(max_eps, 1)))) + 1
        for eps in epsilons
    }
    num_paint_steps = len(df_dataset['t'].unique())

    df_train = df_dataset[df_dataset['phase'] == 'train']
    df_val = df_dataset[df_dataset['phase'] == 'val']
    has_test = 'test' in df_dataset['phase'].values
    df_test = df_dataset[df_dataset['phase'] == 'test'] if has_test else df_val
    df_train_full = df_dataset[df_dataset['phase'].isin(['train', 'val'])] if has_test \
        else df_train

    results_local_dir = os.path.join(RESOURCES_RESULTS_DIR, experiment_name)
    os.makedirs(results_local_dir, exist_ok=True)

    print('-' * NUM_OF_HYPHENS)
    print(f'Train and evaluate decisioner...')
    df_tfl, df_tst, decisioner, \
        y_actual_train_full, y_pred_train_full, y_prob_train_full, indices_train_full, \
        y_actual_test, y_pred_test, y_prob_test, indices_test = \
        trainer_decisioner(decisioner_architechture, batch_size, max_epochs, find_best_epoch,
                           df_train, df_val, df_train_full, df_test, num_paint_steps,
                           epsilons_weights, n_classes, classes, epsilons, device,
                           results_local_dir)

    print('-' * NUM_OF_HYPHENS)
    print(f'Arrange results...')
    df_preds_train_full = get_predictions_df_decisioner(y_actual_train_full, y_pred_train_full,
                                                        y_prob_train_full, indices_train_full, classes)
    df_preds_test = get_predictions_df_decisioner(y_actual_test, y_pred_test,
                                                  y_prob_test, indices_test, classes)
    df_results = arange_results_decisioner(df_preds_train_full, df_preds_test, df_tfl, df_tst, experiment_name)

    print('-' * NUM_OF_HYPHENS)
    print(f'Save results and model...')
    save_args_json(args, results_local_dir)
    results_local_path = os.path.join(results_local_dir, 'results.parquet')
    df_results.to_parquet(results_local_path, compression='snappy', index=False)
    model_local_path = os.path.join(results_local_dir, 'model.pth')
    torch.save(decisioner.state_dict(), model_local_path)
