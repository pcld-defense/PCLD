import os
import pandas as pd
import torch

from util.consts import NUM_OF_HYPHENS, IMAGENET_7_LABELS, RESOURCES_RESULTS_DIR, \
    RESOURCES_MODELS_DIR
from util.datasets import concat_to_one_decisioner_dataset
from util.models import trainer_decisioner, get_predictions_df_decisioner, arange_results_decisioner


def main_train_decisioner(args, device):
    # Fail fast
    dataset, experiment_name, batch_size, max_epochs, find_best_epoch, decisioner_architechture = \
        args.dataset, args.experiment_name, args.batch_size, args.max_epochs, args.find_best_epoch, \
            args.decisioner_architechture

    # =================== Load the dataset =================== #
    print('-' * NUM_OF_HYPHENS)
    print(f'Load the dataset...')
    ds_local_dir = os.path.join(RESOURCES_RESULTS_DIR, dataset)
    df_dataset = concat_to_one_decisioner_dataset(ds_local_dir)

    # sample weights per epsilon
    n_classes = len(IMAGENET_7_LABELS.keys())
    classes = sorted(IMAGENET_7_LABELS.values())
    epsilons = list(sorted(df_dataset.epsilon.unique()))
    epsilons_weights = {eps: 15-eps if eps < 15 else (2 if eps < 100 else 1) for eps in epsilons}
    prob_cols = ['prob_'+c for c in classes]
    num_paint_steps = len(df_dataset['t'].unique())

    # train-test split
    df_train = df_dataset[(df_dataset['phase'].isin(['train']))]
    df_val = df_dataset[(df_dataset['phase'].isin(['val']))]
    df_train_full = df_dataset[(df_dataset['phase'].isin(['train', 'val']))]
    df_test = df_dataset[(df_dataset['phase'] == 'test')]

    # =================== Train decisioner =================== #
    print('-' * NUM_OF_HYPHENS)
    print(f'Train and evaluate decisioner...')
    df_tfl, df_tst, decisioner, \
        y_actual_train_full, y_pred_train_full, y_prob_train_full, indices_train_full, \
        y_actual_test, y_pred_test, y_prob_test, indices_test = \
        trainer_decisioner(decisioner_architechture, batch_size, max_epochs, find_best_epoch,
                           df_train, df_val, df_train_full, df_test, num_paint_steps, prob_cols,
                           epsilons_weights, n_classes, classes,
                           epsilons, device)

    # =================== Save decisioner =================== #
    print('-' * NUM_OF_HYPHENS)
    print(f'Save trained decisioner...')
    model_local_dir = os.path.join(RESOURCES_MODELS_DIR, experiment_name)
    os.makedirs(model_local_dir, exist_ok=True)
    model_local_path = f'{model_local_dir}/model.pth'
    torch.save(decisioner.state_dict(), model_local_path)

    # =================== Arrange results =================== #
    print('-' * NUM_OF_HYPHENS)
    print(f'Arrange results...')
    df_preds_train_full = get_predictions_df_decisioner(y_actual_train_full, y_pred_train_full, y_prob_train_full,
                                                        indices_train_full, classes)
    df_preds_test = get_predictions_df_decisioner(y_actual_test, y_pred_test, y_prob_test,
                                                  indices_test, classes)
    df_results = arange_results_decisioner(df_preds_train_full, df_preds_test, df_tfl, df_tst, experiment_name)

    # =================== Save results =================== #
    print('-' * NUM_OF_HYPHENS)
    print(f'Save results...')
    results_local_dir = os.path.join(RESOURCES_RESULTS_DIR, experiment_name)
    os.makedirs(results_local_dir, exist_ok=True)
    results_local_path = os.path.join(results_local_dir, f'results.csv')
    df_results.to_csv(results_local_path, index=False)


