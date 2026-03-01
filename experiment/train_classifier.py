import argparse
import os
import pandas as pd

from model.classifier import get_net_and_optim
from util.consts import RESOURCES_RESULTS_DIR
from util.datasets import transform_dataset, get_loaders
from util.evaluations import plot_loss_and_acc
from util.models import process_epoch_clf, get_best_epoch, save_best_cls_model


def main_train_classifier(args: argparse.Namespace, device: str) -> None:
    """Entry point for the train_classifier experiment.

    Trains a ResNet classifier on a painted (or raw) image dataset using
    SGD with a StepLR scheduler. Training runs for at most `max_epochs`
    epochs; early stopping is triggered when validation loss has not improved
    for `patience` consecutive epochs. The best checkpoint (lowest validation
    loss) is saved to resources/models/<experiment_name>/best_model.pth.

    Results (per-epoch loss and accuracy for train and validation splits) are
    saved as a CSV and as PNG plots to resources/results/<experiment_name>/.

    Args:
        args: Parsed argument namespace. Reads: dataset, dataset_type, splits,
            experiment_name, model_type, pretrained_weights, lr, patience,
            batch_size, max_epochs.
        device: Target device string resolved by main.py.
    """
    dataset, dataset_type, splits, experiment_name, model_type, pretrained_weights, lr, patience, batch_size, max_epochs = \
        (args.dataset, args.dataset_type, args.splits, args.experiment_name, args.model_type, args.pretrained_weights,
         args.lr, args.patience, args.batch_size, args.max_epochs)

    train_transform = transform_dataset(augmentations=True, dataset_type=dataset_type)
    val_transform = transform_dataset(augmentations=False, dataset_type=dataset_type,
                                      preprocessing=args.preprocessing)
    transform_dict = {"train": train_transform, "val": val_transform}

    loaders = get_loaders(dataset, splits, transform_dict, batch_size)

    train_ds, train_loader = loaders["train"]
    classes = sorted(train_ds.class_to_idx.keys())
    n_classes = len(classes)

    results_local_path = os.path.join(RESOURCES_RESULTS_DIR, experiment_name, 'results.csv')
    plots_local_path = os.path.join(RESOURCES_RESULTS_DIR, experiment_name)
    os.makedirs(os.path.join(RESOURCES_RESULTS_DIR, experiment_name), exist_ok=True)
    results_df = pd.DataFrame()

    net, criterion, optimizer, scheduler = get_net_and_optim(dataset_type, device, lr, model_type, pretrained_weights)
    best_val_loss = float('inf')

    print(f'\nrun train_validate phase to find the best epoch\n')
    for epoch in range(0, max_epochs):
        print(f'train_validate: start epoch {epoch}')
        results_df = process_epoch_clf(experiment=experiment_name, device=device, epoch=epoch, net=net,
                                       loader=loaders['train'][1], loader_name='train',
                                       criterion=criterion, optimizer=optimizer,
                                       results_df=results_df, n_classes=n_classes,
                                       classes=classes, is_train=True, phase='train', save_model=True,
                                       scheduler=scheduler)
        results_df = process_epoch_clf(experiment=experiment_name, device=device, epoch=epoch, net=net,
                                       loader=loaders['val'][1], loader_name='val',
                                       criterion=criterion, optimizer=optimizer,
                                       results_df=results_df, n_classes=n_classes,
                                       classes=classes, is_train=False, phase='validation', save_model=False)

        current_val_loss = results_df[results_df['ds_type'] == 'validation']['avg_loss'].iloc[-1]
        best_val_loss = save_best_cls_model(net, experiment_name, epoch, current_val_loss, best_val_loss)
        best_epoch, best_v_loss, best_v_acc = get_best_epoch(results_df, epoch)

        if epoch - best_epoch >= patience:
            print(f'Early Stopping triggered: No improvement for {patience} epochs.')
            print(f'Best epoch was {best_epoch} with loss {best_v_loss:.4f}')
            break

        print(f'train_validate: finished epoch {epoch}')
        plot_loss_and_acc(results_df, plots_local_path)
        print('save the results locally')
        results_df.to_csv(results_local_path, index=False)

    print(f'Finished training the classifier!')
    plot_loss_and_acc(results_df, plots_local_path)
    print('save the results locally')
    results_df.to_csv(results_local_path, index=False)
