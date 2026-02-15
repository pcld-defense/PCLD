import os
import pandas as pd

from model.pretrained_net import get_net_and_optim
from util.consts import RESOURCES_RESULTS_DIR
from util.datasets import transform_dataset, get_loaders
from util.evaluations import plot_loss_and_acc
from util.models import process_epoch_clf, get_best_epoch, save_best_cls_model


def main_train_classifier(args, device):
    # Fail fast
    dataset, splits, experiment_name, model_type, pretrained_weights, lr, patience, batch_size, max_epochs = \
        (args.dataset, args.splits, args.experiment_name, args.model_type, args.pretrained_weights,
         args.lr, args.patience, args.batch_size, args.max_epochs)

    # =================== Load the dataset =================== #

    train_transform = transform_dataset(augmentations=True)
    val_transform = transform_dataset(augmentations=False)
    transform_dict = {"train": train_transform,
                      "val": val_transform}

    loaders = get_loaders(dataset, splits, transform_dict, batch_size)

    train_ds, train_loader = loaders["train"]
    classes = sorted(train_ds.class_to_idx.keys())
    n_classes = len(classes)

    # =================== prepare results reports =================== #
    results_local_path = os.path.join(RESOURCES_RESULTS_DIR, experiment_name, 'results.csv')
    plots_local_path = os.path.join(RESOURCES_RESULTS_DIR, experiment_name)
    os.makedirs(os.path.join(RESOURCES_RESULTS_DIR, experiment_name), exist_ok=True)
    results_df = pd.DataFrame()
    results_deep_df = pd.DataFrame()

    # =================== train the classifier =================== #
    net, criterion, optimizer, scheduler = get_net_and_optim(n_classes, device, lr, model_type, pretrained_weights)
    best_val_loss = float('inf')

    print(f'\nrun train_validate phase to find the best epoch\n')
    for epoch in range(0, max_epochs):
        print(f'train_validate: start epoch {epoch}')
        # Train
        results_df = process_epoch_clf(experiment=experiment_name,
                                       device=device,
                                       epoch=epoch,
                                       net=net,
                                       loader=loaders['train'][1],
                                       loader_name='train',
                                       n_batches=len(loaders['train'][1]),
                                       criterion=criterion,
                                       optimizer=optimizer,
                                       results_df=results_df,
                                       n_classes=n_classes,
                                       classes=classes,
                                       is_train=True,
                                       phase='train',
                                       save_model=True,
                                       scheduler=scheduler)
        # Validate
        results_df = process_epoch_clf(experiment=experiment_name,
                                       device=device,
                                       epoch=epoch,
                                       net=net,
                                       loader=loaders['val'][1],
                                       loader_name='val',
                                       n_batches=len(loaders['val'][1]),
                                       criterion=criterion,
                                       optimizer=optimizer,
                                       results_df=results_df,
                                       n_classes=n_classes,
                                       classes=classes,
                                       is_train=False,
                                       phase='validation',
                                       save_model=False, )

        # over-fitting detection
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
