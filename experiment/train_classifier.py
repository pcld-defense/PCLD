import os
import pandas as pd

from model.pretrained_net import get_net_and_optim
from util.consts import RESOURCES_RESULTS_DIR
from util.datasets import transform_dataset, get_loaders
from util.models import process_epoch_clf, get_best_epoch


def main_train_classifier(args, device):
    # Fail fast
    dataset, splits, experiment_name, batch_size, max_epochs = \
    args.dataset, args.splits, args.experiment_name, args.batch_size, args.max_epochs

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
    results_deep_local_path = os.path.join(RESOURCES_RESULTS_DIR, experiment_name, 'results_deep.csv')
    os.makedirs(os.path.join(RESOURCES_RESULTS_DIR, experiment_name), exist_ok=True)
    results_df = pd.DataFrame()
    results_deep_df = pd.DataFrame()

    # =================== train the classifier =================== #
    net, criterion, optimizer, scheduler = get_net_and_optim(n_classes, device, 0.01)
    print(f'\nrun train_validate phase to find the best epoch\n')
    for epoch in range(0, max_epochs):
        print(f'train_validate: start epoch {epoch}')
        # Train
        results_df, _ = process_epoch_clf(experiment=experiment_name,
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
                                          deep_evaluate=False,
                                          results_deep_df=None,
                                          scheduler=scheduler)
        # Validate
        results_df, _ = process_epoch_clf(experiment=experiment_name,
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
                                          save_model=False,
                                          deep_evaluate=False,
                                          results_deep_df=None)
        print(f'train_validate: finished epoch {epoch}')
        print('save the results locally')
        results_df.to_csv(results_local_path, index=False)
        print()

        # over-fitting detection
        best_epoch, best_val_loss, best_val_acc = get_best_epoch(results_df, epoch)
        if epoch - best_epoch >= 20:
            print(f'Training had been stopped by OD. Best epoch {best_epoch} ' +
                  f'Best validation loss: {best_val_loss} ' +
                  f'Best validation accuracy {best_val_acc}')
            break

    print('save the deep results locally')
    results_deep_df.to_csv(results_deep_local_path, index=False)

    print(f'Finished training the classifier!')
