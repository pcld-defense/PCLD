import os
import pandas as pd

from model.pretrained_net import get_net_and_optim
from util.consts import RESOURCES_DATASETS_DIR, NUM_OF_HYPHENS, IMAGENET_7_LABELS, RESOURCES_RESULTS_DIR
from util.datasets import transform_dataset, generator_loader_train_full, get_loaders
from util.models import process_epoch_clf, get_best_epoch


def main_train_classifier(args, device):
    # Fail fast
    dataset, experiment_name, batch_size, max_epochs, find_best_epoch = \
        args.dataset, args.experiment_name, args.batch_size, args.max_epochs, args.find_best_epoch

    # =================== Load the dataset =================== #
    n_classes = len(IMAGENET_7_LABELS.keys())
    classes = sorted(IMAGENET_7_LABELS.values())
    net, criterion, optimizer, scheduler = get_net_and_optim(n_classes, device, 0.01)
    train_transform = transform_dataset(augmentations=True)
    test_transform = transform_dataset(augmentations=False)
    loaders = get_loaders(dataset, train_transform, test_transform, batch_size)

    # =================== prepare results reports =================== #
    results_local_path = os.path.join(RESOURCES_RESULTS_DIR, experiment_name, 'results.csv')
    results_deep_local_path = os.path.join(RESOURCES_RESULTS_DIR, experiment_name, 'results_deep.csv')
    os.makedirs(os.path.join(RESOURCES_RESULTS_DIR, experiment_name), exist_ok=True)
    results_df = pd.DataFrame()
    results_deep_df = pd.DataFrame()

    # =================== train the classifier =================== #
    best_epoch = max_epochs
    if find_best_epoch:
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
                                              loader_name='train_for_train_validate',
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
                                              loader_name='validation_for_train_validate',
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

    print(f'\nrun train_full-test phase with {best_epoch} epochs\n')
    net, criterion, optimizer, scheduler = get_net_and_optim(n_classes, device, 0.01)
    for epoch in range(0, best_epoch+1):
        print(f'train_full_test: start epoch {epoch}')
        deep_evaluate = epoch == best_epoch
        # Train Full
        results_df, results_deep_df = process_epoch_clf(experiment=experiment_name,
                                                        device=device,
                                                        epoch=epoch,
                                                        net=net,
                                                        loader=generator_loader_train_full(loaders['train'][1],
                                                                                           loaders['val_to_concat'][1]),
                                                        loader_name='train_full_for_train_full_test',
                                                        n_batches=len(loaders['train'][1])+len(loaders['val'][1]),
                                                        criterion=criterion,
                                                        optimizer=optimizer,
                                                        results_df=results_df,
                                                        n_classes=n_classes,
                                                        classes=classes,
                                                        is_train=True,
                                                        phase='train_full',
                                                        save_model=True,
                                                        deep_evaluate=deep_evaluate,
                                                        results_deep_df=results_deep_df,
                                                        scheduler=scheduler)
        # Test
        results_df, results_deep_df = process_epoch_clf(experiment=experiment_name,
                                                        device=device,
                                                        epoch=epoch,
                                                        net=net,
                                                        loader=loaders['test'][1],
                                                        loader_name='test_for_train_full_test',
                                                        n_batches=len(loaders['test'][1]),
                                                        criterion=criterion,
                                                        optimizer=optimizer,
                                                        results_df=results_df,
                                                        n_classes=n_classes,
                                                        classes=classes,
                                                        is_train=False,
                                                        phase='test',
                                                        save_model=False,
                                                        deep_evaluate=deep_evaluate,
                                                        results_deep_df=results_deep_df)

        print(f'train_full_test: finished epoch {epoch}')
        print('save the results locally')
        results_df.to_csv(results_local_path, index=False)
        print()

    print('save the deep results locally')
    results_deep_df.to_csv(results_deep_local_path, index=False)

    print(f'Finished training the classifier!')


