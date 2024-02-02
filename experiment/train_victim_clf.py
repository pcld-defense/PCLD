from model.model_factory import get_net, get_net_trainers
from util.models import *
from util.consts import NUM_OF_HYPHENS, LOCAL_RESULTS_DIR
import torch
torch.manual_seed(42)


def main_train_victim_clf(experiment: str, ds_local_path: str, device: str,
                          batch_size: int, lr: float, n_epochs: int):
    print('-' * NUM_OF_HYPHENS)
    print('Running train victim classifier experiment...')

    # define image transformations
    train_transform = transform_dataset(augmentations=True)
    test_transform = transform_dataset(augmentations=False)

    print('load datasets to pytorch')
    ds_train_path = os.path.join(ds_local_path, 'train')
    ds_val_path = os.path.join(ds_local_path, 'val')
    ds_test_path = os.path.join(ds_local_path, 'test')
    ds_train, loader_train = create_ds_loader(path=ds_train_path, transform=train_transform, batch_size=batch_size)
    ds_val, loader_val = create_ds_loader(path=ds_val_path, transform=test_transform, batch_size=batch_size)
    ds_test, loader_test = create_ds_loader(path=ds_test_path, transform=test_transform, batch_size=batch_size)
    # we will use this validation set for concat to the training set
    ds_val_to_concat, loader_val_to_concat = create_ds_loader(path=ds_val_path, transform=train_transform, batch_size=batch_size)
    print(f'train: batches {len(loader_train)} size {len(ds_train)}  ' +
          f'\nval: batches {len(loader_val)} size {len(ds_val)}  ' +
          f'\ntest batches {len(loader_test)} size {len(ds_test)}')
    classes = loader_train.dataset.classes
    n_classes = len(classes)
    print(f'finished loading dataset, classes: {loader_train.dataset.classes}')

    print('initialize victim model')
    net = get_net(device, n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = get_net_trainers(net, lr)
    print(f'finished initializing a pretrained resnet18')

    # Results df
    results_local_dir = os.path.join(LOCAL_RESULTS_DIR, experiment)
    results_local_path = os.path.join(results_local_dir, 'results.csv')
    results_deep_local_path = os.path.join(results_local_dir, 'results_deep.csv')
    os.makedirs(results_local_dir, exist_ok=True)
    results_df = pd.DataFrame()
    results_deep_df = pd.DataFrame()

    # train on train_full
    print(f'\nrun with {n_epochs} epochs\n')
    for epoch in range(1, n_epochs+1):
        print(f'start epoch {epoch}')

        deep_evaluate = epoch == n_epochs
        # train-full
        results_df, results_deep_df = process_epoch_victim(experiment=experiment,
                                                           device=device,
                                                           epoch=epoch,
                                                           net=net,
                                                           loader=generator_loader_train_full(loader_train,
                                                                                              loader_val_to_concat),
                                                           loader_name='train_full_for_train_full_test',
                                                           n_batches=len(loader_train)+len(loader_val),
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
        # test
        results_df, results_deep_df = process_epoch_victim(experiment=experiment,
                                                           device=device,
                                                           epoch=epoch,
                                                           net=net,
                                                           loader=loader_test,
                                                           loader_name='test_for_train_full_test',
                                                           n_batches=len(loader_test),
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

        print(f'finished epoch {epoch}')
        print('save results')
        results_df.to_csv(results_local_path, index=False)

    print('save deep results')
    results_deep_df.to_csv(results_deep_local_path, index=False)

    print('Finished running train victim classifier experiment!')
