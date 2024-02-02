from model.model_factory import get_net, get_net_trainers
from util.models import *
from util.consts import NUM_OF_HYPHENS, LOCAL_RESULTS_DIR
import torch
torch.manual_seed(42)


def main_train_surrogate_for_victim_clf_undefended(experiment: str, ds_local_path: str, device: str,
                                                   batch_size: int, lr: float, n_epochs: int,
                                                   net_victim_local_dir: str):
    print('-' * NUM_OF_HYPHENS)
    print('Running train surrogate for victim clf undefended experiment...')

    # define image transformations
    train_transform = transform_dataset(augmentations=True)
    test_transform = transform_dataset(augmentations=False)

    print('load surrogate datasets to pytorch')
    ds_train_path = os.path.join(ds_local_path, 'train')
    ds_val_path = os.path.join(ds_local_path, 'val')
    ds_test_path = os.path.join(ds_local_path, 'test')
    ds_train, loader_train = create_ds_loader(path=ds_train_path, transform=train_transform, batch_size=batch_size)
    ds_val, loader_val = create_ds_loader(path=ds_val_path, transform=train_transform, batch_size=batch_size)
    ds_test, loader_test = create_ds_loader(path=ds_test_path, transform=test_transform, batch_size=batch_size)

    print(f'train: batches {len(loader_train)} size {len(ds_train)}  ' +
          f'\nval: batches {len(loader_val)} size {len(ds_val)}  ' +
          f'\ntest batches {len(loader_test)} size {len(ds_test)}')
    classes = loader_train.dataset.classes
    n_classes = len(classes)
    print(f'finished loading dataset, classes: {loader_train.dataset.classes}')

    print('initialize surrogate and victim models')
    net_surrogate = get_net(device, n_classes)
    net_victim = get_net(device, n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = get_net_trainers(net_surrogate, lr)
    print(f'finished initializing a pretrained resnet18 models')

    print('load pretrained victim model')
    mode_path, last_epoch = get_pretrained_model_path(net_victim_local_dir)
    if last_epoch > 0:
        print(f'found pretrained victim clf model in path: {mode_path}')
        # Remove "module." prefix
        state_dict = torch.load(mode_path, map_location=torch.device(device))
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        net_victim.load_state_dict(new_state_dict)
        print(f'pretrained victim clf model saved and load from local path: {mode_path}')
    else:
        raise Exception(f'did not find any pretrained victim_clf model (in {mode_path})')

    # Results df
    results_local_dir = os.path.join(LOCAL_RESULTS_DIR, experiment)
    results_local_path = os.path.join(results_local_dir, 'results.csv')
    results_deep_local_path = os.path.join(results_local_dir, 'results_deep.csv')
    os.makedirs(results_local_dir, exist_ok=True)
    results_df = pd.DataFrame()
    results_deep_df = pd.DataFrame()

    # Parallelize networks
    parallelize_networks(net_victim, net_surrogate)

    # train again on train_full
    print(f'\nrun full_dataset phase with {n_epochs} epochs\n')
    for epoch in range(1, n_epochs+1):
        print(f'full_dataset: start epoch {epoch}')
        deep_evaluate = epoch == n_epochs

        # Train
        results_df, results_deep_df = process_epoch_surrogate(experiment=experiment,
                                                              device=device,
                                                              epoch=epoch,
                                                              net_surrogate=net_surrogate,
                                                              net_victim=net_victim,
                                                              loader=generator_loader_train_full(loader_train,
                                                                                                 loader_val,
                                                                                                 loader_test),
                                                              loader_name='full_dataset_for_full_dataset',
                                                              n_batches=len(loader_train) + len(loader_val) +
                                                                        len(loader_test),
                                                              criterion=criterion,
                                                              optimizer=optimizer,
                                                              results_df=results_df,
                                                              n_classes=n_classes,
                                                              classes=classes,
                                                              is_train=True,
                                                              phase='full_dataset',
                                                              save_model=True,
                                                              deep_evaluate=deep_evaluate,
                                                              results_deep_df=results_deep_df,
                                                              scheduler=scheduler
                                                              )

        print(f'finished epoch {epoch}')
        print('save results')
        results_df.to_csv(results_local_path, index=False)

    print('save deep results')
    results_deep_df.to_csv(results_deep_local_path, index=False)

    print('Finished running train surrogate for victim clf undefended experiment!')


