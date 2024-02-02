from model.model_factory import get_net, get_net_trainers
from util.models import *
from util.consts import NUM_OF_HYPHENS, LOCAL_RESULTS_DIR, LOCAL_DATASETS_DIR
import torch
torch.manual_seed(42)


def main_attack_surrogate_for_victim_decisioner(experiment: str, ds_local_path: str, device: str,
                                                batch_size: int, epsilons: list, net_surrogate_local_dir: str,
                                                attack: str = 'fgsm'):
    print('-' * NUM_OF_HYPHENS)
    print('Running attack surrogate for victim decisioner experiment...')


    # define image transformations
    train_transform = transform_dataset(augmentations=False)
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
    criterion = nn.CrossEntropyLoss()
    print(f'finished initializing a pretrained resnet18 models')

    print('load pretrained surrogate model')
    mode_path, last_epoch = get_pretrained_model_path(net_surrogate_local_dir)
    if last_epoch > 0:
        print(f'found pretrained surrogate model in path: {mode_path}')
        # Remove "module." prefix
        state_dict = torch.load(mode_path, map_location=torch.device(device))
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        net_surrogate.load_state_dict(new_state_dict)
        print(f'pretrained surrogate model saved and load from local path: {mode_path}')
    else:
        raise Exception(f'did not find any pretrained surrogate model (in {mode_path})')

    # Results df
    results_local_dir = os.path.join(LOCAL_RESULTS_DIR, experiment)
    results_local_path = os.path.join(results_local_dir, 'results.csv')
    results_deep_local_path = os.path.join(results_local_dir, 'results_deep.csv')
    os.makedirs(results_local_dir, exist_ok=True)
    results_df = pd.DataFrame()
    results_deep_df = pd.DataFrame()

    # Parallelize networks
    parallelize_networks(net_surrogate)

    new_ds_dir = ds_local_path + '_' + experiment


    print('attack train')
    results_deep_df = attack_decisioner(experiment=experiment,
                                        device=device,
                                        attack=attack,
                                        net_surrogate=net_surrogate,
                                        loader=loader_train,
                                        loader_name='train',
                                        criterion=criterion,
                                        classes=classes,
                                        epsilons=epsilons,
                                        new_ds_dir=new_ds_dir,
                                        phase='train',
                                        results_deep_df=results_deep_df)

    print('attack validation')
    results_deep_df = attack_decisioner(experiment=experiment,
                                        device=device,
                                        attack=attack,
                                        net_surrogate=net_surrogate,
                                        loader=loader_val,
                                        loader_name='validation',
                                        criterion=criterion,
                                        classes=classes,
                                        epsilons=epsilons,
                                        new_ds_dir=new_ds_dir,
                                        phase='validation',
                                        results_deep_df=results_deep_df)

    print('attack test')
    results_deep_df = attack_decisioner(experiment=experiment,
                                        device=device,
                                        attack=attack,
                                        net_surrogate=net_surrogate,
                                        loader=loader_test,
                                        loader_name='test',
                                        criterion=criterion,
                                        classes=classes,
                                        epsilons=epsilons,
                                        new_ds_dir=new_ds_dir,
                                        phase='test',
                                        results_deep_df=results_deep_df)

    print('save the results')
    results_deep_df.to_csv(results_deep_local_path, index=False)

    print('Finished running attack surrogate for victim decisioner experiment!')
