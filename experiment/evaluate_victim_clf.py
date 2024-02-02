from model.model_factory import get_net
from util.consts import LOCAL_RESULTS_DIR, NUM_OF_HYPHENS
from util.models import *
from util.evaluations import *
import torch.nn as nn
import torch
torch.manual_seed(42)


def main_evaluate_victim_clf(experiment: str, ds_local_path: str, device: str,
                             net_victim_local_dir: str, batch_size: int):
    print('-' * NUM_OF_HYPHENS)
    print('Running evaluate victim clf experiment...')

    # define image transformations
    transform = transform_dataset(augmentations=False)

    print('load datasets to pytorch')
    datasets_loaders = {}
    for dir_path, dir_names, file_names in os.walk(ds_local_path):
        dir_p = str(dir_path)
        if 'train' in dir_names:
            dir_ = dir_p.split('/')[-1]
            ds_train_path = os.path.join(dir_p, 'train')
            ds_test_path = os.path.join(dir_p, 'test')
            if 'validation' in dir_names:
                ds_val_path = os.path.join(dir_p, 'validation')
            else:
                ds_val_path = os.path.join(dir_p, 'val')

            ds_train, loader_train = create_ds_loader(path=ds_train_path, transform=transform, batch_size=batch_size)
            ds_val, loader_val = create_ds_loader(path=ds_val_path, transform=transform, batch_size=batch_size)
            ds_test, loader_test = create_ds_loader(path=ds_test_path, transform=transform, batch_size=batch_size)
            datasets_loaders[dir_] = {'train': [ds_train, loader_train],
                                      'validation': [ds_val, loader_val],
                                      'test': [ds_test, loader_test]}

    print(f'train: batches {len(loader_train)} size {len(ds_train)}  ' +
          f'\nval: batches {len(loader_val)} size {len(ds_val)}  ' +
          f'\ntest batches {len(loader_test)} size {len(ds_test)}')

    classes = loader_train.dataset.classes
    n_classes = len(classes)
    print(loader_train.dataset.classes)

    print('initialize victim model')
    net = get_net(device, n_classes)
    criterion = nn.CrossEntropyLoss()
    print(f'finished initializing a pretrained resnet18')

    print('load pretrained victim model')
    mode_path, last_epoch = get_pretrained_model_path(net_victim_local_dir)
    if last_epoch > 0:
        print(f'found pretrained victim clf model in path: {mode_path}')
        # Remove "module." prefix
        state_dict = torch.load(mode_path, map_location=torch.device(device))
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        net.load_state_dict(new_state_dict)
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

    print(f'run evaluation')
    for dir_ in datasets_loaders.keys():
        print(f'run evaluation on {dir_}')
        for ds_type in datasets_loaders[dir_].keys():
            print(f'ds_type: {ds_type}')
            dataset, loader = datasets_loaders[dir_][ds_type]
            correct = 0
            for i, data in enumerate(loader, 0):
                x, y, paths = data[0].to(device), data[1].to(device), data[2]
                outputs = net(x)
                _, y_pred = outputs.max(1)
                correct += y_pred.eq(y).sum().item()
                results_deep_df = deep_evaluation_general(experiment=experiment,
                                                          res_deep_df=results_deep_df,
                                                          dir=dir_,
                                                          ds_name=ds_type,
                                                          classes=classes,
                                                          images_paths=paths,
                                                          outputs=outputs,
                                                          labels=y,
                                                          criterion=criterion)
        print(f'finished evaluate {dir_}')

    # Save the deep results
    print('save deep results')
    results_deep_df.to_csv(results_deep_local_path, index=False)

    print('Finished running evaluate victim clf experiment!')
