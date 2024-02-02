from util.evaluations import *
from util.datasets import *
from util.consts import LOCAL_MODELS_DIR
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image

from art.attacks.evasion import CarliniLInfMethod, DeepFool, SaliencyMapMethod, AutoAttack
from art.estimators.classification import PyTorchClassifier

np.random.seed(42)


def load_model(model_path, model):
    model.load_state_dict(torch.load(model_path))


def get_pretrained_model_path(dir_path):
    for subdir, dirs, files in os.walk(dir_path):
        print(f'run on: {subdir}')
        if files:
            for f in files:
                if 'epoch' in f:
                    epoch = int(f.split('.pth')[0].split('epoch')[-1])
                    path = os.path.join(dir_path, f)
                    return path, epoch
    raise Exception(f'not a valid model dir: {dir_path}')


def parallelize_networks(net_1, net_2=None):
    print("There are ", torch.cuda.device_count(), " GPUs!")
    if torch.cuda.device_count() > 1:
        net_1 = torch.nn.DataParallel(net_1)
        if net_2:
            net_2 = torch.nn.DataParallel(net_2)
        print('parallelled!')
    else:
        print("no parallelize!")


def process_epoch_victim_adv(experiment, epsilons, device, epoch, net, loader,
                             loader_name, n_batches, criterion, optimizer, results_df, n_classes,
                             classes, is_train=True, phase='train', save_model=True, deep_evaluate=False,
                             results_deep_df=None, scheduler=None, adv_balanced_rate=1):
    total_loss = 0.0
    if is_train:
        net.train()
    else:
        net.eval()
    class_correct = list(0. for i in range(n_classes))
    class_total = list(0. for i in range(n_classes))

    epsilons_non_zero = [e for e in epsilons if e != 0]
    for i, data in enumerate(loader, 0):
        eps_0 = np.random.choice([True, False], p=[adv_balanced_rate, 1-adv_balanced_rate])
        chosen_epsilons = list(np.random.choice(epsilons_non_zero, size=1)) + ([0] if eps_0 else [])
        for epsilon in chosen_epsilons:
            inputs, labels = data[0].to(device), data[1].to(device)
            if 'cuda' in device:
                inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            images_paths = data[2]

            # generate targeted labels for adv attack
            y_classes_targeted_tmp = [(yi.item()+1) % len(classes) for yi in labels]
            y_classes_targeted = np.zeros((len(y_classes_targeted_tmp), max(y_classes_targeted_tmp) + 1))
            y_classes_targeted[np.arange(len(y_classes_targeted_tmp)), y_classes_targeted_tmp] = 1
            y_classes_targeted = torch.Tensor(y_classes_targeted).to(device)

            x_adv = attacker(attack='fgsm', net=net, x=inputs, epsilon=epsilon, targeted=True, y_targeted=y_classes_targeted)
            x_adv = torch.clamp(torch.round(x_adv), min=0, max=255)

            optimizer.zero_grad()
            outputs = net(x_adv)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            # compare predictions to true label
            correct_tensor = pred.eq(labels.data.view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if device == 'cpu' \
                else np.squeeze(correct_tensor.cpu().numpy())
            for j in range(len(labels.data)):
                label = labels.data[j]
                class_correct[label] += correct[j].item()
                class_total[label] += 1

            if deep_evaluate:
                results_deep_df = deep_evaluation_training(experiment,
                                                           results_deep_df,
                                                           epoch,
                                                           loader_name,
                                                           phase,
                                                           loader_name,
                                                           n_classes,
                                                           classes,
                                                           images_paths,
                                                           outputs,
                                                           labels,
                                                           criterion,
                                                           epsilon)

    if is_train and scheduler:
        scheduler.step()

    results_df = evaluate_print(experiment=experiment,
                                res_df=results_df,
                                class_correct=class_correct,
                                class_total=class_total,
                                loss=total_loss,
                                epoch=epoch,
                                ds_type=phase,
                                dataset_size=n_batches,
                                loader_name=loader_name,
                                n_classes=n_classes,
                                classes=classes
                                )

    if save_model:
        # remove all prev models in this dir
        save_dir = os.path.join(LOCAL_MODELS_DIR, experiment)
        os.makedirs(save_dir, exist_ok=True)
        [os.remove(os.path.join(save_dir, d)) for d in os.listdir(save_dir)]
        # save the updated model
        save_path = f'{save_dir}/epoch{epoch}.pth'
        torch.save(net.state_dict(), save_path)

    return results_df, results_deep_df


def process_epoch_victim(experiment, device, epoch, net, loader, loader_name, n_batches,
                         criterion, optimizer,
                         results_df, n_classes, classes, is_train=True,
                         phase='train', save_model=True,
                         deep_evaluate=False, results_deep_df=None, scheduler=None):
    total_loss = 0.0
    if is_train:
        net.train()
    else:
        net.eval()
    class_correct = list(0. for i in range(n_classes))
    class_total = list(0. for i in range(n_classes))

    for i, data in enumerate(loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        if 'cuda' in device:
            inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        images_paths = data[2]
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        if is_train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        # compare predictions to true label
        correct_tensor = pred.eq(labels.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if device == 'cpu' \
            else np.squeeze(correct_tensor.cpu().numpy())
        for j in range(len(labels.data)):
            label = labels.data[j]
            class_correct[label] += correct[j].item()
            class_total[label] += 1

        if deep_evaluate:
            results_deep_df = deep_evaluation_training(experiment,
                                                       results_deep_df,
                                                       epoch,
                                                       loader_name,
                                                       phase,
                                                       loader_name,
                                                       n_classes,
                                                       classes,
                                                       images_paths,
                                                       outputs,
                                                       labels,
                                                       criterion)

    if is_train and scheduler:
        scheduler.step()

    results_df = evaluate_print(experiment=experiment,
                                res_df=results_df,
                                class_correct=class_correct,
                                class_total=class_total,
                                loss=total_loss,
                                epoch=epoch,
                                ds_type=phase,
                                dataset_size=n_batches,
                                loader_name=loader_name,
                                n_classes=n_classes,
                                classes=classes
                                )

    if save_model:
        # remove all prev models in this dir
        save_dir = os.path.join(LOCAL_MODELS_DIR, experiment)
        os.makedirs(save_dir, exist_ok=True)
        [os.remove(os.path.join(save_dir, d)) for d in os.listdir(save_dir)]
        # save the updated model
        save_path = f'{save_dir}/epoch{epoch}.pth'
        torch.save(net.state_dict(), save_path)

    return results_df, results_deep_df


def process_epoch_surrogate(experiment, device, epoch, net_surrogate, net_victim,
                            loader, loader_name, n_batches, criterion, optimizer,
                            results_df, n_classes, classes, is_train=True,
                            phase='train', save_model=True,
                            deep_evaluate=False, results_deep_df=None, scheduler=None, loaders_victim=None):
    total_loss = 0.0
    if is_train:
        net_surrogate.train()
    else:
        net_surrogate.eval()

    class_correct = list(0. for i in range(n_classes))
    class_total = list(0. for i in range(n_classes))


    for i, data in enumerate(loader, 0):
        print(f'run batch {i+1}')
        inputs, labels = data[0].to(device), data[1].to(device)
        images_paths = data[2]
        images_names = [ip.split('/')[-1].split('.')[0] for ip in images_paths]

        if loaders_victim and loaders_victim[0]:
            # Fetch the corresponding batch from loader_victim
            loader_victim_data = []
            loader_victim_found_mask = [False]*len(images_names)
            for loader_v in loaders_victim:
                loader_v_data, found_mask = loader_v.dataset.get_items_by_match_names(images_names)
                loader_victim_data.extend(loader_v_data)
                loader_victim_found_mask = np.logical_or(np.array(found_mask), np.array(loader_victim_found_mask))
            inputs_victim, _ = zip(*[(d[0].to(device), d[1]) for d in loader_victim_data])
            inputs_victim = torch.stack(inputs_victim, dim=0)
            # Process the filtered batch
            inputs = inputs[loader_victim_found_mask]
            labels = labels[loader_victim_found_mask]
            images_paths = [path for j, path in enumerate(images_paths) if loader_victim_found_mask[j]]
            images_names = [ip.split('/')[-1].split('.')[0] for ip in images_paths]
        else:
            inputs_victim = inputs

        optimizer.zero_grad()
        outputs = net_surrogate(inputs)
        outputs_victim = net_victim(inputs_victim)
        _, victim_labels = torch.max(outputs_victim, 1)
        loss = criterion(outputs, victim_labels)
        if is_train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        # compare predictions to true label
        correct_tensor = pred.eq(victim_labels.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if device == 'cpu' \
            else np.squeeze(correct_tensor.cpu().numpy())
        for j in range(len(victim_labels.data)):
            label = victim_labels.data[j]
            class_correct[label] += correct[j].item()
            class_total[label] += 1

        if deep_evaluate:
            results_deep_df = deep_evaluation_training(experiment=experiment,
                                                       res_deep_df=results_deep_df,
                                                       epoch=epoch,
                                                       ds_name=loader_name,
                                                       ds_type='train' if is_train else 'test',
                                                       filter_level=loader_name,
                                                       n_classes=n_classes,
                                                       classes=classes,
                                                       images_paths=images_paths,
                                                       outputs=outputs,
                                                       labels=victim_labels,
                                                       criterion=criterion)

    if is_train and scheduler:
        scheduler.step()

    results_df = evaluate_print(experiment=experiment,
                                res_df=results_df,
                                class_correct=class_correct,
                                class_total=class_total,
                                loss=total_loss,
                                epoch=epoch+1,
                                ds_type='train' if is_train else 'test',
                                dataset_size=n_batches,
                                loader_name=loader_name,
                                n_classes=n_classes,
                                classes=classes
                                )

    if save_model:
        # remove all prev models in this dir
        save_dir = os.path.join(LOCAL_MODELS_DIR, experiment)
        os.makedirs(save_dir, exist_ok=True)
        [os.remove(os.path.join(save_dir, d)) for d in os.listdir(save_dir)]
        # save the updated model
        save_path = f'{save_dir}/epoch{epoch}.pth'
        torch.save(net_surrogate.state_dict(), save_path)

    return results_df, results_deep_df


def process_epoch_surrogate_for_decisioner(experiment, device, epoch, net_surrogate, decisioner_labels,
                                           loader, loader_name, n_batches, criterion, optimizer,
                                           results_df, n_classes, classes, is_train=True,
                                           phase='train', save_model=True,
                                           deep_evaluate=False, results_deep_df=None, scheduler=None):
    total_loss = 0.0
    if is_train:
        net_surrogate.train()
    else:
        net_surrogate.eval()

    class_correct = list(0. for i in range(n_classes))
    class_total = list(0. for i in range(n_classes))

    for i, data in enumerate(loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        images_paths = data[2]
        images_names = [ip.split('/')[-1].split('.')[0] for ip in images_paths]
        optimizer.zero_grad()
        outputs = net_surrogate(inputs)
        outputs_victim = decisioner_labels.iloc[pd.Index(decisioner_labels['image_name_short']).get_indexer(images_names)]
        victim_labels = torch.tensor(outputs_victim['decisioner_pred_label'].values, device=device)
        loss = criterion(outputs, victim_labels)
        if is_train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        # compare predictions to true label
        correct_tensor = pred.eq(victim_labels.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if device == 'cpu' \
            else np.squeeze(correct_tensor.cpu().numpy())
        for j in range(len(victim_labels.data)):
            label = victim_labels.data[j]
            class_correct[label] += correct[j].item()
            class_total[label] += 1

        if deep_evaluate:
            results_deep_df = deep_evaluation_training(experiment=experiment,
                                                       res_deep_df=results_deep_df,
                                                       epoch=epoch,
                                                       ds_name=loader_name,
                                                       ds_type='train' if is_train else 'test',
                                                       filter_level=loader_name,
                                                       n_classes=n_classes,
                                                       classes=classes,
                                                       images_paths=images_paths,
                                                       outputs=outputs,
                                                       labels=victim_labels,
                                                       criterion=criterion)

    if is_train and scheduler:
        scheduler.step()
    results_df = evaluate_print(experiment=experiment,
                                res_df=results_df,
                                class_correct=class_correct,
                                class_total=class_total,
                                loss=total_loss,
                                epoch=epoch+1,
                                ds_type='train' if is_train else 'test',
                                dataset_size=n_batches,
                                loader_name=loader_name,
                                n_classes=n_classes,
                                classes=classes
                                )

    if save_model:
        # remove all prev models in this dir
        save_dir = os.path.join(LOCAL_MODELS_DIR, experiment)
        os.makedirs(save_dir, exist_ok=True)
        [os.remove(os.path.join(save_dir, d)) for d in os.listdir(save_dir)]
        # save the updated model
        save_path = f'{save_dir}/epoch{epoch}.pth'
        torch.save(net_surrogate.state_dict(), save_path)

    return results_df, results_deep_df


def process_tensor(x):
    return x.permute(1, 2, 0).to('cpu', torch.uint8).numpy()



def attacker(attack, net, x, epsilon, n_classes=3, targeted=False, y_targeted=None, batch_size=128):
    if epsilon == 0:
        return x
    if attack == 'fgsm':
        return fast_gradient_method(model_fn=net, x=x, eps=epsilon, norm=np.inf, y=y_targeted, targeted=targeted)
    elif attack == 'pgd':
        return projected_gradient_descent(model_fn=net, x=x, eps=epsilon, eps_iter=1, nb_iter=epsilon, norm=np.inf,
                                          y=y_targeted, targeted=targeted, rand_init=False, sanity_checks=False)
    else: # ART attacks
        # Define the loss function, optimizer
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters())
        # Create an ART classifier with 0-255 input range
        classifier = PyTorchClassifier(
            model=net,
            clip_values=(0, 255),
            loss=loss_fn,
            optimizer=optimizer,
            input_shape=(3, 300, 300),
            nb_classes=n_classes
        )
        if attack == 'cw':
            attack = CarliniLInfMethod(classifier=classifier, targeted=targeted, initial_const=1, largest_const=50,
                                       batch_size=batch_size, confidence=0.6)
        elif attack == 'deepfool':
            attack = DeepFool(classifier=classifier, max_iter=10, epsilon=epsilon, batch_size=batch_size)
        elif attack == 'jsma':
            attack = SaliencyMapMethod(classifier=classifier, theta=epsilon, batch_size=batch_size)
        elif attack == 'autoattack':
            attack = AutoAttack(estimator=classifier, norm=np.inf, eps=epsilon, eps_step=1, targeted=targeted,
                                batch_size=batch_size)
        else:
            raise Exception(f'Unsupported attack exception: {attack}')
        # Convert PyTorch tensors to NumPy arrays
        x_np = x.cpu().detach().numpy()
        y_targeted_np = y_targeted.cpu().detach().numpy()
        x_adv = attack.generate(x=x_np, y=y_targeted_np)
        return x_adv


def attack_clf(experiment, device, attack, net_surrogate, net_victim,
               loader, loader_name, criterion, classes, epsilons, new_ds_dir, phase, results_deep_df):
    net_surrogate.eval()
    net_victim.eval()
    targeted = True
    new_adv_dir = os.path.join(new_ds_dir, attack, 'targeted')
    os.makedirs(new_adv_dir, exist_ok=True)
    n_classes = len(classes)
    print(f'=========> running on {loader_name} <==========')
    for epsilon in epsilons:
        print(f'\nepsilon {epsilon}')
        correct = 0
        correct_adv = 0
        correct_victim = 0
        correct_victim_adv = 0

        new_epsilon_adv_dir = os.path.join(new_adv_dir, f'eps_{epsilon}')
        os.makedirs(new_epsilon_adv_dir, exist_ok=True)

        new_ds_type_adv_dir = os.path.join(new_epsilon_adv_dir, phase)
        os.makedirs(new_ds_type_adv_dir, exist_ok=True)

        for c in classes:
            os.makedirs(os.path.join(new_ds_type_adv_dir, c), exist_ok=True)

        for i, data in enumerate(loader, 0):
            x, y, paths = data[0].to(device), data[1].to(device), data[2]
            y_classes = [yi.item() for yi in y]
            # generate targeted labels for adv attack
            y_classes_targeted_tmp = [(yi.item()+1) % len(classes) for yi in y]
            y_classes_targeted = np.zeros((len(y_classes_targeted_tmp), max(y_classes_targeted_tmp) + 1))
            y_classes_targeted[np.arange(len(y_classes_targeted_tmp)), y_classes_targeted_tmp] = 1
            y_classes_targeted = torch.Tensor(y_classes_targeted).to(device)
            y_classes_targeted_tmp = torch.Tensor(y_classes_targeted_tmp).to(device)

            img_names = [p.split('/')[-1].split('.')[0] for p in paths]
            x_adv = attacker(attack=attack,
                             net=net_surrogate,
                             x=x,
                             epsilon=epsilon,
                             n_classes=n_classes,
                             targeted=targeted,
                             y_targeted=y_classes_targeted)  # attack
            x_adv = torch.Tensor(x_adv).to(device)
            x_adv = torch.clamp(torch.round(x_adv), min=0, max=255)
            outputs_surrogate_orig = net_surrogate(x)
            outputs_surrogate_adv = net_surrogate(x_adv)
            outputs_victim_orig = net_victim(x)
            outputs_victim_adv = net_victim(x_adv)
            _, y_pred = outputs_surrogate_orig.max(1)  # surrogate prediction on clean examples
            _, y_pred_adv = outputs_surrogate_adv.max(1)  # surrogate prediction on adversarial examples
            _, y_pred_victim = outputs_victim_orig.max(1)  # victim prediction on clean examples
            _, y_pred_adv_victim = outputs_victim_adv.max(1)  # victim prediction on adversarial examples

            correct += y_pred.eq(y).sum().item()
            correct_adv += y_pred_adv.eq(y).sum().item()
            correct_victim += y_pred_victim.eq(y).sum().item()
            correct_victim_adv += y_pred_adv_victim.eq(y).sum().item()

            for j in range(len(x)):
                img_name = img_names[j] + '.png'
                img_class = classes[y_classes[j]]
                img_path_adv = os.path.join(new_ds_type_adv_dir, img_class, img_name)
                Image.fromarray(process_tensor(x_adv[j])).save(img_path_adv)
                # upload_file_to_s3(img_path_adv, img_path_adv, s3_resource=s3_resource)

            results_deep_df = deep_evaluation_attack(experiment=experiment,
                                                     attack=attack,
                                                     epsilon=epsilon,
                                                     ds_name=loader_name,
                                                     classes=classes,
                                                     images_paths=paths,
                                                     outputs_surrogate_orig=outputs_surrogate_orig,
                                                     outputs_surrogate_adv=outputs_surrogate_adv,
                                                     outputs_victim_orig=outputs_victim_orig,
                                                     outputs_victim_adv=outputs_victim_adv,
                                                     labels=y,
                                                     criterion=criterion,
                                                     res_deep_df=results_deep_df)

        n = len(loader.dataset)
        print("acc on clean examples (%): {:.3f} ({:.3f})".format(correct / n * 100.0, correct))
        print("acc on {} adversarial examples (%): {:.3f} ({:.3f})".format(attack, correct_adv / n * 100.0,
                                                                           correct_adv))
        print("victim acc on clean examples (%): {:.3f} ({:.3f})".format(correct_victim / n * 100.0,
                                                                         correct_victim))
        print("victim acc on {} adversarial examples (%): {:.3f} ({:.3f})".format(attack,
                                                                                  correct_victim_adv / n * 100.0,
                                                                                  correct_victim_adv))

    return results_deep_df


def attack_decisioner(experiment, device, attack, net_surrogate,
                      loader, loader_name, criterion, classes, epsilons, new_ds_dir,
                      phase, results_deep_df):
    net_surrogate.eval()
    targeted = True
    new_adv_dir = os.path.join(new_ds_dir, attack, 'targeted')
    os.makedirs(new_adv_dir, exist_ok=True)
    n_classes = len(classes)
    print(f'=========> running on {loader_name} <==========')
    for epsilon in epsilons:
        print(f'\nepsilon {epsilon}')
        correct = 0
        correct_adv = 0

        new_epsilon_adv_dir = os.path.join(new_adv_dir, f'eps_{epsilon}')
        os.makedirs(new_epsilon_adv_dir, exist_ok=True)

        new_ds_type_adv_dir = os.path.join(new_epsilon_adv_dir, phase)
        os.makedirs(new_ds_type_adv_dir, exist_ok=True)

        for c in classes:
            os.makedirs(os.path.join(new_ds_type_adv_dir, c), exist_ok=True)

        for i, data in enumerate(loader, 0):
            x, y, paths = data[0].to(device), data[1].to(device), data[2]
            y_classes = [yi.item() for yi in y]
            # generate targeted labels for adv attack
            y_classes_targeted_tmp = [(yi.item()+1) % len(classes) for yi in y]
            y_classes_targeted = np.zeros((len(y_classes_targeted_tmp), max(y_classes_targeted_tmp) + 1))
            y_classes_targeted[np.arange(len(y_classes_targeted_tmp)), y_classes_targeted_tmp] = 1
            y_classes_targeted = torch.Tensor(y_classes_targeted).to(device)
            y_classes_targeted_tmp = torch.Tensor(np.array(y_classes_targeted_tmp)).to(torch.long).to(device)

            img_names = [p.split('/')[-1].split('.')[0] for p in paths]
            x_adv = attacker(attack=attack,
                             net=net_surrogate,
                             x=x,
                             epsilon=epsilon,
                             n_classes=n_classes,
                             targeted=targeted,
                             y_targeted=y_classes_targeted)  # attack
            x_adv = torch.Tensor(x_adv).to(device)
            x_adv = torch.clamp(torch.round(x_adv), min=0, max=255)
            outputs_surrogate_orig = net_surrogate(x)
            outputs_surrogate_adv = net_surrogate(x_adv)
            _, y_pred = outputs_surrogate_orig.max(1)  # surrogate prediction on clean examples
            _, y_pred_adv = outputs_surrogate_adv.max(1)  # surrogate prediction on adversarial examples

            correct += y_pred.eq(y).sum().item()
            correct_adv += y_pred_adv.eq(y).sum().item()

            for j in range(len(x)):
                img_name = img_names[j] + '.png'
                img_class = classes[y_classes[j]]
                img_path_adv = os.path.join(new_ds_type_adv_dir, img_class, img_name)
                Image.fromarray(process_tensor(x_adv[j])).save(img_path_adv)
                # upload_file_to_s3(img_path_adv, img_path_adv, s3_resource=s3_resource)

            results_deep_df = deep_evaluation_attack_for_decisioner(experiment=experiment,
                                                                    attack=attack,
                                                                    epsilon=epsilon,
                                                                    ds_name=loader_name,
                                                                    classes=classes,
                                                                    images_paths=paths,
                                                                    outputs_surrogate_orig=outputs_surrogate_orig,
                                                                    outputs_surrogate_adv=outputs_surrogate_adv,
                                                                    labels=y,
                                                                    criterion=criterion,
                                                                    res_deep_df=results_deep_df)

        n = len(loader.dataset)
        print("acc on clean examples (%): {:.3f} ({:.3f})".format(correct / n * 100.0, correct))
        print("acc on {} adversarial examples (%): {:.3f} ({:.3f})".format(attack, correct_adv / n * 100.0,
                                                                           correct_adv))

    return results_deep_df