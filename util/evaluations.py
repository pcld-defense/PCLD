import numpy as np
import pandas as pd
import torch


def evaluate_print_decisioner(class_correct, class_total, loss, epoch,
                   dataset_size, n_classes, classes,
                   epsilon_stats):
    avg_loss = loss / dataset_size
    correct_sum = np.sum(class_correct)
    sample_size = np.sum(class_total)
    accuracy = correct_sum / sample_size

    res_dict = {
        'epoch': [epoch],
        'avg_loss': [avg_loss],
        'accuracy': [accuracy]
    }
    print(f'Epoch %d, loss: %.8f Accuracy (Overall): %2d%% (%2d/%2d)' %(epoch, avg_loss, 100. * accuracy,
                                                                        correct_sum, sample_size))
    print(f'Accuracy by epsilon:')
    for eps in epsilon_stats.keys():
        eps_correct = epsilon_stats[eps][0]
        eps_count = epsilon_stats[eps][1]
        acc = eps_correct/eps_count
        # epsilon_stats[eps][0] = acc
        print(f'eps {eps}: {acc} ({eps_correct} / {eps_count})')



def evaluate_print(experiment, res_df, class_correct, class_total, loss, epoch,
                   ds_type, dataset_size, loader_name, n_classes, classes):
    avg_loss = round(loss / dataset_size, 3)
    correct_sum = np.sum(class_correct)
    sample_size = np.sum(class_total)
    accuracy = correct_sum / sample_size

    res_dict = {
        'experiment': [experiment],
        'epoch': [epoch],
        'ds_name': [loader_name],
        'ds_type': [ds_type],
        'avg_loss': [avg_loss],
        'accuracy': [accuracy]
    }
    print(f'Epoch %d, loss: %.8f \t{loader_name} Accuracy (Overall): %2d%% (%2d/%2d)' %(epoch, avg_loss, 100. * accuracy,
                                                                                        correct_sum, sample_size))
    for i in range(n_classes):
        class_correct_i = np.sum(class_correct[i])
        class_size_i = np.sum(class_total[i])
        class_acc_i = class_correct[i] / class_total[i]
        res_dict['accuracy_' + classes[i]] = [class_acc_i]
        print(f'{ds_type} Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_acc_i,
            class_correct_i, class_size_i))
    res_df = pd.concat([res_df, pd.DataFrame(res_dict)], axis=0, ignore_index=True)
    return res_df


def deep_evaluation_adv_training(experiment, res_deep_df, epoch, ds_name, ds_type,
                                 n_classes, classes, images_paths, outputs, labels,
                                 epsilon, nb_iter, attack_norm):
    res_deep_dict = {
        'experiment': [],
        'epoch': [],
        'ds_name': [],
        'ds_type': [],
        'image_name': [],
        'image_path': [],
        'real_label': [],
        'real_label_name': [],
        'pred_label': [],
        'pred_label_name': []
    }
    res_deep_dict['epsilon'] = [epsilon] * len(images_paths)
    res_deep_dict['nb_iter'] = [nb_iter] * len(images_paths)
    res_deep_dict['attack_norm'] = [attack_norm] * len(images_paths)

    _, pred_labels = torch.max(outputs, 1)
    for i in range(len(images_paths)):
        pred_label = pred_labels[i].item()
        image_path = images_paths[i]
        image_name = image_path.split('/')[-1].split('.')[0]
        real_label = labels[i].item()
        real_label_name = classes[real_label]
        pred_label_name = classes[pred_label]

        res_deep_dict['experiment'].append(experiment)
        res_deep_dict['epoch'].append(epoch)
        res_deep_dict['ds_name'].append(ds_name)
        res_deep_dict['ds_type'].append(ds_type)
        res_deep_dict['image_name'].append(image_name)
        res_deep_dict['image_path'].append(image_path)
        res_deep_dict['real_label'].append(real_label)
        res_deep_dict['real_label_name'].append(real_label_name)
        res_deep_dict['pred_label'].append(pred_label)
        res_deep_dict['pred_label_name'].append(pred_label_name)

    res_deep_df = pd.concat([res_deep_df, pd.DataFrame(res_deep_dict)],
                            axis=0,
                            ignore_index=True)
    return res_deep_df

def deep_evaluation_training(experiment, res_deep_df, epoch, ds_name, ds_type, filter_level,
                             n_classes, classes, images_paths, outputs, labels,
                             criterion, epsilon=None, step=None):
    res_deep_dict = {
        'experiment': [],
        'epoch': [],
        'ds_name': [],
        'ds_type': [],
        'image_name': [],
        'image_path': [],
        'filter_level': [],
        'real_label': [],
        'real_label_name': [],
        'pred_prob_cat': [],
        'pred_prob_dog': [],
        'pred_prob_squirrel': [],
        'pred_label': [],
        'pred_label_name': [],
        'loss': []
    }
    if epsilon:
        res_deep_dict['epsilon'] = [epsilon] * len(images_paths)
    if step:
        res_deep_dict['step'] = [step] * len(images_paths)

    _, pred_labels = torch.max(outputs, 1)
    for i in range(len(images_paths)):
        pred_label = pred_labels[i].item()
        image_path = images_paths[i]
        image_name = image_path.split('/')[-1].split('.')[0]
        real_label = labels[i].item()
        real_label_name = classes[real_label]
        pred_prob_cat = outputs[i][classes.index("cat")].item()
        pred_prob_dog = outputs[i][classes.index("dog")].item()
        pred_prob_squirrel = outputs[i][classes.index("squirrel")].item()
        pred_label_name = classes[pred_label]
        loss = criterion(outputs[i], labels[i]).item()

        res_deep_dict['experiment'].append(experiment)
        res_deep_dict['epoch'].append(epoch)
        res_deep_dict['ds_name'].append(ds_name)
        res_deep_dict['ds_type'].append(ds_type)
        res_deep_dict['image_name'].append(image_name)
        res_deep_dict['image_path'].append(image_path)
        res_deep_dict['filter_level'].append(filter_level)
        res_deep_dict['real_label'].append(real_label)
        res_deep_dict['real_label_name'].append(real_label_name)
        res_deep_dict['pred_prob_cat'].append(pred_prob_cat)
        res_deep_dict['pred_prob_dog'].append(pred_prob_dog)
        res_deep_dict['pred_prob_squirrel'].append(pred_prob_squirrel)
        res_deep_dict['pred_label'].append(pred_label)
        res_deep_dict['pred_label_name'].append(pred_label_name)
        res_deep_dict['loss'].append(loss)

    res_deep_df = pd.concat([res_deep_df, pd.DataFrame(res_deep_dict)],
                            axis=0,
                            ignore_index=True)
    return res_deep_df


def deep_evaluation_defense(experiment, res_deep_df, epoch_surrogate, epoch_victim, epsilon,
                            ds_name, ds_type,
                            n_classes, classes, images_paths,
                            outputs_surrogate,
                            outputs_victim,
                            labels,
                            criterion):

    res_deep_dict = {
        'experiment': [],
        'epoch_surrogate': [],
        'epoch_victim': [],
        'epsilon': [],
        'ds_name': [],
        'ds_type': [],
        'image_name': [],
        'image_path': [],
        'real_label': [],
        'real_label_name': [],
        'surrogate_pred_prob_cat': [],
        'surrogate_pred_prob_dog': [],
        'surrogate_pred_prob_squirrel': [],
        'surrogate_pred_label': [],
        'surrogate_pred_label_name': [],
        'surrogate_loss': [],
        'victim_pred_prob_cat': [],
        'victim_pred_prob_dog': [],
        'victim_pred_prob_squirrel': [],
        'victim_pred_label': [],
        'victim_pred_label_name': [],
        'victim_loss': []
    }

    _, surrogate_pred_labels = torch.max(outputs_surrogate, 1)
    _, victim_pred_labels = torch.max(outputs_victim, 1)
    for i in range(len(images_paths)):
        surrogate_pred_label = surrogate_pred_labels[i].item()
        victim_pred_label= victim_pred_labels[i].item()

        image_path = images_paths[i]
        image_name = image_path.split('/')[-1].split('.')[0]
        real_label = labels[i].item()
        real_label_name = classes[real_label]
        surrogate_pred_prob_cat = outputs_surrogate[i][classes.index("cat")].item()
        surrogate_pred_prob_dog = outputs_surrogate[i][classes.index("dog")].item()
        surrogate_pred_prob_squirrel = outputs_surrogate[i][classes.index("squirrel")].item()

        surrogate_pred_label_name = classes[surrogate_pred_label]
        surrogate_loss = criterion(outputs_surrogate[i], labels[i]).item()

        victim_pred_prob_cat = outputs_victim[i][classes.index("cat")].item()
        victim_pred_prob_dog = outputs_victim[i][classes.index("dog")].item()
        victim_pred_prob_squirrel = outputs_victim[i][classes.index("squirrel")].item()

        victim_pred_label_name = classes[victim_pred_label]
        victim_loss = criterion(outputs_victim[i], labels[i]).item()

        res_deep_dict['experiment'].append(experiment)
        res_deep_dict['epoch_surrogate'].append(epoch_surrogate)
        res_deep_dict['epoch_victim'].append(epoch_victim)
        res_deep_dict['epsilon'].append(epsilon)
        res_deep_dict['ds_name'].append(ds_name)
        res_deep_dict['ds_type'].append(ds_type)
        res_deep_dict['image_name'].append(image_name)
        res_deep_dict['image_path'].append(image_path)
        res_deep_dict['real_label'].append(real_label)
        res_deep_dict['real_label_name'].append(real_label_name)

        res_deep_dict['surrogate_pred_prob_cat'].append(surrogate_pred_prob_cat)
        res_deep_dict['surrogate_pred_prob_dog'].append(surrogate_pred_prob_dog)
        res_deep_dict['surrogate_pred_prob_squirrel'].append(surrogate_pred_prob_squirrel)
        res_deep_dict['surrogate_pred_label'].append(surrogate_pred_label)
        res_deep_dict['surrogate_pred_label_name'].append(surrogate_pred_label_name)
        res_deep_dict['surrogate_loss'].append(surrogate_loss)
        res_deep_dict['victim_pred_prob_cat'].append(victim_pred_prob_cat)
        res_deep_dict['victim_pred_prob_dog'].append(victim_pred_prob_dog)
        res_deep_dict['victim_pred_prob_squirrel'].append(victim_pred_prob_squirrel)
        res_deep_dict['victim_pred_label'].append(victim_pred_label)
        res_deep_dict['victim_pred_label_name'].append(victim_pred_label_name)
        res_deep_dict['victim_loss'].append(victim_loss)

    res_deep_df = pd.concat([res_deep_df, pd.DataFrame(res_deep_dict)],
                            axis=0,
                            ignore_index=True)
    return res_deep_df


def deep_evaluation_attack(experiment, attack, surrogate_name, victim_name, epsilon,
                           ds_name, classes, images_paths,
                           outputs_surrogate_orig, outputs_surrogate_adv,
                           outputs_victim_orig, outputs_victim_adv,
                           labels, criterion, res_deep_df, inference_time):
    # print('Evaluate attack')
    res_deep_dict = {
        'experiment': [],
        'attack': [],
        'surrogate_name': [],
        'victim_name': [],
        'epsilon': [],
        'ds_name': [],
        'image_name': [],
        'image_path': [],
        'real_label': [],
        'real_label_name': [],
        'surrogate_pred_prob_cat_orig': [],
        'surrogate_pred_prob_dog_orig': [],
        'surrogate_pred_prob_squirrel_orig': [],
        'surrogate_pred_prob_cat_adv': [],
        'surrogate_pred_prob_dog_adv': [],
        'surrogate_pred_prob_squirrel_adv': [],
        'surrogate_pred_label_orig': [],
        'surrogate_pred_label_orig_name': [],
        'surrogate_pred_label_adv': [],
        'surrogate_pred_label_adv_name': [],
        'surrogate_loss_orig': [],
        'surrogate_loss_adv': [],
        'victim_pred_prob_cat_orig': [],
        'victim_pred_prob_dog_orig': [],
        'victim_pred_prob_squirrel_orig': [],
        'victim_pred_prob_cat_adv': [],
        'victim_pred_prob_dog_adv': [],
        'victim_pred_prob_squirrel_adv': [],
        'victim_pred_label_orig': [],
        'victim_pred_label_orig_name': [],
        'victim_pred_label_adv': [],
        'victim_pred_label_adv_name': [],
        'victim_loss_orig': [],
        'victim_loss_adv': [],
        'inference_time': []
    }

    _, surrogate_pred_labels_orig = torch.max(outputs_surrogate_orig, 1)
    _, surrogate_pred_labels_adv = torch.max(outputs_surrogate_adv, 1)
    _, victim_pred_labels_orig = torch.max(outputs_victim_orig, 1)
    _, victim_pred_labels_adv = torch.max(outputs_victim_adv, 1)
    for i in range(len(images_paths)):
        surrogate_pred_label_orig = surrogate_pred_labels_orig[i].item()
        surrogate_pred_label_adv = surrogate_pred_labels_adv[i].item()
        victim_pred_label_orig = victim_pred_labels_orig[i].item()
        victim_pred_label_adv = victim_pred_labels_adv[i].item()

        image_path = images_paths[i]
        image_name = image_path.split('/')[-1].split('.')[0]
        real_label = labels[i].item()
        real_label_name = classes[real_label]
        surrogate_pred_prob_cat_orig = outputs_surrogate_orig[i][classes.index("cat")].item()
        surrogate_pred_prob_dog_orig = outputs_surrogate_orig[i][classes.index("dog")].item()
        surrogate_pred_prob_squirrel_orig = outputs_surrogate_orig[i][classes.index("squirrel")].item()
        surrogate_pred_prob_cat_adv = outputs_surrogate_adv[i][classes.index("cat")].item()
        surrogate_pred_prob_dog_adv = outputs_surrogate_adv[i][classes.index("dog")].item()
        surrogate_pred_prob_squirrel_adv = outputs_surrogate_adv[i][classes.index("squirrel")].item()

        surrogate_pred_label_orig_name = classes[surrogate_pred_label_orig]
        surrogate_loss_orig = criterion(outputs_surrogate_orig[i], labels[i]).item()
        surrogate_pred_label_adv_name = classes[surrogate_pred_label_adv]
        surrogate_loss_adv = criterion(outputs_surrogate_adv[i], labels[i]).item()

        victim_pred_prob_cat_orig = outputs_victim_orig[i][classes.index("cat")].item()
        victim_pred_prob_dog_orig = outputs_victim_orig[i][classes.index("dog")].item()
        victim_pred_prob_squirrel_orig = outputs_victim_orig[i][classes.index("squirrel")].item()
        victim_pred_prob_cat_adv = outputs_victim_adv[i][classes.index("cat")].item()
        victim_pred_prob_dog_adv = outputs_victim_adv[i][classes.index("dog")].item()
        victim_pred_prob_squirrel_adv = outputs_victim_adv[i][classes.index("squirrel")].item()

        victim_pred_label_orig_name = classes[victim_pred_label_orig]
        victim_loss_orig = criterion(outputs_victim_orig[i], labels[i]).item()
        victim_pred_label_adv_name = classes[victim_pred_label_adv]
        victim_loss_adv = criterion(outputs_victim_adv[i], labels[i]).item()

        res_deep_dict['experiment'].append(experiment)
        res_deep_dict['attack'].append(attack)
        res_deep_dict['surrogate_name'].append(surrogate_name)
        res_deep_dict['victim_name'].append(victim_name)
        res_deep_dict['epsilon'].append(epsilon)
        res_deep_dict['ds_name'].append(ds_name)
        res_deep_dict['image_name'].append(image_name)
        res_deep_dict['image_path'].append(image_path)
        res_deep_dict['real_label'].append(real_label)
        res_deep_dict['real_label_name'].append(real_label_name)

        res_deep_dict['surrogate_pred_prob_cat_orig'].append(surrogate_pred_prob_cat_orig)
        res_deep_dict['surrogate_pred_prob_dog_orig'].append(surrogate_pred_prob_dog_orig)
        res_deep_dict['surrogate_pred_prob_squirrel_orig'].append(surrogate_pred_prob_squirrel_orig)
        res_deep_dict['surrogate_pred_prob_cat_adv'].append(surrogate_pred_prob_cat_adv)
        res_deep_dict['surrogate_pred_prob_dog_adv'].append(surrogate_pred_prob_dog_adv)
        res_deep_dict['surrogate_pred_prob_squirrel_adv'].append(surrogate_pred_prob_squirrel_adv)
        res_deep_dict['surrogate_pred_label_orig'].append(surrogate_pred_label_orig)
        res_deep_dict['surrogate_pred_label_orig_name'].append(surrogate_pred_label_orig_name)
        res_deep_dict['surrogate_pred_label_adv'].append(surrogate_pred_label_adv)
        res_deep_dict['surrogate_pred_label_adv_name'].append(surrogate_pred_label_adv_name)
        res_deep_dict['surrogate_loss_orig'].append(surrogate_loss_orig)
        res_deep_dict['surrogate_loss_adv'].append(surrogate_loss_adv)
        res_deep_dict['victim_pred_prob_cat_orig'].append(victim_pred_prob_cat_orig)
        res_deep_dict['victim_pred_prob_dog_orig'].append(victim_pred_prob_dog_orig)
        res_deep_dict['victim_pred_prob_squirrel_orig'].append(victim_pred_prob_squirrel_orig)
        res_deep_dict['victim_pred_prob_cat_adv'].append(victim_pred_prob_cat_adv)
        res_deep_dict['victim_pred_prob_dog_adv'].append(victim_pred_prob_dog_adv)
        res_deep_dict['victim_pred_prob_squirrel_adv'].append(victim_pred_prob_squirrel_adv)
        res_deep_dict['victim_pred_label_orig'].append(victim_pred_label_orig)
        res_deep_dict['victim_pred_label_orig_name'].append(victim_pred_label_orig_name)
        res_deep_dict['victim_pred_label_adv'].append(victim_pred_label_adv)
        res_deep_dict['victim_pred_label_adv_name'].append(victim_pred_label_adv_name)
        res_deep_dict['victim_loss_orig'].append(victim_loss_orig)
        res_deep_dict['victim_loss_adv'].append(victim_loss_adv)
        res_deep_dict['inference_time'].append(inference_time)

    res_deep_df = pd.concat([res_deep_df, pd.DataFrame(res_deep_dict)],
                            axis=0,
                            ignore_index=True)
    print('Evaluate attack ended!')
    return res_deep_df



def deep_evaluation_attack_for_decisioner(experiment, attack, surrogate_name, epsilon,
                                          ds_name, classes, images_paths,
                                          outputs_surrogate_orig, outputs_surrogate_adv,
                                          labels, criterion, res_deep_df):

    res_deep_dict = {
        'experiment': [],
        'attack': [],
        'surrogate_name': [],
        'epsilon': [],
        'ds_name': [],
        'image_name': [],
        'image_path': [],
        'real_label': [],
        'real_label_name': [],
        'surrogate_pred_prob_cat_orig': [],
        'surrogate_pred_prob_dog_orig': [],
        'surrogate_pred_prob_squirrel_orig': [],
        'surrogate_pred_prob_cat_adv': [],
        'surrogate_pred_prob_dog_adv': [],
        'surrogate_pred_prob_squirrel_adv': [],
        'surrogate_pred_label_orig': [],
        'surrogate_pred_label_orig_name': [],
        'surrogate_pred_label_adv': [],
        'surrogate_pred_label_adv_name': [],
        'surrogate_loss_orig': [],
        'surrogate_loss_adv': []
    }

    _, surrogate_pred_labels_orig = torch.max(outputs_surrogate_orig, 1)
    _, surrogate_pred_labels_adv = torch.max(outputs_surrogate_adv, 1)
    for i in range(len(images_paths)):
        surrogate_pred_label_orig = surrogate_pred_labels_orig[i].item()
        surrogate_pred_label_adv = surrogate_pred_labels_adv[i].item()

        image_path = images_paths[i]
        image_name = image_path.split('/')[-1].split('.')[0]
        real_label = labels[i].item()
        real_label_name = classes[real_label]
        surrogate_pred_prob_cat_orig = outputs_surrogate_orig[i][classes.index("cat")].item()
        surrogate_pred_prob_dog_orig = outputs_surrogate_orig[i][classes.index("dog")].item()
        surrogate_pred_prob_squirrel_orig = outputs_surrogate_orig[i][classes.index("squirrel")].item()
        surrogate_pred_prob_cat_adv = outputs_surrogate_adv[i][classes.index("cat")].item()
        surrogate_pred_prob_dog_adv = outputs_surrogate_adv[i][classes.index("dog")].item()
        surrogate_pred_prob_squirrel_adv = outputs_surrogate_adv[i][classes.index("squirrel")].item()

        surrogate_pred_label_orig_name = classes[surrogate_pred_label_orig]
        surrogate_loss_orig = criterion(outputs_surrogate_orig[i], labels[i]).item()
        surrogate_pred_label_adv_name = classes[surrogate_pred_label_adv]
        surrogate_loss_adv = criterion(outputs_surrogate_adv[i], labels[i]).item()

        res_deep_dict['experiment'].append(experiment)
        res_deep_dict['attack'].append(attack)
        res_deep_dict['surrogate_name'].append(surrogate_name)
        res_deep_dict['epsilon'].append(epsilon)
        res_deep_dict['ds_name'].append(ds_name)
        res_deep_dict['image_name'].append(image_name)
        res_deep_dict['image_path'].append(image_path)
        res_deep_dict['real_label'].append(real_label)
        res_deep_dict['real_label_name'].append(real_label_name)

        res_deep_dict['surrogate_pred_prob_cat_orig'].append(surrogate_pred_prob_cat_orig)
        res_deep_dict['surrogate_pred_prob_dog_orig'].append(surrogate_pred_prob_dog_orig)
        res_deep_dict['surrogate_pred_prob_squirrel_orig'].append(surrogate_pred_prob_squirrel_orig)
        res_deep_dict['surrogate_pred_prob_cat_adv'].append(surrogate_pred_prob_cat_adv)
        res_deep_dict['surrogate_pred_prob_dog_adv'].append(surrogate_pred_prob_dog_adv)
        res_deep_dict['surrogate_pred_prob_squirrel_adv'].append(surrogate_pred_prob_squirrel_adv)
        res_deep_dict['surrogate_pred_label_orig'].append(surrogate_pred_label_orig)
        res_deep_dict['surrogate_pred_label_orig_name'].append(surrogate_pred_label_orig_name)
        res_deep_dict['surrogate_pred_label_adv'].append(surrogate_pred_label_adv)
        res_deep_dict['surrogate_pred_label_adv_name'].append(surrogate_pred_label_adv_name)
        res_deep_dict['surrogate_loss_orig'].append(surrogate_loss_orig)
        res_deep_dict['surrogate_loss_adv'].append(surrogate_loss_adv)

    res_deep_df = pd.concat([res_deep_df, pd.DataFrame(res_deep_dict)],
                            axis=0,
                            ignore_index=True)
    return res_deep_df


def deep_evaluation_general(experiment, res_deep_df, dir,
                            ds_name, classes, images_paths,
                            outputs, labels, criterion):

    res_deep_dict = {
        'experiment': [],
        'dir': [],
        'ds_name': [],
        'image_name': [],
        'image_path': [],
        'real_label': [],
        'real_label_name': [],
        'pred_prob_cat': [],
        'pred_prob_dog': [],
        'pred_prob_squirrel': [],
        'pred_label': [],
        'pred_label_name': [],
        'loss': []
    }

    _, pred_labels = torch.max(outputs, 1)
    for i in range(len(images_paths)):
        pred_label = pred_labels[i].item()

        image_path = images_paths[i]
        image_name = image_path.split('/')[-1].split('.')[0]
        real_label = labels[i].item()
        real_label_name = classes[real_label]
        pred_prob_cat = outputs[i][classes.index("cat")].item()
        pred_prob_dog = outputs[i][classes.index("dog")].item()
        pred_prob_squirrel = outputs[i][classes.index("squirrel")].item()

        pred_label_name = classes[pred_label]
        loss = criterion(outputs[i], labels[i]).item()

        res_deep_dict['experiment'].append(experiment)
        res_deep_dict['dir'].append(dir)
        res_deep_dict['ds_name'].append(ds_name)
        res_deep_dict['image_name'].append(image_name)
        res_deep_dict['image_path'].append(image_path)
        res_deep_dict['real_label'].append(real_label)
        res_deep_dict['real_label_name'].append(real_label_name)

        res_deep_dict['pred_prob_cat'].append(pred_prob_cat)
        res_deep_dict['pred_prob_dog'].append(pred_prob_dog)
        res_deep_dict['pred_prob_squirrel'].append(pred_prob_squirrel)
        res_deep_dict['pred_label'].append(pred_label)
        res_deep_dict['pred_label_name'].append(pred_label_name)
        res_deep_dict['loss'].append(loss)

    res_deep_df = pd.concat([res_deep_df, pd.DataFrame(res_deep_dict)],
                            axis=0,
                            ignore_index=True)
    return res_deep_df