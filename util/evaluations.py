import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


def evaluate_print_decisioner(class_correct: list, class_total: list,
                               loss: float, epoch: int, dataset_size: int,
                               n_classes: int, classes: list[str],
                               epsilon_stats: dict) -> None:
    """Prints decisioner training/evaluation metrics to stdout.

    Computes overall accuracy and per-epsilon accuracy from the running
    counters and prints a formatted summary for the current epoch.

    Args:
        class_correct: Per-class count of correct predictions, length n_classes.
        class_total: Per-class count of total samples, length n_classes.
        loss: Accumulated (summed) loss over the entire epoch.
        epoch: Current epoch index.
        dataset_size: Number of batches in the loader (used to compute avg loss).
        n_classes: Number of output classes.
        classes: Sorted list of class name strings.
        epsilon_stats: Dict mapping epsilon value → [correct_count, total_count].
    """
    avg_loss = loss / dataset_size
    correct_sum = np.sum(class_correct)
    sample_size = np.sum(class_total)
    accuracy = correct_sum / sample_size

    print(f'Epoch %d, loss: %.8f Accuracy (Overall): %2d%% (%2d/%2d)' % (
        epoch, avg_loss, 100. * accuracy, correct_sum, sample_size))
    print(f'Accuracy by epsilon:')
    for eps in epsilon_stats.keys():
        eps_correct = epsilon_stats[eps][0]
        eps_count = epsilon_stats[eps][1]
        acc = eps_correct / eps_count
        print(f'eps {eps}: {acc} ({eps_correct} / {eps_count})')


def plot_loss_and_acc(df: pd.DataFrame, output_path: str) -> None:
    """Saves training and validation loss/accuracy plots to disk.

    Generates two PNG files in `output_path`:
    - `average_loss.png`: train vs. validation average loss per epoch.
    - `accuracy_per_epoch.png`: train vs. validation accuracy per epoch.

    Args:
        df: Results DataFrame containing columns 'ds_type', 'epoch',
            'avg_loss', and 'accuracy'. Rows with ds_type='train' and
            ds_type='validation' are plotted separately.
        output_path: Directory path where the PNG files will be saved.
    """
    train_df = df[df['ds_type'] == 'train']
    val_df = df[df['ds_type'] == 'validation']

    plt.figure(figsize=(10, 6))
    plt.plot(train_df['epoch'], train_df['avg_loss'], label='Train Loss', marker='o')
    plt.plot(val_df['epoch'], val_df['avg_loss'], label='Validation Loss', marker='o')
    plt.title('Average Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_path}/average_loss.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(train_df['epoch'], train_df['accuracy'], label='Train Accuracy', marker='o')
    plt.plot(val_df['epoch'], val_df['accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_path}/accuracy_per_epoch.png')
    plt.close()

    print("Plots generated: average_loss.png, accuracy_per_epoch.png")


def evaluate_print(experiment: str, res_df: pd.DataFrame,
                   class_correct: list, class_total: list, loss: float,
                   epoch: int, ds_type: str, dataset_size: int,
                   loader_name: str, n_classes: int,
                   classes: list[str]) -> pd.DataFrame:
    """Prints classifier metrics and appends the epoch results to a DataFrame.

    Computes overall and per-class accuracy, prints a formatted summary, and
    concatenates the new metrics row to `res_df`.

    Args:
        experiment: Experiment name stored in the results row.
        res_df: Existing results DataFrame to append to.
        class_correct: Per-class correct-prediction counts, length n_classes.
        class_total: Per-class total sample counts, length n_classes.
        loss: Accumulated loss over the epoch.
        epoch: Current epoch index.
        ds_type: Split label (e.g. 'train' or 'validation').
        dataset_size: Number of batches (denominator for avg loss).
        loader_name: Human-readable loader name printed in the summary.
        n_classes: Number of output classes.
        classes: Sorted list of class name strings.

    Returns:
        Updated results DataFrame with the current epoch metrics appended.
    """
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
    print(
        f'Epoch %d, loss: %.8f \t{loader_name} Accuracy (Overall): %2d%% (%2d/%2d)' % (
            epoch, avg_loss, 100. * accuracy, correct_sum, sample_size))
    for i in range(n_classes):
        class_correct_i = np.sum(class_correct[i])
        class_size_i = np.sum(class_total[i])
        class_acc_i = class_correct[i] / class_total[i]
        res_dict['accuracy_' + classes[i]] = [class_acc_i]
        print(f'{ds_type} Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_acc_i, class_correct_i, class_size_i))
    res_df = pd.concat([res_df, pd.DataFrame(res_dict)], axis=0, ignore_index=True)
    return res_df


def deep_evaluation_adv_training(experiment: str, res_deep_df: pd.DataFrame,
                                  epoch: int, ds_name: str, ds_type: str,
                                  n_classes: int, classes: list[str],
                                  images_paths: list[str],
                                  outputs: torch.Tensor, labels: torch.Tensor,
                                  epsilon: float, nb_iter: int,
                                  attack_norm: str) -> pd.DataFrame:
    """Records per-image predictions from adversarial training into a DataFrame.

    For each image in the batch, stores the true label, predicted label, and
    attack configuration, then concatenates the new rows to `res_deep_df`.

    Args:
        experiment: Experiment name stored in every row.
        res_deep_df: Existing deep-evaluation DataFrame to append to.
        epoch: Current epoch index.
        ds_name: Dataset loader name.
        ds_type: Split label (e.g. 'train' or 'test').
        n_classes: Number of output classes.
        classes: Sorted list of class name strings.
        images_paths: List of image file paths for the current batch.
        outputs: Model logits of shape (B, num_classes).
        labels: Ground-truth label tensor of shape (B,).
        epsilon: L-inf perturbation budget used in the attack.
        nb_iter: Number of attack iterations.
        attack_norm: Attack norm string (e.g. 'Linf').

    Returns:
        Updated DataFrame with one new row per image in the batch.
    """
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
                            axis=0, ignore_index=True)
    return res_deep_df


def deep_evaluation_defense(experiment: str, res_deep_df: pd.DataFrame,
                             epoch_surrogate: int, epoch_victim: int,
                             epsilon: float, ds_name: str, ds_type: str,
                             n_classes: int, classes: list[str],
                             images_paths: list[str],
                             outputs_surrogate: torch.Tensor,
                             outputs_victim: torch.Tensor,
                             labels: torch.Tensor,
                             criterion) -> pd.DataFrame:
    """Records per-image defence metrics comparing a surrogate and victim model.

    Stores probabilities, predicted labels, and per-sample losses for both the
    surrogate and victim models, then appends to `res_deep_df`.

    Args:
        experiment: Experiment name stored in every row.
        res_deep_df: Existing deep-evaluation DataFrame to append to.
        epoch_surrogate: Surrogate model training epoch.
        epoch_victim: Victim model training epoch.
        epsilon: L-inf perturbation budget.
        ds_name: Dataset loader name.
        ds_type: Split label.
        n_classes: Number of output classes.
        classes: Sorted list of class name strings.
        images_paths: File paths for the current batch.
        outputs_surrogate: Surrogate model logits of shape (B, num_classes).
        outputs_victim: Victim model logits of shape (B, num_classes).
        labels: Ground-truth label tensor of shape (B,).
        criterion: Loss function used to compute per-sample loss values.

    Returns:
        Updated DataFrame with one new row per image in the batch.
    """
    res_deep_dict = {
        'experiment': [], 'epoch_surrogate': [], 'epoch_victim': [],
        'epsilon': [], 'ds_name': [], 'ds_type': [],
        'image_name': [], 'image_path': [],
        'real_label': [], 'real_label_name': [],
        'surrogate_pred_prob_cat': [], 'surrogate_pred_prob_dog': [],
        'surrogate_pred_prob_squirrel': [], 'surrogate_pred_label': [],
        'surrogate_pred_label_name': [], 'surrogate_loss': [],
        'victim_pred_prob_cat': [], 'victim_pred_prob_dog': [],
        'victim_pred_prob_squirrel': [], 'victim_pred_label': [],
        'victim_pred_label_name': [], 'victim_loss': []
    }

    _, surrogate_pred_labels = torch.max(outputs_surrogate, 1)
    _, victim_pred_labels = torch.max(outputs_victim, 1)
    for i in range(len(images_paths)):
        surrogate_pred_label = surrogate_pred_labels[i].item()
        victim_pred_label = victim_pred_labels[i].item()
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
                            axis=0, ignore_index=True)
    return res_deep_df


def deep_evaluation_attack(experiment: str, attack: str, surrogate_name: str,
                            victim_name: str, epsilon: float, ds_name: str,
                            classes: list[str], images_paths: list[str],
                            outputs_surrogate_orig: torch.Tensor,
                            outputs_surrogate_adv: torch.Tensor,
                            outputs_victim_orig: torch.Tensor,
                            outputs_victim_adv: torch.Tensor,
                            labels: torch.Tensor, criterion,
                            res_deep_df: pd.DataFrame,
                            inference_time: float) -> pd.DataFrame:
    """Records per-image attack outcomes for both surrogate and victim models.

    Captures clean and adversarial probabilities, predicted labels, and losses
    for the surrogate and victim under attack, then appends to `res_deep_df`.

    Args:
        experiment: Experiment name stored in every row.
        attack: Attack algorithm name (e.g. 'pgd').
        surrogate_name: Identifier of the surrogate model.
        victim_name: Identifier of the victim model.
        epsilon: L-inf perturbation budget.
        ds_name: Dataset loader name.
        classes: Sorted list of class name strings.
        images_paths: File paths for the current batch.
        outputs_surrogate_orig: Surrogate logits on clean inputs, shape (B, C).
        outputs_surrogate_adv: Surrogate logits on adversarial inputs, shape (B, C).
        outputs_victim_orig: Victim logits on clean inputs, shape (B, C).
        outputs_victim_adv: Victim logits on adversarial inputs, shape (B, C).
        labels: Ground-truth label tensor of shape (B,).
        criterion: Loss function used to compute per-sample loss values.
        res_deep_df: Existing deep-evaluation DataFrame to append to.
        inference_time: Wall-clock seconds for the inference step.

    Returns:
        Updated DataFrame with one new row per image in the batch.
    """
    res_deep_dict = {
        'experiment': [], 'attack': [], 'surrogate_name': [], 'victim_name': [],
        'epsilon': [], 'ds_name': [], 'image_name': [], 'image_path': [],
        'real_label': [], 'real_label_name': [],
        'surrogate_pred_prob_cat_orig': [], 'surrogate_pred_prob_dog_orig': [],
        'surrogate_pred_prob_squirrel_orig': [], 'surrogate_pred_prob_cat_adv': [],
        'surrogate_pred_prob_dog_adv': [], 'surrogate_pred_prob_squirrel_adv': [],
        'surrogate_pred_label_orig': [], 'surrogate_pred_label_orig_name': [],
        'surrogate_pred_label_adv': [], 'surrogate_pred_label_adv_name': [],
        'surrogate_loss_orig': [], 'surrogate_loss_adv': [],
        'victim_pred_prob_cat_orig': [], 'victim_pred_prob_dog_orig': [],
        'victim_pred_prob_squirrel_orig': [], 'victim_pred_prob_cat_adv': [],
        'victim_pred_prob_dog_adv': [], 'victim_pred_prob_squirrel_adv': [],
        'victim_pred_label_orig': [], 'victim_pred_label_orig_name': [],
        'victim_pred_label_adv': [], 'victim_pred_label_adv_name': [],
        'victim_loss_orig': [], 'victim_loss_adv': [], 'inference_time': []
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
                            axis=0, ignore_index=True)
    print('Evaluate attack ended!')
    return res_deep_df


def deep_evaluation_attack_for_decisioner(experiment: str, attack: str,
                                           surrogate_name: str, epsilon: float,
                                           ds_name: str, classes: list[str],
                                           images_paths: list[str],
                                           outputs_surrogate_orig: torch.Tensor,
                                           outputs_surrogate_adv: torch.Tensor,
                                           labels: torch.Tensor, criterion,
                                           res_deep_df: pd.DataFrame) -> pd.DataFrame:
    """Records per-image surrogate attack outcomes for decisioner dataset creation.

    Captures clean and adversarial probabilities and losses from the surrogate
    model only, intended for building the dataset used to train the decisioner.

    Args:
        experiment: Experiment name stored in every row.
        attack: Attack algorithm name.
        surrogate_name: Identifier of the surrogate model.
        epsilon: L-inf perturbation budget.
        ds_name: Dataset loader name.
        classes: Sorted list of class name strings.
        images_paths: File paths for the current batch.
        outputs_surrogate_orig: Surrogate logits on clean inputs, shape (B, C).
        outputs_surrogate_adv: Surrogate logits on adversarial inputs, shape (B, C).
        labels: Ground-truth label tensor of shape (B,).
        criterion: Loss function for per-sample loss computation.
        res_deep_df: Existing DataFrame to append to.

    Returns:
        Updated DataFrame with one new row per image in the batch.
    """
    res_deep_dict = {
        'experiment': [], 'attack': [], 'surrogate_name': [], 'epsilon': [],
        'ds_name': [], 'image_name': [], 'image_path': [],
        'real_label': [], 'real_label_name': [],
        'surrogate_pred_prob_cat_orig': [], 'surrogate_pred_prob_dog_orig': [],
        'surrogate_pred_prob_squirrel_orig': [], 'surrogate_pred_prob_cat_adv': [],
        'surrogate_pred_prob_dog_adv': [], 'surrogate_pred_prob_squirrel_adv': [],
        'surrogate_pred_label_orig': [], 'surrogate_pred_label_orig_name': [],
        'surrogate_pred_label_adv': [], 'surrogate_pred_label_adv_name': [],
        'surrogate_loss_orig': [], 'surrogate_loss_adv': []
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
                            axis=0, ignore_index=True)
    return res_deep_df
