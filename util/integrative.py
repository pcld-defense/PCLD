import argparse


def str_to_int_list(str_var, sep):
    return [int(v) for v in str_var.split(sep)] if sep in str_var else [int(str_var)]


def str_to_float_list(str_var, sep):
    return [float(v) for v in str_var.split(sep)] if sep in str_var else [float(str_var)]


def parse_args():
    parser = argparse.ArgumentParser(description='Parse the experiment arguments')
    parser.add_argument('--experiment_type', '-ext', type=str, required=True, help='the type of the experiment to run')
    parser.add_argument('--experiment_name', '-exn', type=str, required=True, default='test', help='added name to the experiment')
    parser.add_argument('--ds_local_path', '-dsm', type=str, required=True, help='the main dataset')
    parser.add_argument('--batch_size', '-bsz', type=int, required=False, help='batch size for training nets')
    parser.add_argument('--lr', '-lr', type=float, required=False, help='learning rate for training nets')
    parser.add_argument('--n_epochs', '-nep', type=int, required=False, help='num epochs for training nets')
    parser.add_argument('--ds_victim_local_path', '-dsv', type=str, required=False, help='the victim dataset')
    parser.add_argument('--net_victim_local_dir', '-nvd', type=str, required=False, help='the victim net directory')
    parser.add_argument('--net_surrogate_local_dir', '-nsd', type=str, required=False, help='the surrogate net directory')
    parser.add_argument('--ds_decisioner_local_path', '-dsd', type=str, required=False, help='the decisioner predictions path')
    parser.add_argument('--model_decisioner_local_path', '-mld', type=str, required=False, help='the decisioner model path')
    parser.add_argument('--attack', '-att', type=str, required=False, help='the attack to use (fgsm/pgd/deepfool)')
    parser.add_argument('--epsilons', '-eps', type=str, required=False, help='the attack magnitudes')
    parser.add_argument('--save_every', '-sve', type=str, required=False, help='the paint steps (t) to save')
    parsed = parser.parse_args()

    if parsed.epsilons:
        if '.' in parsed.epsilons:
            parsed.epsilons = str_to_float_list(parsed.epsilons, '|')
        else:
            parsed.epsilons = str_to_int_list(parsed.epsilons, '|')

    return parsed

