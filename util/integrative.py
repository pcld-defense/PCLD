import argparse


def str_to_int_list(str_var, sep):
    return [int(v) for v in str_var.split(sep)] if sep in str_var else [int(str_var)]


def str_to_float_list(str_var, sep):
    return [float(v) for v in str_var.split(sep)] if sep in str_var else [float(str_var)]


def parse_args():
    parser = argparse.ArgumentParser(description='Parse the experiment arguments')
    parser.add_argument('--experiment_type', '-ext', type=str, required=True,
                        help='the type of the experiment to run')
    parser.add_argument('--experiment_suff', '-exs', type=str, required=True, default='test',
                        help='added name to the experiment')
    parser.add_argument('--dataset', '-dta', type=str, required=True, help='the main dataset')
    parser.add_argument('--batch_size', '-bsz', type=int, required=False, default=16, help='batch size')
    parser.add_argument('--output_every', '-oev', type=str, required=False,
                        default="50,100,200,300,400,500,600,700,950,1200,1700,2200,3200,4200,5200",
                        help='the selection of paint steps (t)')
    parser.add_argument('--max_epochs', '-mxp', type=int, required=False, default=51,
                        help='max epochs for training the model')
    parser.add_argument('--find_best_epoch', '-fbp', type=int, required=False, default=0,
                        help='whether to apply overfitting detection with train-validation phase')
    # Pre-trained clf
    parser.add_argument('--classifier_experiment', '-clx', type=str, required=False,
                        help='the pre-trained classifier folder')
    # Pre-trained decisioner
    parser.add_argument('--decisioner_experiment', '-dcx', type=str, required=False,
                        help='the pre-trained decisioner folder')
    # decisioner architechture
    parser.add_argument('--decisioner_architechture', '-dca', type=str, required=False,
                        help='conv/fc')
    # Attack Args
    parser.add_argument('--epsilons', '-eps', type=str, required=False, default='8', help='attack epsilon')
    parser.add_argument('--attack', '-atk', type=str, required=False, default='pgd',
                        help='attack name (fgsm/pgd/cw/aa)')
    parser.add_argument('--attack_direction', '-atd', type=str, required=False, default='untargeted',
                        help='untargeted/targeted')
    parser.add_argument('--attack_nb_iter', '-atn', type=int, required=False, default=10,
                        help='attack iterations')
    parser.add_argument('--run_naive_attack', '-rna', type=int, required=False, default=0,
                        help='whether to run naive attack in addition to the adaptive attack')
    parser.add_argument('--attack_train', '-att', type=int, required=False, default=0,
                        help='whether to attack the train and the validation sets')

    parsed = parser.parse_args()

    # parsed.epsilons = str_to_float_list(parsed.epsilons, '|')
    parsed.output_every = str_to_int_list(parsed.output_every, ',')
    parsed.epsilons = str_to_int_list(parsed.epsilons, '|')

    return parsed

