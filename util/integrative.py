import argparse


def str_to_int_list(str_var: str, sep: str) -> list[int]:
    """Splits a delimited string into a list of integers.

    Args:
        str_var: Input string, e.g. '50,100,200' or '8'.
        sep: Delimiter character used to split the string.

    Returns:
        List of integers parsed from the split string. Returns a single-element
        list when the separator is not present in the string.
    """
    return [int(v) for v in str_var.split(sep)] if sep in str_var else [int(str_var)]


def str_to_float_list(str_var: str, sep: str) -> list[float]:
    """Splits a delimited string into a list of floats.

    Args:
        str_var: Input string, e.g. '0.03|0.06' or '0.01'.
        sep: Delimiter character used to split the string.

    Returns:
        List of floats parsed from the split string. Returns a single-element
        list when the separator is not present in the string.
    """
    return [float(v) for v in str_var.split(sep)] if sep in str_var else [float(str_var)]


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the PCLD experiment runner.

    Defines all arguments required to configure any experiment type
    (paint_dataset, train_classifier, attack_pcl, train_decisioner,
    attack_pcld). After parsing, converts `output_every` from a
    comma-separated string to a list of ints and `epsilons` from a
    pipe-separated string to a list of ints.

    Returns:
        Populated Namespace with all experiment configuration fields.
    """
    parser = argparse.ArgumentParser(description='Parse the experiment arguments')
    parser.add_argument('--experiment_type', '-ext', type=str, required=True,
                        help='the type of the experiment to run')
    parser.add_argument('--experiment_name', '-exn', type=str, required=True, default='test',
                        help='added name to the experiment')

    ### DATASET
    parser.add_argument('--dataset', '-dta', type=str, required=True, help='the main dataset')
    parser.add_argument('--dataset_type', '-dtat', type=str, required=True,
                        help='the dataset type (cifar10 or imagenet')
    parser.add_argument('--splits', '-sp', type=str, nargs='+', required=True,
                        help='dataset type (e.g. train val test)')
    parser.add_argument('--batch_size', '-bsz', type=int, required=False, default=16, help='batch size')

    ### PAINTER
    parser.add_argument('--output_every', '-oev', type=str, required=False,
                        default="50,100,200,300,400,500,600,700,950,1200,1700,2200,3200,4200,5200",
                        help='the selection of paint steps (t)')

    ### Classifier
    parser.add_argument('--model_type', '-mt', type=str, required=False, default="resnet18",
                        help='architecture type (e.g. resnet18)')
    parser.add_argument('--pretrained_weights', '-ptw', type=str, required=False,
                        help='pretrained weights path')
    parser.add_argument('--max_epochs', '-mxp', type=int, required=False, default=51,
                        help='max epochs for training the model')
    parser.add_argument('--lr', '-lr', type=float, required=False, default=0.01,
                        help='learning rate')
    parser.add_argument('--patience', '-pat', type=int, required=False, default=5,
                        help='patience for early stopping')
    parser.add_argument('--classifier_experiment', '-clx', type=str, required=False,
                        help='the pre-trained classifier folder')
    parser.add_argument('--decisioner_experiment', '-dcx', type=str, required=False,
                        help='the pre-trained decisioner folder')
    parser.add_argument('--decisioner_architechture', '-dca', type=str, required=False,
                        help='conv/fc')

    ### Attack
    parser.add_argument('--epsilons', '-eps', type=str, required=False, default='8', help='attack epsilon')
    parser.add_argument('--attack', '-atk', type=str, required=False, default='pgd',
                        help='attack name (fgsm/pgd/cw/aa)')
    parser.add_argument('--attack_direction', '-atd', type=str, required=False, default='untargeted',
                        help='untargeted/targeted')
    parser.add_argument('--attack_nb_iter', '-atn', type=int, required=False, default=10,
                        help='attack iterations')
    parser.add_argument('--run_naive_attack', '-rna', type=int, required=False, default=0,
                        help='whether to run naive attack in addition to the adaptive attack')

    parsed = parser.parse_args()

    parsed.output_every = str_to_int_list(parsed.output_every, ',')
    parsed.epsilons = str_to_int_list(parsed.epsilons, '|')

    if isinstance(parsed.splits, str):
        parsed.splits = [s.strip() for s in parsed.splits.split(',')]
    else:
        parsed.splits = list(parsed.splits)

    return parsed
