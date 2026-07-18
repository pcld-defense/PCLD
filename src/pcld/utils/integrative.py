import argparse
import json
import os


def save_args_json(args: argparse.Namespace, output_dir: str) -> None:
    """Saves the experiment argument namespace to a JSON file in output_dir.

    Args:
        args: Parsed argument namespace to serialise.
        output_dir: Directory where ``args.json`` will be written.
    """
    path = os.path.join(output_dir, 'args.json')
    with open(path, 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f'Saved args to {path}')


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


def str_to_num_list(str_var: str, sep: str) -> list:
    """Splits a delimited string into numbers, keeping ints as ints.

    Integral values become ints (matching the legacy int-only parsing);
    non-integral values (e.g. l2 epsilon budgets like '0.5') stay floats.

    Args:
        str_var: Input string, e.g. '8', '3|9', or '0.5|1.5'.
        sep: Delimiter character used to split the string.

    Returns:
        List of ints/floats parsed from the split string. Returns a
        single-element list when the separator is not present in the string.
    """
    def _num(v: str):
        f = float(v)
        return int(f) if f.is_integer() else f

    return [_num(v) for v in str_var.split(sep)] if sep in str_var else [_num(str_var)]


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
    pipe-separated string to a list of numbers (ints, with non-integral
    l2 budgets preserved as floats).

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
    parser.add_argument('--num_samples', '-nsm', type=int, required=False, default=None,
                        help='cap evaluation to the first N samples per split (None = full split)')
    parser.add_argument('--data_source', '-dsc', type=str, required=False, default='folder',
                        help='folder | robustbench (fixed deterministic test-set prefix)')

    ### PAINTER
    parser.add_argument('--output_every', '-oev', type=str, required=False,
                        default="50,100,200,300,400,500,600,700,950,1200,1700,2200,3200,4200,5200",
                        help='the selection of paint steps (t)')
    parser.add_argument('--painter_max_step', '-pms', type=int, required=False, default=80,
                        help='painter total step budget (halved per phase when divide > 1)')
    parser.add_argument('--painter_divide', '-pdv', type=int, required=False, default=5,
                        help='painter Phase-2 patch-grid side length (1 disables Phase 2)')

    ### Classifier
    parser.add_argument('--preprocessing', '-pre', type=str, required=False, default=None,
                        help='eval preprocessing key from PREPROCESSINGS dict '
                             '(e.g. Res256Crop224, Res224, BicubicRes256Crop224); '
                             'None = plain ToTensor + Normalize')
    parser.add_argument('--train_preprocessing', '-tpre', type=str, required=False, default=None,
                        help='train preprocessing key from PREPROCESSINGS dict '
                             '(e.g. augmented); None = plain ToTensor + Normalize')
    parser.add_argument('--model_type', '-mt', type=str, required=False, default='wrn-70-16',
                        help='architecture type: wrn-70-16 | wrn-34-10 | xcit-m12')
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
    parser.add_argument('--find_best_epoch', '-fbe', type=int, required=False, default=1,
                        help='1 = run train/val phase to find best epoch before final training')
    parser.add_argument('--max_train_epsilon', '-mte', type=int, required=False, default=20,
                        help='filter training data to epsilon <= this value; set to -1 to disable')

    ### Attack
    parser.add_argument('--epsilons', '-eps', type=str, required=False, default='8',
                        help="'|'-separated attack epsilons. For linf: pixel-space "
                             "ints divided by 255 internally (e.g. '3|9'). For l2: "
                             "absolute budgets, non-integral floats allowed (e.g. '0.5').")
    parser.add_argument('--attack', '-atk', type=str, required=False, default='pgd',
                        help='attack name (fgsm/pgd/cw/aa)')
    parser.add_argument('--attack_direction', '-atd', type=str, required=False, default='untargeted',
                        help='untargeted/targeted')
    parser.add_argument('--attack_nb_iter', '-atn', type=int, required=False, default=10,
                        help='attack iterations')
    parser.add_argument('--run_naive_attack', '-rna', type=int, required=False, default=0,
                        help='whether to run naive attack in addition to the adaptive attack')
    parser.add_argument('--save_parquet', '-spq', type=int, required=False, default=1,
                        help='1 = save full Parquet with softmaxes (default); '
                             '0 = save lightweight CSV without softmax vectors')
    parser.add_argument('--attack_norm', '-anm', type=str, required=False, default='linf',
                        help='attack norm: linf or l2')
    parser.add_argument('--targeted_jumps_allowed', '-tja', type=int, required=False, default=6,
                        help='max random class offset for targeted attack label generation; '
                             'use 6 for ImageNet (1000 classes), 1 for CIFAR-10 (10 classes)')
    parser.add_argument('--attack_nb_restarts', '-anr', type=int, required=False, default=1,
                        help='number of random PGD restarts; the adversarial example with '
                             'the highest loss is kept. Default 1 (no restarts).')
    parser.add_argument('--multi_step_loss_weight', '-msl', type=float, required=False, default=0.0,
                        help='weight λ for the intermediate per-step cross-entropy loss added '
                             'to the decisioner CE in the PCLD attack. 0.0 disables it. '
                             'Recommended range: 0.1–0.5.')
    parser.add_argument('--eot_samples', '-eot', type=int, required=False, default=1,
                        help='number of EOT gradient samples to average per PGD step. '
                             '1 = no EOT (default). Higher values reduce surrogate '
                             'approximation noise at the cost of runtime.')
    parser.add_argument('--use_apgd', '-apgd', type=int, required=False, default=0,
                        help='1 = enable APGD adaptive step-size schedule for PGD '
                             '(halves alpha when loss stalls). 0 = fixed step size (default).')
    parser.add_argument('--aa_version', '-aav', type=str, required=False, default='standard',
                        help='AutoAttack ensemble version: standard | rand. '
                             '"rand" uses the EOT-based ensemble for randomised defenses.')
    parser.add_argument('--resume', '-res', type=int, required=False, default=0,
                        help='1 = resume training from an existing checkpoint (used by '
                             'train_surrogate_painter). 0 = start fresh (default).')
    parser.add_argument('--joint_surrogate', '-js', type=str, required=False, default=None,
                        help='Path to a JointPainterSurrogate .pth file. When provided, '
                             'the joint model at that path is used/saved instead of separate '
                             'per-step PainterSurrogate_ models. Omit (default) to use separate models.')
    parser.add_argument('--surrogate_type', '-st', type=str, required=False, default='learned',
                        help='learned | straight_through. "straight_through" uses '
                             'AllIdentitySurrogate as the BPDA backward (no surrogate '
                             'checkpoints needed); "learned" (default) uses trained surrogates.')

    ### Gradient-validity battery (R01 gate)
    parser.add_argument('--battery_epsilons', '-bep', type=str, required=False, default='8,16,32,64',
                        help='comma-separated integer epsilon budgets (per 255) for the '
                             'gradient-battery epsilon-to-zero sweep.')
    parser.add_argument('--battery_fd_dirs', '-bfd', type=int, required=False, default=8,
                        help='number of random unit directions probed by the '
                             'gradient-battery finite-difference check.')
    parser.add_argument('--battery_fd_delta', '-bfdl', type=float, required=False, default=1e-3,
                        help='central-difference step size for the gradient-battery '
                             'finite-difference check.')

    parsed = parser.parse_args()

    parsed.output_every = str_to_int_list(parsed.output_every, ',')
    parsed.epsilons = str_to_num_list(parsed.epsilons, '|')
    parsed.battery_epsilons = str_to_int_list(parsed.battery_epsilons, ',')

    if isinstance(parsed.splits, str):
        parsed.splits = [s.strip() for s in parsed.splits.split(',')]
    else:
        parsed.splits = list(parsed.splits)

    return parsed
