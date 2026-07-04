"""CLI to fetch/cache a registered dataset into the datasets root.

    python scripts/data/download.py --dataset cifar10 --splits train test
    python scripts/data/download.py --dataset cifar10 --root /data/pcld

Datasets that are not auto-downloadable (imagenet, subset_of_imagenet) raise
with setup instructions instead of silently doing nothing.
"""

import argparse

from pcld.data.registry import DATASETS, ensure_dataset
from pcld.utils.consts import RESOURCES_DATASETS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=DATASETS.keys(),
                        help='registered dataset name')
    parser.add_argument('--splits', nargs='+', default=['test'],
                        help='splits to download (default: test)')
    parser.add_argument('--root', default=RESOURCES_DATASETS_DIR,
                        help='datasets root (default: RESOURCES_DATASETS_DIR)')
    args = parser.parse_args()

    for split in args.splits:
        path = ensure_dataset(args.dataset, split, root=args.root)
        print(f'{args.dataset}/{split} ready at {path}')


if __name__ == '__main__':
    main()