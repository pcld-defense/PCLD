"""Tests for the RobustBench fixed-prefix data path.

Covers the RBPrefixDataset item contract (attacker() unpacks
``x, y, img_paths`` from each batch), the SHA-256 fingerprint stability, and
the ``num_samples`` capping logic in ``build_eval_loaders``.

CPU-only and network-free: robustbench itself is never imported (the dataset
is built from synthetic tensors). Skipped automatically if torch is not
installed.
"""

import types

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')


def _synthetic_tensors(n=8):
    torch.manual_seed(0)
    x = torch.rand(n, 3, 32, 32)
    y = torch.randint(0, 10, (n,))
    return x, y


def test_rb_prefix_dataset_item_contract():
    from pcld.data.robustbench_data import RBPrefixDataset

    x, y = _synthetic_tensors(8)
    classes = [str(i) for i in range(10)]
    ds = RBPrefixDataset(x, y, classes)

    assert len(ds) == 8
    xi, yi, path = ds[3]
    assert torch.equal(xi, x[3])
    assert isinstance(yi, int) and yi == int(y[3])
    assert path == 'rb_00003.png'

    # Ordering preserved across the whole dataset.
    for i in range(len(ds)):
        xi, yi, path = ds[i]
        assert torch.equal(xi, x[i])
        assert yi == int(y[i])
        assert path == f'rb_{i:05d}.png'

    assert ds.classes == classes


def test_rb_prefix_loader_is_deterministic_over_dataset():
    from pcld.data.robustbench_data import RBPrefixDataset

    x, y = _synthetic_tensors(8)
    ds = RBPrefixDataset(x, y, [str(i) for i in range(10)])
    loader = torch.utils.data.DataLoader(ds, batch_size=3, shuffle=False,
                                         num_workers=0)

    xs, ys, paths = [], [], []
    for bx, by, bp in loader:
        xs.append(bx)
        ys.append(by)
        paths.extend(bp)

    assert torch.equal(torch.cat(xs), x)
    assert torch.equal(torch.cat(ys), y)
    assert paths == [f'rb_{i:05d}.png' for i in range(8)]


def test_prefix_fingerprint_stable():
    from pcld.data.robustbench_data import prefix_fingerprint

    x, y = _synthetic_tensors(8)

    fp1 = prefix_fingerprint(x, y)
    fp2 = prefix_fingerprint(x.clone(), y.clone())
    assert fp1 == fp2
    assert fp1['n'] == 8
    assert fp1['shape'] == [8, 3, 32, 32]

    # Permuting the rows must change the image hash.
    perm = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6])
    fp_perm = prefix_fingerprint(x[perm], y)
    assert fp_perm['x_sha256'] != fp1['x_sha256']


def test_write_fingerprint_roundtrip(tmp_path):
    import json

    from pcld.data.robustbench_data import (prefix_fingerprint,
                                            write_fingerprint)

    x, y = _synthetic_tensors(4)
    fp = prefix_fingerprint(x, y)
    path = write_fingerprint(fp, str(tmp_path))

    assert path.endswith('rb_prefix_fingerprint.json')
    with open(path) as f:
        assert json.load(f) == fp


class _FakeFolderDS(torch.utils.data.Dataset):
    """Minimal stand-in for ImageFolderWithPaths (item = (x, y, path))."""

    def __init__(self, n=10):
        torch.manual_seed(1)
        self.x = torch.rand(n, 3, 32, 32)
        self.y = torch.randint(0, 10, (n,))
        self.classes = [str(i) for i in range(10)]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], int(self.y[i]), f'img_{i}.png'


def test_build_eval_loaders_folder_cap(monkeypatch):
    import pcld.data.datasets as datasets_mod

    fake_ds = _FakeFolderDS(10)
    fake_loader = torch.utils.data.DataLoader(fake_ds, batch_size=2,
                                              shuffle=False, num_workers=0)

    def _fake_get_loaders(dataset, splits, transform_dict, batch_size):
        return {split: [fake_ds, fake_loader] for split in splits}

    monkeypatch.setattr(datasets_mod, 'get_loaders', _fake_get_loaders)

    args = types.SimpleNamespace(dataset='cifar10', dataset_type='cifar10',
                                 splits=['test'], data_source='folder',
                                 num_samples=4)
    loaders = datasets_mod.build_eval_loaders(args, batch_size=2)

    ds, loader = loaders['test']
    assert isinstance(ds, torch.utils.data.Subset)
    assert len(ds) == 4
    assert ds.classes == fake_ds.classes
    assert ds.class_to_idx == fake_ds.class_to_idx

    # The capped loader yields exactly the first 4 samples, in order.
    seen = [item for _, _, batch_paths in loader for item in batch_paths]
    assert seen == ['img_0.png', 'img_1.png', 'img_2.png', 'img_3.png']


def test_build_eval_loaders_no_cap_passthrough(monkeypatch):
    import pcld.data.datasets as datasets_mod

    fake_ds = _FakeFolderDS(10)
    fake_loader = torch.utils.data.DataLoader(fake_ds, batch_size=2,
                                              shuffle=False, num_workers=0)
    sentinel = {'test': [fake_ds, fake_loader]}

    monkeypatch.setattr(datasets_mod, 'get_loaders',
                        lambda *a, **k: sentinel)

    args = types.SimpleNamespace(dataset='cifar10', dataset_type='cifar10',
                                 splits=['test'], data_source='folder',
                                 num_samples=None)
    loaders = datasets_mod.build_eval_loaders(args, batch_size=2)

    # num_samples=None returns the get_loaders result untouched.
    assert loaders is sentinel
    assert loaders['test'][0] is fake_ds
    assert loaders['test'][1] is fake_loader
