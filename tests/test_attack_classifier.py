"""Tests for the attack_classifier experiment plumbing.

The epsilon-schedule helper is torch-free and always runs; the navigator
dispatch test imports the experiment modules and self-skips without torch.
"""

import argparse

import pytest

from pcld.experiments.attack_classifier import build_epsilon_list


def test_build_epsilon_list_prepends_zero_once():
    assert build_epsilon_list([8]) == [0, 8]
    assert build_epsilon_list([3, 9]) == [0, 3, 9]
    # 0 already present: not duplicated, order untouched.
    assert build_epsilon_list([0, 8]) == [0, 8]
    assert build_epsilon_list([8, 0]) == [8, 0]


def test_build_epsilon_list_keeps_float_l2_budgets():
    assert build_epsilon_list([0.5]) == [0, 0.5]
    assert build_epsilon_list([0, 0.5, 1.5]) == [0, 0.5, 1.5]


def test_build_epsilon_list_does_not_mutate_input():
    eps = [8]
    build_epsilon_list(eps)
    assert eps == [8]


def test_navigator_dispatches_attack_classifier(monkeypatch):
    pytest.importorskip('torch')
    from pcld.experiments import experiment_navigator as nav

    called = {}

    def fake_main(args, device):
        called['args'] = args
        called['device'] = device

    monkeypatch.setattr(nav, 'main_attack_classifier', fake_main)
    args = argparse.Namespace(experiment_type='attack_classifier')
    nav.apply_experiment(args=args, device='cpu')

    assert called['args'] is args
    assert called['device'] == 'cpu'
