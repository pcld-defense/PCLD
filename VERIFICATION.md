# Verifying the refactor is behaviour-preserving

The refactor must produce the **same numerical attack results** through the new
config-driven interface as the pre-refactor code did. This document gives the
exact commands to prove it on the cluster.

## Why the smoke config is the equivalence config

`configs/experiment/smoke_test.yaml` uses **FGSM, untargeted**. That path has
**no randomness in the attack itself** (a single signed-gradient step), so its
per-image output is deterministic given the model and input — independent of
seed or DataLoader order. `scripts/verify_equivalence.py` sorts both result
files by `(image, t, epsilon, phase, attacked_model)` before comparing, so
shuffle order is irrelevant too. This makes the comparison exact, not
approximate.

(Adaptive PGD uses random restarts and random init; those are seeded via
`seed_everything`, but FGSM is chosen for the proof to remove that variable
entirely.)

## Procedure

Run from the repo root, with the `.env` pointing at the cluster paths and the
`.venv` activated (`make setup-cuda` if not yet created).

### 1. Baseline — pre-refactor code

Check out the pre-refactor commit in a throwaway worktree and run the **legacy
CLI** to produce the baseline result file:

```bash
git worktree add /tmp/pcld_baseline refactor_rc     # pre-refactor commit
cd /tmp/pcld_baseline
python main.py \
  --experiment_type attack_pcld \
  --experiment_name equiv_baseline \
  --dataset subset_of_imagenet --dataset_type imagenet --splits test \
  --batch_size 4 \
  --classifier_experiment train_clf_bp \
  --decisioner_experiment train_decisioner_conv_fgsm \
  --attack fgsm --attack_direction untargeted --epsilons '0|3' \
  --save_parquet 1
# -> $RESOURCES_RESULTS_DIR/equiv_baseline/test_eps3_linf_results.parquet
cd -
```

### 2. New pipeline — refactored code

From this branch, run the **same experiment via the config**:

```bash
python scripts/run.py experiment=smoke_test \
  experiment_name=equiv_new \
  model.classifier_experiment=train_clf_bp \
  model.decisioner_experiment=train_decisioner_conv_fgsm
# -> $RESOURCES_RESULTS_DIR/equiv_new/test_eps3_linf_results.parquet
```

### 3. Diff

```bash
python scripts/verify_equivalence.py \
  "$RESOURCES_RESULTS_DIR/equiv_baseline/test_eps3_linf_results.parquet" \
  "$RESOURCES_RESULTS_DIR/equiv_new/test_eps3_linf_results.parquet"
```

Expected output:

```
OK: all N predictions identical
Max abs softmax diff: <=1e-5 ...
EQUIVALENT: refactored pipeline reproduces baseline output.
```

Record the printed prediction count and max-softmax-diff in the PR description.

## Smoke test (does the new pipeline run end to end?)

```bash
make smoke        # == python scripts/run.py experiment=smoke_test
```

## Unit tests (run anywhere, no GPU / no torch needed)

```bash
pip install pytest
PYTHONPATH=src pytest tests/ -q
```
