# Refactor: config-driven experiment framework (branch: refactor/experiment-framework)

## Phase 1 — Explore & plan
- [x] Create branch `refactor/experiment-framework`
- [x] Map repo: entry points, attack code, data/model loading, hyperparams, metrics IO
- [x] Report architecture, pain points, target structure, migration plan, risks
- [ ] **WAIT FOR APPROVAL** (+ answers to open questions below)

## Phase 2 — Reorganize (after approval) — DONE (decisions: Hydra, cluster verify, torch 2.x)
- [x] Commit 1 (3918aee) — hygiene: untrack .pyc, expand .gitignore
- [x] Commit 2 (4935289) — src/pcld package, modules moved verbatim, imports rewritten, pyproject/Makefile/setup_env.ps1, torch 2.5.1 pin, +gdown/pyarrow/hydra, -unused deps
- [x] Commit 3 (9805d93) — Hydra config layer mirroring argparse defaults 1:1; seed_everything; per-run config snapshot; tests/test_config.py
- [x] Commit 4 (905b14c) — registries: generic Registry, ATTACKS (fgsm/pgd/aa) with attack_batch dispatch, DATASETS with download-on-first-use, download CLI
- [x] Commit 5 (4733ba4) — eval/metrics (robust_accuracy, metrics.json, comparison table CSV/LaTeX), scripts/sweep.py, tests/test_metrics.py
- [x] Commit 6 (2443f44) — README rewrite + VERIFICATION.md + verify_equivalence.py
- [ ] DEFERRED (behaviour-affecting, not done to keep refactor behaviour-identical):
      attack_pcld.py still hardcodes torchvision resnet18 + `model.pth` + ImageNet
      class list, ignoring model_type/dataset_type. Preserved on purpose. Fix
      behind a flag in a follow-up once equivalence is signed off.

## Verification — RUNS ON CLUSTER (user), see VERIFICATION.md
- [x] Config + metrics unit tests pass locally (8/8, torch-free)
- [x] All pcld.* import targets resolve; all sources byte-compile
- [ ] Cluster: `make setup-cuda` + `make smoke` end-to-end
- [ ] Cluster: legacy (refactor_rc worktree) vs new smoke_test → verify_equivalence.py; record numbers in PR
- [x] Commit log clean and logical (6 commits)

## Review
- Structure delivered: configs/ + src/pcld/{utils,data,models,attacks,painter,eval,experiments} + scripts/ + tests/.
- Behaviour-preserving by construction: config flattens to the SAME Namespace; attack dispatch keeps identical per-attack code paths; ensure_dataset is a no-op when data exists. Numeric equivalence proof is scripted but must be run on the cluster (GPU + weights + data live there; local box has no torch).
- Local box: Python 3.10, RTX 3050 4GB, no torch installed; weights present under resources/models but .env points at cluster paths.

## Open questions for user (blocking Phase 2 details, not structure)
1. Verification machine: local Windows (RTX 3050 4GB, weights present in resources/models, .env must be repointed) or Linux cluster?
2. Config system: Hydra (recommended) vs plain OmegaConf+argparse?
3. Torch pin: keep 1.13.1 or pin what the cluster venv actually runs?
4. Which painter config is "the paper" (PainterConsts now MAX_STEP=40/DIVIDE=1 CIFAR-style vs docs' 80/5 ImageNet)?