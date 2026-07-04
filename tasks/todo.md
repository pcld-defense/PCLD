# Refactor: config-driven experiment framework (branch: refactor/experiment-framework)

## Phase 1 — Explore & plan
- [x] Create branch `refactor/experiment-framework`
- [x] Map repo: entry points, attack code, data/model loading, hyperparams, metrics IO
- [x] Report architecture, pain points, target structure, migration plan, risks
- [ ] **WAIT FOR APPROVAL** (+ answers to open questions below)

## Phase 2 — Reorganize (after approval)
- [ ] Commit 1 — hygiene: untrack .pyc, .gitignore, pyproject/pinned deps, venv setup script, add missing `gdown` dep
- [ ] Commit 2 — package skeleton `src/pcld/`, move modules verbatim (imports only, no logic changes)
- [ ] Commit 3 — config layer (YAML + CLI overrides) mirroring current argparse defaults exactly; per-run output folder with config snapshot
- [ ] Commit 4 — registries: datasets (incl. download-on-first-use), attacks (Attack interface: fgsm/pgd/aa), models (extend existing CLASSIFIER_REGISTRY; RobustBench comparison list)
- [ ] Commit 5 — seeding utility + structured metrics (metrics.json, robust-accuracy summary, comparison table → CSV/LaTeX)
- [ ] Commit 6 — example configs (paper main, smoke test, RB sweep) + README rewrite
- [ ] Fix-behind-flag (separate commit, default=legacy): attack_pcld hardcoded resnet18 classifier + IMAGENET class list

## Verification
- [ ] Fresh `.venv` + pinned install, smoke config end-to-end
- [ ] Baseline: run CURRENT code (pre-refactor) on deterministic mini-config (fixed seed, shuffle off, small N), save parquet
- [ ] Re-run through new pipeline, diff numbers, state them
- [ ] Commit log clean and logical

## Open questions for user (blocking Phase 2 details, not structure)
1. Verification machine: local Windows (RTX 3050 4GB, weights present in resources/models, .env must be repointed) or Linux cluster?
2. Config system: Hydra (recommended) vs plain OmegaConf+argparse?
3. Torch pin: keep 1.13.1 or pin what the cluster venv actually runs?
4. Which painter config is "the paper" (PainterConsts now MAX_STEP=40/DIVIDE=1 CIFAR-style vs docs' 80/5 ImageNet)?