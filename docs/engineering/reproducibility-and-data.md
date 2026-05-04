# Reproducibility and data

## Colab: `ModuleNotFoundError: No module named 'gradient_ascent'`

Google Colab’s working directory is often **`/content`**, not the cloned repository, so Python cannot see `src/gradient_ascent` until the repo is on `sys.path`.

The **first code cell** in `notebooks/experiments.ipynb` fixes this in stdlib only: walk parents of `cwd` for `pyproject.toml` + `src/`, optionally `git clone` to `/content/gradient-ascent` (with `GITHUB_TOKEN` for a private repo), `os.chdir` into the project, then prepend the project root and `src/` to `sys.path`. Run that cell before any other code cell.

## CIFAR-10 download fallbacks

**Problem.** `torchvision.datasets.CIFAR10` uses a single Toronto URL; the host sometimes returns **503** or is unavailable, breaking fresh Colab or CI with `download=True`.

**Constraint.** The tarball must match the official **MD5** (`c58f30108f718f92721af3b95e74349a`) — same as torchvision’s `cifar.py`.

**Solution.** `load_cifar10_datasets` in `src/gradient_ascent/data.py` tries URLs in order (Toronto mirror, Brainchip mirror, Azure ML public blob, etc.) with exponential backoff on transient failures.

**Custom mirrors.** Set `GRADIENT_ASCENT_CIFAR10_URLS` to a comma-separated ordered list to try your own host first.

**References:**

- Krizhevsky, CIFAR-10 official page: https://www.cs.toronto.edu/~kriz/cifar.html  
- PyTorch Vision `CIFAR10`: https://github.com/pytorch/vision/blob/main/torchvision/datasets/cifar.py  
- Azure ML examples pipeline for the same archive.

---

## Colab: reusing unlearned checkpoints

**Goal.** Skip rerunning expensive unlearning if artefacts already exist on Drive (same idea as reusing `original_net.pt` / `retrained_from_scratch_net.pt`).

**How.** `run_core_checkpoints(..., reuse_unlearned_checkpoints=True)` (surfaced in `notebook_helpers` / notebook config). For each algorithm, if **all** of the following exist, that run is skipped:

- `unlearned_net_<algorithm>.pt`
- `unlearning_snapshots_<algorithm>/epoch_*.pt`
- classwise CSV/PNGs for that algorithm

Otherwise only the **missing** algorithm is recomputed.

**Notebook.** Set `REUSE_UNLEARNED_CHECKPOINTS = True` and point `OUT_DIR` at your persistent output folder.

**References:** Pineau et al. (2021) on reproducibility; Sculley et al. (2015) on ML systems discipline; Géron (2022) on checkpointing practice.
