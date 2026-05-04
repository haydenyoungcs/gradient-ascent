# Project overview

## Purpose

Reproducible **PyTorch** pipeline to train a **ResNet-50** on **CIFAR-10**, **unlearn** a chosen class (or run all ten forget classes), compare against **retrain-from-scratch** references, and evaluate with **classwise accuracy**, **representation similarity** (CKA, CCA, cosine, Euclidean, symmetric KL), and **membership inference attacks (MIA)** along unlearning trajectories.

## Main entry point

- **Notebook:** `notebooks/experiments.ipynb` (thin cells; drivers in `src/gradient_ascent/experiments_notebook.py`)  
  Orchestrates training, unlearning, similarity/MIA trajectories, and combined plots via helpers in `src/gradient_ascent/`.

## Core package (`src/gradient_ascent/`)

| Module | Role |
|--------|------|
| `notebook_helpers.py` | Notebook API: runtime prep, core + trajectory display helpers |
| `notebook_bootstrap.py` | Environment setup (e.g. Colab/local, editable install) |
| `notebook_runtime.py` | `NotebookRuntime`, `SimilaritySetup`, trajectory defaults, W&B config for core runs |
| `experiments/` (package) | `run_core_checkpoints`, `run_trajectory_analysis`, configs, combined similarity+MIA figure |
| `pipelines/multitarget.py` | Forget-label sweep and macro-averaged CSV/plot aggregation |
| `trajectories.py` | Snapshot iteration, similarity plots, activation caching |
| `similarity.py` | Activation extraction and metric definitions |
| `mia.py` | Attack models and trajectory MIA metrics |
| `reporting.py` | CSV/plot helpers, metric orientation for figures |
| `data.py` | CIFAR-10 loading (with download fallbacks), probe subsets |
| `training.py` | Training loops for original and retrained models |
| `models.py` | ResNet-50 (CIFAR-adapted) |
| `unlearning/` | GA, SSD, SalUn, SCRUB, certified last-layer removal |

## Typical experiment flow

1. **Train** original model on full CIFAR-10; **train** retrained model on retain set (single forget class) or repeat for each class.
2. **Unlearn** from the original checkpoint with GA, SSD, SalUn, SCRUB, or certified method; save **snapshots** per epoch/step.
3. **Trajectories:** for each snapshot, compute similarity vs **original** and **retrained** references; run **MIA** with fixed protocol.
4. **Optional:** `run_multitarget_averaged_experiment` in `pipelines/multitarget.py` (re-exported from `notebook_helpers`) aggregates metrics over forget labels `0..9` into `multitarget_aggregate/`.

## Where to read next

- Macro-averaged dissertation results: [evaluation/multitarget-averaging.md](evaluation/multitarget-averaging.md)
- MIA design and caveats: [evaluation/membership-inference.md](evaluation/membership-inference.md)
- SSD/SCRUB methodology: [methods/ssd-and-scrub.md](methods/ssd-and-scrub.md)
- Similarity plots and runtime: [similarity/visualisation-and-performance.md](similarity/visualisation-and-performance.md)
