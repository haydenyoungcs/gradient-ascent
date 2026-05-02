# Source layout (`src/gradient_ascent`)

| Path | Responsibility |
|------|----------------|
| `constants.py` | Shared lists such as unlearning algorithm order |
| `data.py` | CIFAR-10 loading, forget/retain splits, MIA probe sampling |
| `models.py` | ResNet (CIFAR) factory (`Net`) |
| `training.py` | Train/eval loops |
| `unlearning/` | GA, SSD, SalUn, SCRUB, certified removal implementations |
| `experiments/` | **Orchestration:** `run_core_checkpoints`, `run_trajectory_analysis`, configs, combined similarity+MIA figure |
| `similarity.py` | Metrics (CKA, CCA, …) and activation collection |
| `mia.py` | Membership inference attackers and trajectory rows |
| `trajectories.py` | Snapshot iteration, similarity plots, timing |
| `reporting.py` | CSVs and matplotlib helpers |
| `notebook_runtime.py` | `NotebookRuntime`, `SimilaritySetup`, trajectory defaults, W&B config flattening |
| `notebook_helpers.py` | Notebook entrypoints: prepare runtime, core/trajectory display pipelines |
| `notebook_bootstrap.py` | Colab/local environment bootstrap |
| `pipelines/multitarget.py` | Full forget-class sweep + macro-averaged exports |

Imports remain stable: `from gradient_ascent.experiments import run_core_checkpoints` and `from gradient_ascent.notebook_helpers import prepare_notebook_runtime` work as before.
