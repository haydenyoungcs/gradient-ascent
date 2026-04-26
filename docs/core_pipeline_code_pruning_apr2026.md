# Core Pipeline Code Pruning (April 2026)

## Scope and motivation

This cleanup is intentionally restricted to the current notebook core pipeline: runtime setup, original training, retraining-without-forget-class, and the five unlearning baselines (GA, SSD, SalUn, Certified Removal, SCRUB) executed before any similarity analysis.

The goal is to reduce maintenance and dissertation risk by removing code paths that are not used by that pipeline, while keeping algorithm behavior unchanged.

## How I determined what was unused

I used a simple static call-path audit:

1. Start from notebook entry points:
   - `prepare_notebook_runtime(...)`
   - `run_notebook_core_experiment(...)`
2. Follow these into:
   - `run_core_checkpoints(...)`
   - `run_ga_unlearning(...)`, `run_ssd_unlearning(...)`, `run_salun_unlearning(...)`, `run_certified_unlearning(...)`, `run_scrub_unlearning(...)`
3. Run repository-wide symbol search to check whether candidate functions are called from anywhere else.

This is standard dead-code identification: code with no inbound references from executable entry points can be removed without changing runtime behavior.

## What was removed

### 1) Unused snapshot wrapper helpers

Removed the following functions (not referenced by the notebook core pipeline, tests, or package internals):

- `run_ga_snapshots(...)`
- `run_ssd_snapshots(...)`
- `run_salun_snapshots(...)`
- `run_certified_snapshots(...)`
- `run_scrub_snapshots(...)`

These were thin wrappers that reconstructed a model from checkpoint and then immediately called the corresponding `run_*_unlearning(...)` function. The notebook core pipeline already calls `run_*_unlearning(...)` directly, so these wrappers were redundant.

### 2) Obsolete compatibility shim

Removed `src/gradient_ascent/certified.py`.

Reason: it only re-exported symbols from `gradient_ascent.unlearning.certified` and was not imported anywhere in this repository. Keeping a dead shim increases namespace surface without providing value for the current workflow.

### 3) Package export tightening

Updated `src/gradient_ascent/unlearning/__init__.py` to export only actively used public APIs for this workflow (configs, `run_*_unlearning(...)`, and analysis helpers still used by implementations).

## Why this is methodologically safe

- No algorithm implementation logic was changed.
- No hyperparameters or training loops were changed.
- The same core notebook pipeline still reaches the same training/unlearning paths.
- Only unreferenced helper entry points and an unreferenced compatibility module were removed.

So the scientific claims (forgetting and utility behavior of the five methods) are unaffected by this pruning.

## Second pass: similarity and MIA stage (before similarity tests)

After the first cleanup, I ran a second call-path audit specifically from notebook cells 5-9:

- `prepare_similarity_setup(...)`
- `run_notebook_trajectory_experiment(...)`
- `save_notebook_combined_comparison(...)`

and followed these into:

- `run_trajectory_analysis(...)`
- `compute_epoch_rows_from_snapshots(...)`
- `compute_mia_baseline(...)`, `compute_mia_trajectory_rows(...)`
- `save_*` plotting/CSV utilities used by trajectory and combined comparison outputs.

### Removed in second pass

The following similarity-stage utilities had no inbound references from this pipeline or elsewhere in the repository:

1. `EarthMoversDistance` in `src/gradient_ascent/similarity.py`
2. `GromovWassersteinDistance` in `src/gradient_ascent/similarity.py`
3. `plot_grouped_bars(...)` in `src/gradient_ascent/similarity.py`
4. `build_mean_metric_series(...)` in `src/gradient_ascent/reporting.py`

Because these were removed, now-unused heavy imports were also removed from `similarity.py` (`ot`, `scipy.stats`, `scipy.spatial.distance.cdist`).

### Why this is safe for your current experiments

- Your current metric bundle from `build_default_metrics()` uses:
  - CKA (linear),
  - CCA,
  - cosine similarity,
  - euclidean distance,
  - symmetric KL divergence.
- None of the removed classes/functions were called by this active metric bundle, the trajectory runner, MIA code, or combined plot generation.
- MIA code (`collect_attack_stats`, threshold/logreg probes, baseline and trajectory builders) was retained fully because all of it is on the active execution path.

This keeps the exact behavior of your current similarity/MIA experiments unchanged while reducing code surface and dependency burden.

## Notebook condensation pass

To keep `notebooks/hayden_notebook.ipynb` concise while preserving all outputs, notebook orchestration logic was moved into package helpers under `src/gradient_ascent/`:

- `notebook_bootstrap.py`
  - `bootstrap_notebook_environment(...)` for Colab/local setup, editable install, and wandb login.
- `notebook_helpers.py`
  - `run_and_display_notebook_core_pipeline(...)`
  - `run_and_display_notebook_trajectory_pipeline(...)`
  - `run_and_display_notebook_combined_comparison(...)`

The notebook now mostly delegates to these helpers, which improves readability and reproducibility:

- less in-notebook boilerplate,
- easier unit-level inspection of non-visual logic,
- lower risk of copy-paste drift between notebook cells.

Methodologically, this is a pure extraction refactor: execution order, configurations, and produced artifacts remain unchanged.

## References

1. Fowler, M. *Refactoring: Improving the Design of Existing Code* (2nd ed.). Addison-Wesley, 2018.  
   (Dead code removal is a standard refactoring for lowering maintenance risk.)
2. Sadowski, C., et al. “Tricorder: Building a Program Analysis Ecosystem.” *ICSE SEIP*, 2015.  
   [https://research.google/pubs/pub43322/](https://research.google/pubs/pub43322/)  
   (Supports static analysis/search-based approaches to code health and unused-path detection.)
3. Foster, J., Schoepf, S., Brintrup, A. *Fast Machine Unlearning Without Retraining Through Selective Synaptic Dampening.*  
   [https://arxiv.org/abs/2308.07707](https://arxiv.org/abs/2308.07707)
4. Kurmanji, M., et al. *Towards Unbounded Machine Unlearning.*  
   [https://arxiv.org/abs/2302.09880](https://arxiv.org/abs/2302.09880)
5. Fan, C., et al. *SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency in Both Image Classification and Generation.*  
   [https://arxiv.org/abs/2310.12508](https://arxiv.org/abs/2310.12508)
6. Guo, C., Goldstein, T., Hannun, A., van der Maaten, L. *Certified Data Removal from Machine Learning Models.*  
   [https://arxiv.org/abs/1911.03030](https://arxiv.org/abs/1911.03030)
