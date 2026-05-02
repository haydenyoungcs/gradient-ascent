# Codebase maintenance notes

## Core pipeline pruning (April 2026)

**Scope.** Notebook core path only: runtime setup, original training, retrain-without-forget-class, five unlearning baselines **before** similarity analysis.

**Method.** Static call-path audit from `prepare_notebook_runtime`, `run_notebook_core_experiment` → `run_core_checkpoints` and `run_*_unlearning`, plus repo-wide symbol search.

**Removed (examples).**

- Unused snapshot wrappers: `run_ga_snapshots`, `run_ssd_snapshots`, etc. (notebook calls `run_*_unlearning` directly).
- Dead shim `src/gradient_ascent/certified.py` (re-export only, no imports).
- Tightened `unlearning/__init__.py` exports to actively used APIs.

**Safety.** No algorithm logic or hyperparameters changed; only unreferenced helpers removed.

## Similarity / MIA stage pruning

Second audit from `prepare_similarity_setup`, `run_notebook_trajectory_experiment`, `save_notebook_combined_comparison` → `run_trajectory_analysis`, `compute_epoch_rows_from_snapshots`, MIA builders, reporting saves.

**Removed (examples).** Unused distance/plot helpers in `similarity.py` and `reporting.py` (e.g. EMD/GW helpers not used by `build_default_metrics()`), plus heavy imports only they needed.

**Retained.** CKA (linear), CCA, cosine, Euclidean, symmetric KL; full MIA path on active execution.

## Notebook extraction

Orchestration moved into:

- `notebook_bootstrap.py` — Colab/local bootstrap, editable install  
- `notebook_helpers.py` — `run_and_display_*` pipelines  

**Rationale.** Less copy-paste in `hayden_notebook.ipynb`, clearer reproducibility, same execution order and artefacts (refactor only).

## References

- Fowler (2018), *Refactoring*.  
- Sadowski et al. (2015), Tricorder / program analysis ecosystem.  
- Method papers for the five baselines (Foster SSD; Kurmanji SCRUB; Fan SalUn; Guo certified) — for dissertation context, not required for this maintenance note.
