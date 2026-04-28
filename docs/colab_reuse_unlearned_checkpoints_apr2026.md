# Colab Reuse of Unlearned Checkpoints (April 2026)

## Goal

Enable the notebook to reuse already-saved unlearning outputs from Drive (GA, SSD, SalUn, Certified, SCRUB), in the same spirit as reusing `original_net.pt` and `retrained_from_scratch_net.pt`.

## What was changed

1. Added a new option to the core pipeline:
   - `reuse_unlearned_checkpoints` in `run_core_checkpoints(...)`
   - surfaced through notebook helpers and notebook cell config.
2. For each unlearning algorithm, the pipeline now checks whether these cached artifacts exist:
   - `unlearned_net_<algorithm>.pt`
   - `unlearning_snapshots_<algorithm>/epoch_*.pt`
   - classwise outputs (`classwise_accuracy_<algorithm>.csv`, `classwise_percent_change_<algorithm>.png`, `classwise_absolute_accuracy_<algorithm>.png`)
3. If all are present and reuse is enabled:
   - load the saved final model checkpoint,
   - reuse the existing snapshot/plot artifact paths,
   - skip rerunning that unlearning algorithm.
4. If anything is missing:
   - rerun only that algorithm and regenerate its artifacts.

This gives a robust partial-cache behavior (reuse what exists, recompute only missing parts).

## Notebook usage

In `notebooks/hayden_notebook.ipynb`, set:

- `REUSE_UNLEARNED_CHECKPOINTS = True`

and keep `OUT_DIR` pointing to your Drive output folder (for Colab runs typically under `.../MyDrive/gradient-ascent-out`).

## Why this approach is sound

- It is a standard reproducible-ML pattern to checkpoint expensive stages and reload artifacts to avoid unnecessary recomputation.
- The logic still validates required files before reuse, then falls back to recomputation if cache is incomplete.
- Scientific outputs are unchanged relative to a full rerun when the cached artifacts correspond to the same configuration.

## References

1. Géron, A. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.), O’Reilly, 2022.  
   (Practical guidance on checkpointing and resumable training workflows.)
2. Pineau, J., et al. “Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program).” *JMLR* 22(164), 2021.  
   [https://www.jmlr.org/papers/v22/20-303.html](https://www.jmlr.org/papers/v22/20-303.html)
3. Sculley, D., et al. “Hidden Technical Debt in Machine Learning Systems.” *NeurIPS*, 2015.  
   [https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems)  
   (Supports disciplined artifact management and robust pipeline behavior.)
