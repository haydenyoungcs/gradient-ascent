# Refresh similarity + correlation without retraining

## Goal

Regenerate all similarity trajectory plots and correlation outputs after changing how activations are collected (for example, switching from full test set to forget-set activations), while reusing existing trained checkpoints.

## What was added

In `src/gradient_ascent/experiments_notebook.py`, a dedicated notebook helper was added:

- `experiments_refresh_similarity_and_correlation(...)`
- Config dataclass: `RefreshSimilarityCorrelationConfig`

The refresh helper does three things in one call:

1. Reuses existing checkpoints (original, retrained, and unlearned) by forcing all core reuse flags to `True`.
2. Forces trajectory recomputation (`reuse_trajectory_outputs=False`) so similarity/MIA CSVs and plots are rewritten from current code.
3. Reruns correlation analysis on the regenerated trajectory CSVs.

## Why this is the correct fix

The existing multitarget pipeline supports caching and output reuse for speed. This is useful normally, but it can preserve stale similarity outputs after code changes:

- Per-target trajectory outputs are reused when expected files already exist.
- Activation caches can be reused across reruns if cache paths are unchanged.

So, if similarity logic changes but outputs/caches are reused, notebook reruns can still show old results. The refresh helper addresses this by explicitly invalidating those fast paths.

## Optional invalidation controls

`RefreshSimilarityCorrelationConfig` includes:

- `clear_similarity_activation_cache=True` (default): deletes `target_*/similarity_activation_cache/`.
- `clear_existing_similarity_outputs=True` (default): deletes old similarity/MIA trajectory artefacts and `multitarget_aggregate/correlation/` before rerun.
- `similarity_data_mode="forget"` (default): keeps evaluation on forget-class inputs unless you intentionally switch to `retain` or `test`.

## Notebook usage

```python
from gradient_ascent.experiments_notebook import (
    DEFAULT_REFRESH_SIMILARITY_CORRELATION_CONFIG,
    experiments_refresh_similarity_and_correlation,
)

multi_target_artifacts, correlation_paths = experiments_refresh_similarity_and_correlation(
    runtime,
    out_dir=OUT_DIR,
    wandb_module=wandb,
    config=DEFAULT_REFRESH_SIMILARITY_CORRELATION_CONFIG,
)
```

## Notes and assumptions

- This path still runs the multitarget orchestration; it just reuses existing checkpoints and recomputes trajectory/correlation artefacts.
- If a required checkpoint is missing for a target/algorithm, the underlying pipeline may train that missing piece. To avoid this, keep checkpoint directories complete.

## References

1. Bouthillier, X., Laurent, C., & Vincent, P. (2021). Accounting for Variance in Machine Learning Benchmarks. *Proceedings of Machine Learning and Systems*, 3, 747-769.
2. Dietterich, T. G. (1998). Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. *Neural Computation*, 10(7), 1895-1923.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. (See discussion of train/eval pipeline consistency and cached/intermediate computations).
