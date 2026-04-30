# Similarity Before/After Grouped Plot (Apr 2026)

## What was added

I added a new static similarity figure that compares only the first and last unlearning snapshots for each `(algorithm, reference)` pair:

- file pattern: `similarity_vs_unlearning_epoch_<algo>_vs_<reference>_before_after_grouped_bars.png`
- x-axis: 5 similarity metric groups (`cka_linear`, `cca`, `cosine`, `euclidean`, `kl_sym`)
- inside each metric group: two subgroups (`Before`, `After`)
- inside each subgroup: one bar per layer (5 layers)

This gives `5 metrics * 2 snapshots * 5 layers = 50 bars` in one plot, matching the dissertation requirement for a clear before/after comparison.

## Why this change was made

The existing evolving GIFs are useful for trajectory inspection, but they make it harder to quickly compare "start vs end" at a glance. The new plot is intended for:

- concise dissertation figures where space is limited;
- visual comparison of representational movement by metric and layer;
- easier cross-checking against MIA outcomes at the final unlearning step.

## How I knew this was a reasonable implementation

The implementation follows standard grouped-bar plotting practice in Matplotlib (metric groups, subgroup offsets, and per-series bar offsets) and keeps the same oriented similarity scale already used in this project:

- Matplotlib grouped bar chart pattern: [Grouped bar chart example](https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html)
- Existing project convention for orienting similarity metrics to a common "higher = more similar" interpretation is reused via `orient_epoch_rows_for_similarity(...)`.

## Integration points

- Plot function added in `src/gradient_ascent/trajectories.py`:
  - `save_similarity_before_after_grouped_bar_plot(...)`
- Called from `run_trajectory_analysis(...)` in `src/gradient_ascent/experiments.py`
- Stored in `SimilarityArtifact.before_after_grouped_plot_path`
- Displayed in notebook output flow in `src/gradient_ascent/notebook_helpers.py`
- Logged to Weights & Biases under key:
  - `plots/similarity_vs_unlearning_epoch_<algo>_vs_<reference>_before_after_grouped_bars`

## Notes for dissertation write-up

When discussing this figure, describe it as:

- a representation-similarity endpoint comparison (step 0 vs final step),
- stratified by both metric family and layer,
- intended to support analysis of whether similarity to retraining corresponds to MIA behavior.
