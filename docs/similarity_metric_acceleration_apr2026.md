# Similarity metric acceleration (April 2026)

This note records why acceleration changes were made to the trajectory similarity stage and how to override them for stricter (slower) experiments.

## 1. Linear CKA without n×n Gram matrices

**Problem.** The previous implementation formed `x @ x.T` and `y @ y.T` (each n×n), doubly-centred them, then combined them with element-wise products. For n in the low thousands (several dataloader batches), that cost scales as O(n²) in memory and time.

**Change.** Linear CKA with the same doubly-centred Gram definition is computed via column-centred activations and Frobenius norms of d×d statistics (`xc.T @ yc`, `xc.T @ xc`, `yc.T @ yc`), following the algebra used in the linear CKA literature.

**Reference.** Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). *Similarity of Neural Network Representations Revisited*. arXiv:1905.00414. The implementation matches the legacy numeric output (see `tests/test_similarity_acceleration.py`).

## 2. Subsampled activation rows

**Problem.** Even after CKA optimisation, CCA and other metrics scale with the number of rows n.

**Change.** `TrajectoryExperimentConfig` now includes `max_activation_samples` (default `1024`) and `activation_subsample_seed` (default `42`). `collect_model_activations` trims rows after concatenation, using the same index set for every tracked layer so representations stay aligned. `evaluate_pair_rows` applies the same cap when given pre-collected activations without trimming.

**Trade-off.** Scores are Monte Carlo estimates over a fixed random subset of inputs rather than the full collected batch stack. For thesis text, state the cap and seed; set `max_activation_samples=None` on `TrajectoryExperimentConfig` (or pass `None` through `collect_model_activations` / `evaluate_pair_rows`) to use every collected row at higher cost.

## 3. CCA cost reductions

**SVD without U and V.** The CCA path only needs singular values of `qx.T @ qy`. Calling `numpy.linalg.svd(..., compute_uv=False)` skips the expensive construction of the full orthogonal factors while returning the same singular values (up to floating-point noise).

**Optional column subsampling.** QR on an `n × d` activation matrix is costly when `d` is large (e.g. ResNet block channels). Before centering, we can randomly subsample columns (neurons) to a cap (`cca_max_columns`, default `512` in `TrajectoryExperimentConfig` / `prepare_similarity_setup`). This is a Monte Carlo estimate over features, analogous in spirit to dimensionality reduction used in SVCCA (Raghu et al., 2017, *SVCCA: Singular Vector Canonical Correlation Analysis for deep learning dynamics and interpretability*). Set `cca_max_columns=None` for the previous full-width CCA (slowest, closest to the exact wide-matrix pipeline).

## 4. Notebook / cell progress lines

During the trajectory similarity stage, `TrajectoryExperimentConfig.log_similarity_progress` (default `True`) prints two lines per unlearning snapshot epoch: when activations are collected and when similarity scoring starts (`compute_epoch_rows_from_snapshots` in `trajectories.py`). Per-layer / per-metric lines are not printed. Set `log_similarity_progress=False` to disable these epoch lines as well.

## 5. Reuse snapshot activations across both references

**Problem.** In the trajectory loop, each unlearning snapshot was evaluated against two references (`retrained` and `original`) in separate passes. That meant loading each checkpoint and running a forward pass twice, even though the snapshot activations themselves are identical regardless of which reference they are compared against.

**Change.** Added `compute_epoch_rows_from_snapshots_multi_reference` in `trajectories.py`, and updated `run_trajectory_analysis` to call it once per algorithm. The function now:

1. Loads each snapshot once,
2. Collects activations once for that snapshot,
3. Reuses those activations to score against both references.

This reduces duplicate forward-pass work in the similarity stage while keeping exactly the same metrics and output files.

**Why this is valid.** All similarity metrics in this notebook compare a fixed pair of activation matrices `X` and `Y` at each epoch. Reusing `X` (snapshot activations) across multiple `Y` references does not change the computation; it only avoids recomputing `X`.

## 6. Remove double row-subsampling during similarity

**Problem.** We were subsampling rows in two places:

- `collect_model_activations(..., max_activation_samples=...)`, and
- `evaluate_pair_rows(..., max_activation_samples=...)`.

Applying both means a second random slice over data that was already capped once.

**Change.** The notebook trajectory wiring now performs row capping only in activation collection, and disables the second cap in pair evaluation (`max_activation_samples=None` there).

**Why this is valid.** A single fixed subsample already gives the intended runtime/variance trade-off. A second subsample does not provide extra statistical benefit for this setup, and only adds extra indexing/copy cost.

## 7. Pool activations on GPU before CPU transfer

**Problem.** For convolutional outputs shaped like `N x C x H x W`, transferring activations to CPU before spatial pooling moves much more data than required. In this project we compare pooled channel representations (roughly `N x C`), so copying full feature maps is unnecessary overhead.

**Change.** In `collect_model_activations`, spatial averaging now happens on-device (`output.mean(...)`) before `.cpu()`. This preserves the exact pooled representation used by the metrics, while reducing host-device transfer volume.

**Why this is valid.** Mean pooling is a linear reduction. Computing it before or after transfer yields the same value up to floating-point rounding, but doing it first avoids moving redundant elements.

**Reference.** NVIDIA CUDA Best Practices Guide recommends minimizing host-device transfers and moving less data whenever possible: [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/).

## 8. Cache snapshot activations for repeated trajectory runs

**Problem.** During notebook iteration, trajectory similarity is often rerun after plotting or MIA changes. Without caching, the code reloads every snapshot and recomputes activations even when checkpoints are unchanged.

**Change.** `compute_epoch_rows_from_snapshots_multi_reference` now supports activation caching to `.npz` files (one cache file per epoch per algorithm). `run_trajectory_analysis` enables this by default via:

- `TrajectoryExperimentConfig.cache_similarity_activations=True`
- `TrajectoryExperimentConfig.similarity_activation_cache_subdir="similarity_activation_cache"`

When cache files exist, similarity scoring reuses cached activations and skips snapshot forward passes.

**Why this is valid.** Similarity metrics are deterministic functions of `(snapshot checkpoint, dataloader order, layer list, activation extraction code)`. Reusing previously materialized activations for the same inputs does not change metric definitions; it only avoids repeated feature extraction.

## 9. Remove repeated per-metric preprocessing inside pair evaluation

**Problem.** For each layer, `evaluate_pair_rows` iterates over all metrics. Before this change, each metric class repeated similar preparation work (reshape/contiguous conversion, row normalisation, probability conversion for KL), so the same arrays were transformed multiple times per layer.

**Change.** `evaluate_pair_rows` now prepares each layer pair once (`float32`, 2D), then computes reusable intermediates once:

- row-normalised arrays for cosine/euclidean metrics,
- row-wise probability arrays for symmetric KL.

The final scalar formulas are unchanged from the metric classes.

**Why this is valid.** These preprocessing operations are deterministic algebraic transforms of `X` and `Y`. Hoisting them out of the per-metric loop only removes duplicate computation and does not change the metric definitions.

## 10. Use uncompressed NPZ for activation cache

**Problem.** Compressed cache files (`np.savez_compressed`) reduce disk footprint, but add CPU overhead on every cache write/read due to DEFLATE compression. In iterative notebook work, this overhead can dominate reruns.

**Change.** Activation cache writes in `trajectories.py` now use `np.savez` (uncompressed `.npz`) to prioritise throughput over storage size.

**Why this is valid.** The arrays stored are identical; only container compression changes. Similarity scores are unaffected.

**Reference.** NumPy I/O docs describe `savez` (uncompressed) vs `savez_compressed` (ZIP/DEFLATE compression): [numpy.savez](https://numpy.org/doc/stable/reference/generated/numpy.savez.html), [numpy.savez_compressed](https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html).

## 11. Reuse precomputed reference-side transforms across all snapshot epochs

**Problem.** In trajectory similarity we compare each snapshot against two fixed references (`original`, `retrained`) across many unlearning steps. Before this change, per-layer preprocessing for those references (reshape/contiguous conversion, row normalisation for cosine/euclidean, and row-wise probability conversion for symmetric KL) was recomputed on every epoch.

**Change.** Added two helpers in `similarity.py`:

- `prepare_activations_for_evaluation(...)` precomputes per-layer reusable arrays (`raw`, `row_norm`, `prob`) once.
- `evaluate_pair_rows_prepared(...)` consumes precomputed structures and computes the same metric scalars.

`run_trajectory_analysis(...)` now accepts `reference_activation_preparer`, and the notebook wiring prepares `original`/`retrained` once, then reuses them for all epochs.

**April 29 follow-up (step 1 implementation for this notebook).** We also removed duplicate *snapshot-side* preprocessing across references: in `compute_epoch_rows_from_snapshots_multi_reference`, each epoch now prepares snapshot activations once (if an activation preparer is provided) and reuses that prepared structure for both `vs retrained` and `vs original` scoring. Previously, the same snapshot activations were prepared once per reference, which duplicated deterministic array transforms.

**Why this is valid.** These transforms are deterministic functions of a fixed activation matrix. Reusing them does not alter metric definitions; it only removes repeated linear algebra and array passes for unchanged inputs (on both reference and snapshot sides).

**Reference.** This follows the standard systems principle of avoiding redundant work by memoization/caching pure deterministic computations. For a general discussion in Python practice, see the stdlib `functools` cache documentation: [Python docs: functools.cache / lru_cache](https://docs.python.org/3/library/functools.html).

## 12. Small timing utility for trajectory profiling

**Motivation.** To support fair before/after comparisons of acceleration changes, we need machine-readable timing logs rather than only console prints.

**Change.** `run_trajectory_analysis(...)` now records stage runtimes (seconds) and writes:

- `out/trajectory_timing_seconds.csv` with columns `stage,seconds`,
- the same table to W&B as `tables/trajectory_timing_seconds` (when W&B is active).

`TrajectoryExperimentArtifacts` now carries:

- `timing_csv_path: str`
- `timing_seconds: Dict[str, float]`

and the notebook helper prints the CSV path after the trajectory pipeline finishes.

**Typical stage keys.** The CSV includes top-level stages (for example `trajectory_total`, `build_test_loader`, `prepare_mia_subsets_and_loaders`), plus per-algorithm entries such as:

- `similarity_ga_total`, `similarity_ssd_total`, ...
- `similarity_ga_vs_retrained`, `similarity_ga_vs_original`, ...
- `mia_ga_total`, `mia_ssd_total`, ...

**Why this is methodologically useful.** It makes runtime claims reproducible and allows direct aggregation across repeated runs (mean/variance) under fixed hardware and config.

## 13. Precompute fixed reference-side linear algebra for CCA and linear CKA

**Problem.** In trajectory analysis, each snapshot is compared against fixed references (`original`, `retrained`) for many epochs. Before this change, CCA and linear CKA recomputed reference-side centering and matrix factorizations at every epoch, even though the reference activations do not change.

**Change.** During reference preparation (`prepare_activations_for_evaluation(..., precompute_metric_reference_cache=True)`), we now cache:

- **CCA reference cache:** subsampled reference columns (same deterministic seed rule), centered reference activations, and reduced QR factor `Q_y`.
- **Linear CKA reference cache:** centered reference activations `Y_c` and `||Y_c^T Y_c||_F`.

At scoring time, snapshot activations only compute the snapshot-side terms and reuse these cached reference terms.

**Why this is valid.**

- For CCA, canonical correlations are singular values of `Q_x^T Q_y`. If `Q_y` is fixed for a fixed reference, reusing it is algebraically identical to recomputing it each epoch.
- For linear CKA, the numerator and denominator are built from centered cross-covariance/covariance terms. Reusing the fixed reference-centered terms preserves the same formula and only removes redundant repeated work.

**References.**

- Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). *Similarity of Neural Network Representations Revisited*. arXiv:1905.00414.
- Hardoon, D. R., Szedmak, S., & Shawe-Taylor, J. (2004). *Canonical Correlation Analysis: An Overview with Application to Learning Methods*. Neural Computation, 16(12), 2639-2664.
