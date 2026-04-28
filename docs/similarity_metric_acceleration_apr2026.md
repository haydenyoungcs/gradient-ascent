# Similarity metric acceleration (April 2026)

This note records why two changes were made to the trajectory similarity stage and how to override them for stricter (slower) experiments.

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
