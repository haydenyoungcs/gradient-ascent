# Similarity trajectories: plots and performance

This note covers **what figures the trajectory stage produces**, **why they look that way**, and **how similarity computation was accelerated** without changing metric definitions.

---

## Part A — Visualisation choices

### Evolving bar GIFs (per step)

For each algorithm and reference (`retrained`, `original`), the pipeline saves:

- a static **summary** line figure (mean/min/max over layers), and  
- an **evolving bar chart GIF**: one frame per unlearning step.

**Per frame:** layer-wise metrics are computed, lower-is-better metrics are oriented so **higher = more similar**, then the **mean across layers** is taken for each metric — so the GIF tracks **model-level** similarity trends derived from all tracked layers.

**Why bars + animation:** stepwise bar charts make cross-metric comparisons easy for readers; the GIF preserves temporal order (step 0 → final) without many separate static figures. (Heatmaps were tried but were harder to explain in thesis narrative.)

**Outputs:** pattern `..._evolving_bars.gif`; see `trajectories.py` (`save_similarity_evolving_bar_plot`), `experiments/trajectory.py`, `notebook_helpers.py`.

### Before/after grouped bar plot

**Purpose.** GIFs help inspect full trajectories; a **single static** figure compares **only step 0 vs final** for each `(algorithm, reference)`.

**Layout:**

- x-axis: metric groups (`cka_linear`, `cca`, `cosine`, `euclidean`, `kl_sym`)
- within each group: subgroups **Before** / **After**
- within each subgroup: one bar per tracked layer

**File pattern:** `similarity_vs_unlearning_epoch_<algo>_vs_<reference>_before_after_grouped_bars.png`

**Dissertation use:** describe as endpoint comparison (start vs end), stratified by metric and layer, to relate representation movement to MIA or accuracy outcomes.

**Implementation:** `save_similarity_before_after_grouped_bar_plot` in `trajectories.py`; uses the same orientation as `orient_epoch_rows_for_similarity` in `reporting.py`.

### References (visualisation)

- Cleveland & McGill (1984), *Graphical Perception*, JASA.  
- Heer & Robertson (2007), *Animated Transitions in Statistical Data Graphics*, IEEE TVCG.  
- Tufte (2001), *The Visual Display of Quantitative Information*.

---

## Part B — Performance and numerical details

### 1. Linear CKA without n×n Gram matrices

**Was:** build `n×n` Gram matrices for both sides.  
**Now:** column-centred activations and Frobenius norms of `d×d` statistics (Kornblith et al., *Similarity of Neural Network Representations Revisited*, arXiv:1905.00414). Numeric parity checked in `tests/test_similarity_acceleration.py`.

### 2. Subsampled activation rows

`TrajectoryExperimentConfig.max_activation_samples` (default `1024`) and `activation_subsample_seed` cap rows **after** concatenation with the **same index set** per layer. Set `None` for full rows (slower).

### 3. CCA cost

- `numpy.linalg.svd(..., compute_uv=False)` when only singular values are needed.  
- Optional **column** subsampling: `cca_max_columns` (e.g. 256); analogous in spirit to SVCCA (Raghu et al., 2017). `None` = full width (slowest).

### 4. Progress logging

`TrajectoryExperimentConfig.log_similarity_progress` prints epoch-level progress for activation collection and per-reference scoring (not every layer/metric).

### 5. One forward pass per snapshot for two references

`compute_epoch_rows_from_snapshots_multi_reference` loads each snapshot once, collects activations once, scores against **both** `retrained` and `original`. Snapshot activations do not depend on which reference is used — only the pairwise metric does.

### 6. Single row subsample

Row capping applies in **activation collection**; pair evaluation should not apply a second random cap (`max_activation_samples=None` there) to avoid double subsampling.

### 7. Pool on GPU before CPU transfer

For `N×C×H×W` tensors, **spatial mean** to `N×C` on device before `.cpu()` reduces transfer; same pooled representation the metrics use.

### 8. Activation cache

Optional `.npz` cache per epoch/algorithm (`cache_similarity_activations`, `similarity_activation_cache_subdir`). Uncompressed `np.savez` for faster iteration than compressed.

### 9. Shared preprocessing in `evaluate_pair_rows`

Per layer, prepare arrays once (float32 2D, row-norm for cosine/Euclidean, probabilities for KL), then run metric formulas.

### 10. Precomputed reference-side terms for CKA/CCA

Reference activations are fixed across epochs: cache centred matrices / QR factors / Frobenius terms once, reuse when scoring each snapshot.

### 11. Shared-index CCA column subsampling

When both sides have the **same** feature width, use one random column index set for both `X` and `Y` so identical activations at step 0 still give CCA ≈ 1 (up to numerics).

### 12. Timing

`run_trajectory_analysis` can write `out/trajectory_timing_seconds.csv` and per-metric bars `out/similarity_metric_timing_<algo>.png` for cost breakdown.

### 13. Optional stride

`similarity_step_stride` > 1 evaluates fewer checkpoints (coarser trajectories, faster).

### References

- Kornblith et al. (2019), arXiv:1905.00414.  
- Raghu et al. (2017), SVCCA.  
- Hardoon et al. (2004), CCA overview.  
- NVIDIA CUDA best practices (minimise H2D transfer).  
- NumPy `savez` vs `savez_compressed` documentation.
