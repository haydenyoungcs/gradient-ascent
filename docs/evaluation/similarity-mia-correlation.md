# Scalar similarity change vs membership inference

## Motivation

Layer-wise similarity trajectories produce **many** numbers per unlearning step (one value per layer and per metric). Forget-set accuracy and MIA scores are naturally **scalar** at each step. This analysis collapses similarity movement into a small set of scalars so you can ask exploratory questions such as: *if the unlearned model moves closer to the retrain-from-scratch reference in representation space, does the forget-set MIA score improve in parallel?*

## Scalar features (top‑k layers by absolute change)

For each similarity metric and each layer:

1. Read the **raw (unscaled)** similarity at the first and last recorded unlearning steps from the trajectory CSV.
2. Define `delta = end − start` and `abs_delta = |delta|`.
3. Rank layers by `abs_delta` and identify the **single most-changed layer** (largest `|delta|`).
4. Report:
   - **`max_changed_layer_delta`**: signed endpoint change on that layer in native units, with a unified sign convention: **positive always means more similar**. For lower-is-better metrics (`euclidean`, `kl_sym`) this is implemented by sign-flipping the raw endpoint difference.
   - **`max_changed_layer_abs_delta`**, **`max_abs_delta`**: still written to the CSV for manual diagnostics; **run-level correlation plots and pooled correlations use signed x-features only** (no duplicate abs-magnitude plots).

`top_k_layers` is still available for legacy table columns, but primary scatter/bar correlation outputs use the single-layer max-change signed feature above.

## Which reference?

- **Retrained** (`retrained`): primary reference for the dissertation question — alignment with the gold *retrain-from-scratch* model that never saw the forgotten class.
- **Original**: optional diagnostic — movement relative to the pre-unlearning teacher does not answer the same causal question as agreement with the retrained reference.

## MIA reduction

Default scalar: **`forget_logreg_mean_member_prob`** — mean out-of-fold logistic-regression membership probability on the **forget-set** probe (see `membership-inference.md`).

Definitions:

- **`mia_delta`** = end − start (raw change in the attack score).
- **`mia_reduction`** = start − end so that **positive** values mean the attack became **less confident** on the forget set after unlearning (the usual “better privacy” direction for this score).

Override the MIA column with `mia_value_col` when comparing other attackers.

## Forget-class accuracy change (utility alongside MIA)

From `classwise_accuracy_<algo>.csv`, the run-level table includes **`forget_accuracy_reduction`** = (forget accuracy at first step) − (forget accuracy at last step), in 0–1 units. **Positive** values mean the forgotten class became **harder** to classify after unlearning (the usual successful-unlearning direction for utility on that class). Pooled correlations and scatter plots use the **same signed similarity deltas** as for MIA, with `y_feature = forget_accuracy_reduction`, so you can read whether representation movement toward the retrain reference aligns with forget-set accuracy drops as well as with MIA reduction.

*Rationale:* Shokri et al. (2017) and Carlini et al. (2022) motivate MIA as a privacy probe; unlearning papers including Bourtoule et al. (2021) also report **task accuracy** on forgotten data. Treating both as outcomes checks whether a representation proxy tracks **privacy** and **utility** signals jointly or only one of them.

## Input data and orientation

- Similarity CSVs on disk (`similarity_vs_unlearning_epoch_<algo>_vs_<reference>.csv`) store **raw** metric values.
- Run-level endpoint deltas in this section are computed on **raw (unscaled)** values, so x-axes are in the original metric units.
- Epoch-wise correlations (Section "Step-wise correlations") still use `orient_epoch_rows_for_similarity` layer means to stay consistent with trajectory figures (higher = more similar for every metric column).
- By default, those similarity CSVs are now generated from **forget-class test data only** (`TrajectoryExperimentConfig.similarity_data_mode="forget"`). This is intentional: the downstream MIA target is also the forget set, so the representation proxy is aligned with the privacy outcome being tested.

### Changing the similarity data source

You can switch representation input data with:

- `"forget"` (default): only forgotten class samples from `testset`
- `"retain"`: all non-forgotten class samples from `testset`
- `"test"`: full `testset`

Section 2 (single target) and Section 4 (multi-target) both support this switch; Section 5 correlations automatically use whichever similarity CSVs were produced upstream.

## Outputs

When you call `run_similarity_mia_correlation_analysis(out_dir, ...)` (or `run_similarity_mia_correlation_from_notebook` after a multitarget run), artefacts are written under:

`<out_dir>/correlation/`

Typical files:

| File | Role |
|------|------|
| `similarity_mia_correlation_table.csv` | One row per algorithm, forget class, reference, and similarity metric |
| `similarity_mia_correlation_summary.csv` | Pooled Pearson/Spearman correlations for **`mia_reduction`** and **`forget_accuracy_reduction`** (`y_feature` column distinguishes them; `p_value` from SciPy) |
| `scatter_<reference>_<metric>_<signed_feature>_vs_<y>.png` | Scatter by algorithm colour with a dashed least-squares line and an inset box with **Pearson/Spearman r**, **p**, and **n** (same pooled points as the plot; `scatter_similarity_vs_mia(..., corr_text_loc=...)` can move the box: `upper left` / `upper right` / …) (`y` is `mia_reduction` or `forget_accuracy_reduction`) |
| `correlation_summary_<reference>_<signed_feature>_vs_<y>.png` | Bar chart of Spearman r across similarity metrics for a chosen signed x feature and outcome `y` |

For multitarget experiments, pass the **aggregate directory** that contains `target_0`, …, `target_9` (default in the notebook helper: `<runtime.out_dir>/multitarget_aggregate`).

## Interpreting correlations

- **Positive r** between `max_changed_layer_delta` (retrained) and `mia_reduction`: runs whose most-shifted layer moved **toward** retrain also tended to show **larger** MIA reductions (attack confidence dropped more).
- **Positive r** between the same signed delta and `forget_accuracy_reduction`: more movement toward the retrain reference tended to co-occur with **larger** drops in forget-class accuracy (aligned privacy and utility signals on the forgotten class).
- **Negative r**: more retrain similarity associated with **smaller** MIA reductions (explore confounders — utility, hyperparameters, or class difficulty).
- **Weak \|r\| with large p**: little linear/monotonic association across the ~50 run-level points; do not over-interpret.
- **Strong \|r\| with small p**: suggestive only — this is **exploratory** analysis on a modest sample size (5 algorithms × 10 forget classes per metric unless you enrich with more runs or intermediate steps).

## Caveats

- About **50** independent-ish rows per metric at the run level (unless you add more algorithms, classes, or seeds).
- Correlation does not imply causation; algorithm and forget class are entangled with both similarity and MIA.
- Top‑k layer selection is a **summarisation choice**; robustness checks can vary k or average over all layers.
- Forget-only similarity can over-focus on target behavior; robustness checks should include retain/full settings and report whether conclusions are stable.

## Step-wise correlations (as unlearning progresses)

For each unlearning **epoch** and each reference (`retrained`, `original`), the pipeline builds a pooled dataset of roughly **50 rows** (5 algorithms × 10 forget classes). Each row has:

- **Layer-mean oriented similarity** for each metric (`sim_mean_cka_linear`, …) — same orientation as trajectory plots.
- **MIA:** default column `forget_logreg_mean_member_prob`.
- **Utility:** `forget_accuracy` and `retain_mean_accuracy` from `classwise_accuracy_<algo>.csv`.

At every epoch it computes **Spearman** and **Pearson** correlations between each similarity mean and each outcome across runs. Artefacts:

- `correlation/epochwise/epoch_level_multitarget_long.csv` — long table of all runs and steps.
- `correlation/epochwise/epochwise_similarity_outcome_correlations.csv` — one row per `(reference, epoch, similarity_metric, outcome)`.
- `correlation/epochwise/epochwise_spearman_r_<reference>_<outcome>.png` — lines over time (one line per similarity metric).

Use this to see whether any metric tracks MIA or accuracy **during** unlearning, not only at endpoints. `min_n` (default 15) skips sparse epochs.

## Code entry points

- `gradient_ascent.correlation.run_similarity_mia_correlation_analysis`
- `gradient_ascent.correlation.run_epochwise_similarity_outcome_correlation`
- `gradient_ascent.pipelines.multitarget` re-exports both
- `gradient_ascent.notebook_helpers.run_similarity_mia_correlation_from_notebook`
- `gradient_ascent.notebook_helpers.run_epochwise_similarity_proxy_analysis_from_notebook`

## References for rationale

- Bourtoule et al. (2021), "Machine Unlearning", IEEE S&P, [https://doi.org/10.1109/SP40001.2021.00019](https://doi.org/10.1109/SP40001.2021.00019)  
- Shokri et al. (2017), "Membership Inference Attacks Against Machine Learning Models", IEEE S&P, [https://doi.org/10.1109/SP.2017.41](https://doi.org/10.1109/SP.2017.41)  
- Carlini et al. (2022), "Membership Inference Attacks From First Principles", IEEE S&P, [https://doi.org/10.1109/SP46214.2022.9833649](https://doi.org/10.1109/SP46214.2022.9833649)  
