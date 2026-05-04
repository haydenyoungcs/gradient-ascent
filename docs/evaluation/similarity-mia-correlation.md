# Scalar similarity change vs membership inference

## Motivation

Layer-wise similarity trajectories produce **many** numbers per unlearning step (one value per layer and per metric). Forget-set accuracy and MIA scores are naturally **scalar** at each step. This analysis collapses similarity movement into a small set of scalars so you can ask exploratory questions such as: *if the unlearned model moves closer to the retrain-from-scratch reference in representation space, does the forget-set MIA score improve in parallel?*

## Scalar features (top‑k layers by absolute change)

For each similarity metric and each layer:

1. Read the **oriented** similarity at the first and last recorded unlearning steps (same min–max orientation as the trajectory plots: higher = more similar).
2. Define `delta = end − start` and `abs_delta = |delta|`.
3. Rank layers by `abs_delta` and take the top **k** layers (default **k = 2**), motivated by the common pattern that one or two blocks move most during unlearning.
4. Report:
   - **`top2_mean_delta`** (when k = 2): mean of the signed deltas on the two selected layers — positive means the run ended **more similar** to the reference than it started, on average over those layers.
   - **`top2_mean_abs_delta`**: mean absolute delta — **how much** representation moved toward or away from the reference, ignoring direction.
   - **`max_abs_delta`**: the largest single-layer absolute movement (diagnostic).

For k ≠ 2, the table uses `topk_mean_delta` / `topk_mean_abs_delta` instead of the `top2_*` names.

## Which reference?

- **Retrained** (`retrained`): primary reference for the dissertation question — alignment with the gold *retrain-from-scratch* model that never saw the forgotten class.
- **Original**: optional diagnostic — movement relative to the pre-unlearning teacher does not answer the same causal question as agreement with the retrained reference.

## MIA reduction

Default scalar: **`forget_logreg_mean_member_prob`** — mean out-of-fold logistic-regression membership probability on the **forget-set** probe (see `membership-inference.md`).

Definitions:

- **`mia_delta`** = end − start (raw change in the attack score).
- **`mia_reduction`** = start − end so that **positive** values mean the attack became **less confident** on the forget set after unlearning (the usual “better privacy” direction for this score).

Override the MIA column with `mia_value_col` when comparing other attackers.

## Input data and orientation

- Similarity CSVs on disk (`similarity_vs_unlearning_epoch_<algo>_vs_<reference>.csv`) store **raw** metric values.
- This analysis applies `orient_epoch_rows_for_similarity` from `reporting.py` before computing deltas, matching the thesis figures (higher = more similar for every metric column).

## Outputs

When you call `run_similarity_mia_correlation_analysis(out_dir, ...)` (or `run_similarity_mia_correlation_from_notebook` after a multitarget run), artefacts are written under:

`<out_dir>/correlation/`

Typical files:

| File | Role |
|------|------|
| `similarity_mia_correlation_table.csv` | One row per algorithm, forget class, reference, and similarity metric |
| `similarity_mia_correlation_summary.csv` | Pooled Pearson/Spearman correlations (`p_value` from SciPy) |
| `scatter_<reference>_<metric>_<feature>_vs_mia_reduction.png` | Scatter by algorithm colour with a dashed least-squares line |
| `correlation_summary_<reference>_<feature>.png` | Bar chart of Spearman r across metrics for a chosen x feature |

For multitarget experiments, pass the **aggregate directory** that contains `target_0`, …, `target_9` (default in the notebook helper: `<runtime.out_dir>/multitarget_aggregate`).

## Interpreting correlations

- **Positive r** between `top2_mean_delta` (retrained) and `mia_reduction`: runs that became **more** similar to retrain also tended to show **larger** MIA reductions (attack confidence dropped more).
- **Negative r**: more retrain similarity associated with **smaller** MIA reductions (explore confounders — utility, hyperparameters, or class difficulty).
- **Weak \|r\| with large p**: little linear/monotonic association across the ~50 run-level points; do not over-interpret.
- **Strong \|r\| with small p**: suggestive only — this is **exploratory** analysis on a modest sample size (5 algorithms × 10 forget classes per metric unless you enrich with more runs or intermediate steps).

## Caveats

- About **50** independent-ish rows per metric at the run level (unless you add more algorithms, classes, or seeds).
- Correlation does not imply causation; algorithm and forget class are entangled with both similarity and MIA.
- Top‑k layer selection is a **summarisation choice**; robustness checks can vary k or average over all layers.

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
