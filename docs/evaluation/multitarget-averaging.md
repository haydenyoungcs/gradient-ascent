# Multi-target macro averaging

## Why this exists

Single-forget-class analysis (for example, frog only) can be noisy: some classes are easier or harder to remove. The notebook helper supports a **full forget-label sweep** over CIFAR-10 (`0..9`), then **averages** per-step utility, similarity, and MIA curves across those runs. That reduces dependence on one chosen class and matches common practice of reporting a central tendency over task instances.

## Implementation

**Entry point:** `run_multitarget_averaged_experiment(...)` in `src/gradient_ascent/pipelines/multitarget.py` (also exported from `notebook_helpers` for the notebook).

**Output root:** `<out_dir>/multitarget_aggregate/`

### Utility (per algorithm)

At each unlearning step, for each forget-label run:

1. **Forgotten class accuracy** — accuracy on the class being unlearned in that run.
2. **Retained mean accuracy** — mean accuracy over the other nine classes.

Then **average** both series across the ten forget-label runs.

**Typical artefacts:**

- `classwise_binary_aggregate_<algo>.csv`
- `classwise_binary_relative_change_<algo>.png`
- `classwise_binary_absolute_accuracy_<algo>.png`
- `classwise_binary_final_change_bar_<algo>.png`

### Similarity (per algorithm, per reference)

For each `(algorithm, reference)` pair, load the ten trajectory CSVs:

- `similarity_vs_unlearning_epoch_<algo>_vs_<reference>.csv`

Take the **mean** of each metric at each `(epoch, layer)` across targets, then pass through the usual plotting pipeline (summary, evolving bars, grouped evolving bars, before/after).

**Typical artefacts:**

- `similarity_vs_unlearning_epoch_<algo>_vs_<reference>_mean_over_targets.csv`
- `..._mean_over_targets_summary.png`
- `..._mean_over_targets_evolving_bars.gif`
- `..._mean_over_targets_grouped_evolving_bars.gif`
- `..._mean_over_targets_before_after.png`

### MIA (per algorithm)

Average trajectory metrics at each unlearning step across the ten targets.

**Inputs:** `mia_vs_unlearning_epoch_<algo>.csv` per target.

**Outputs:**

- `mia_vs_unlearning_epoch_<algo>_mean_over_targets.csv`
- `mia_frog_trajectory_<algo>_mean_over_targets.png` (naming may reflect legacy “frog”; curves are averaged over all forget classes)
- `mia_forget_vs_retain_logreg_mean_member_prob_<algo>_mean_over_targets.png`

**Retrained baseline:** averaged from each target’s `mia_retrained_baseline.csv` → `mia_retrained_baseline_mean_over_targets.csv`.

## Why this is reasonable

1. **Macro-style averaging** across categories is standard when per-class behaviour differs and no single class should dominate the narrative.
2. **Repeated evaluation** reduces variance in experimental claims.
3. Unlearning quality here is judged jointly across utility and similarity; averaging both over forget labels makes conclusions less sensitive to a single class choice.

## Caveats

- Does **not** remove all noise (e.g. optimizer stochasticity).
- Averaging can **hide** class-specific failure modes — keep per-label runs for appendices or diagnostics.
- **Runtime** scales roughly with the number of forget labels.

## References

1. Dietterich, T. G. (1998). Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. *Neural Computation*, 10(7), 1895–1923.
2. Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. *Information Processing & Management*, 45(4), 427–437.
3. Bouthillier, X., Laurent, C., & Vincent, P. (2021). Accounting for Variance in Machine Learning Benchmarks. *Proceedings of Machine Learning and Systems*, 3, 747–769.
