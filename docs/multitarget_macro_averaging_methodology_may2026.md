# Multi-Target Macro Averaging Methodology (May 2026)

## Why this change was made

Single-forget-class analysis (for example, frog only) can be noisy because some forget classes are easier or harder to remove than others. To reduce this class-selection variance, the notebook helper now supports a full forget-label sweep:

- train/retrain and unlearn once for each forget label in CIFAR-10 (`0..9`),
- collect per-step utility and similarity outputs for every run,
- average those outputs into one per-algorithm aggregate view.

This follows the same motivation as repeated-evaluation practice in ML: reduce dependence on one particular split/seed/task instance and report central tendency more robustly.

## What is now produced

The new helper `run_multitarget_averaged_experiment(...)` in `src/gradient_ascent/notebook_helpers.py` runs all requested forget labels and writes aggregate artefacts under:

- `<out_dir>/multitarget_aggregate/`

### Utility aggregation (per algorithm)

For each unlearning algorithm (`ga`, `ssd`, `salun`, `certified`, `scrub`), we derive two curves at every unlearning step:

1. **Forgotten class accuracy**: for each run, take the accuracy of that run's forgotten label.
2. **Retained mean accuracy**: for each run, average the accuracies of the remaining 9 labels.

Then average both quantities across all forget labels.

Saved files:

- `classwise_binary_aggregate_<algo>.csv`
- `classwise_binary_relative_change_<algo>.png`
- `classwise_binary_absolute_accuracy_<algo>.png`
- `classwise_binary_final_change_bar_<algo>.png`

This gives the requested two-line view ("forgotten class" vs "retained classes mean") and a two-bar final summary.

### Similarity aggregation (per algorithm, per reference)

For each `(algorithm, reference)` pair, the code loads all 10 trajectory CSVs:

- `similarity_vs_unlearning_epoch_<algo>_vs_<reference>.csv`

and computes the mean metric value at each `(epoch, layer)` cell across forget labels. The averaged rows are then fed into the existing plotting pipeline (summary, evolving bars, grouped evolving bars, before/after grouped bars).

Saved files:

- `similarity_vs_unlearning_epoch_<algo>_vs_<reference>_mean_over_targets.csv`
- `similarity_vs_unlearning_epoch_<algo>_vs_<reference>_mean_over_targets_summary.png`
- `similarity_vs_unlearning_epoch_<algo>_vs_<reference>_mean_over_targets_evolving_bars.gif`
- `similarity_vs_unlearning_epoch_<algo>_vs_<reference>_mean_over_targets_grouped_evolving_bars.gif`
- `similarity_vs_unlearning_epoch_<algo>_vs_<reference>_mean_over_targets_before_after.png`

### MIA aggregation (per algorithm)

For each algorithm, the code now also averages MIA trajectories over forget labels:

- input files: `mia_vs_unlearning_epoch_<algo>.csv` from each target run,
- averaging rule: mean each metric at each unlearning step across targets,
- output files:
  - `mia_vs_unlearning_epoch_<algo>_mean_over_targets.csv`
  - `mia_frog_trajectory_<algo>_mean_over_targets.png`
  - `mia_forget_vs_retain_logreg_mean_member_prob_<algo>_mean_over_targets.png`

The retrained baseline is averaged the same way from each target's
`mia_retrained_baseline.csv`, producing:

- `mia_retrained_baseline_mean_over_targets.csv`

## How I knew this was a reasonable approach

1. **Macro-style averaging across categories** is standard when per-class behavior differs and we do not want one class to dominate interpretation.
2. **Repeated evaluation / aggregation** is a standard method for reducing high variance in experimental claims.
3. In this project, unlearning quality is already judged jointly across utility and similarity trajectories, so averaging both over forget labels makes the evidence less sensitive to the frog-only choice.

## Caveats

- This does **not** remove all noise (for example, optimizer stochasticity still exists), but it removes one major source: forget-label selection.
- Averaging can hide class-specific failure modes, so per-label runs should still be kept for appendix diagnostics.
- Runtime is much higher because this multiplies core and trajectory work by the number of forget labels.

## References

1. Dietterich, T. G. (1998). Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. *Neural Computation*, 10(7), 1895-1923.
2. Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. *Information Processing & Management*, 45(4), 427-437.
3. Bouthillier, X., Laurent, C., & Vincent, P. (2021). Accounting for Variance in Machine Learning Benchmarks. *Proceedings of Machine Learning and Systems*, 3, 747-769.
