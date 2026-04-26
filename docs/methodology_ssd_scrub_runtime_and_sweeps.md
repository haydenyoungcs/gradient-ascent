# Methodology Note: SSD/SCRUB Runtime Controls and Fixed-Budget Sweeps

## Why these code changes are methodologically valid

This project compares unlearning algorithms on CIFAR-10 (frog forget class) using a ResNet-50 backbone. The changes below are intended to preserve each method's algorithmic intent while preventing implementation details from unfairly dominating runtime.

## 1) SSD default moved to batch-level Fisher accumulation

### What changed

- `SSDConfig` now includes `fisher_mode` with default `"batch"`.
- The previous per-sample Fisher estimator remains available as `"per_sample"` research mode.

### Why this is valid

SSD is introduced as a **fast, retrain-free, post-hoc** method. The official repository implementation computes importance by accumulating squared parameter gradients from standard batch backpropagation steps. This corresponds to a practical diagonal Fisher approximation and keeps compute low enough for SSD's intended use case.

Using per-sample `autograd.grad` for every selected example can greatly increase runtime without being required by the SSD objective itself. Keeping that mode optional preserves research flexibility while aligning the default with canonical "fast SSD" behavior.

References:

1. Foster, Schoepf, Brintrup. *Fast Machine Unlearning Without Retraining Through Selective Synaptic Dampening.* AAAI 2024 / arXiv:2308.07707.  
   [https://arxiv.org/abs/2308.07707](https://arxiv.org/abs/2308.07707)
2. Official SSD code repository (if-loops):  
   [https://github.com/if-loops/selective-synaptic-dampening](https://github.com/if-loops/selective-synaptic-dampening)

## 2) SCRUB now enforces hard update caps by default

### What changed

- `SCRUBConfig` defaults now include explicit bounded values for:
  - `max_forget_batches_per_epoch`
  - `max_retain_batches_per_epoch`
  - `recovery_max_forget_batches_per_epoch`
- `run_scrub_unlearning(...)` now validates that these three caps are set (non-`None`), so all entry paths are hard-capped unless code is intentionally modified.

### Why this is valid

SCRUB is iterative and naturally more expensive than one-shot methods, but its purpose is not to perform unrestricted full retain-set re-training each unlearning run. In practice, bounded destructive+recovery schedules are a reasonable implementation choice for class-unlearning comparisons where runtime is an evaluation axis.

The core SCRUB mechanism (teacher-student divergence behavior on forget data plus retain-side utility preservation) remains intact; only the number of update steps is constrained for fair and reproducible benchmarking.

References:

1. Kurmanji et al. *Towards Unbounded Machine Unlearning.* NeurIPS 2023 / arXiv:2302.09880.  
   [https://arxiv.org/abs/2302.09880](https://arxiv.org/abs/2302.09880)
2. Official SCRUB code repository (meghdadk):  
   [https://github.com/meghdadk/SCRUB](https://github.com/meghdadk/SCRUB)

## 3) Fixed-budget SSD/SCRUB sweeps added for fair tuning

### What changed

- Added notebook helper `run_fixed_budget_ssd_scrub_sweeps(...)`.
- It runs compact grids under fixed update budgets and writes `out/fixed_budget_ssd_scrub_sweeps.csv`.
- Swept knobs:
  - SSD: `alpha`, `lambda_`
  - SCRUB: `beta`, `recovery_beta_scale`, retain cap

### Why this is valid

Hyperparameter tuning is necessary for stable forgetting/utility trade-offs. However, tuning under unconstrained compute can bias results toward methods with longer schedules. Fixed-budget sweeps provide a fairer comparison by evaluating quality differences at similar computational budgets.

This is especially important for dissertation reporting where algorithm quality and efficiency are both claims under comparison.

## Practical interpretation for dissertation text

You can justify this methodology as:

1. **Algorithm-faithful defaults:** SSD default follows canonical fast usage; SCRUB keeps its core objective structure.
2. **Fair runtime control:** hard update caps avoid accidental full fine-tuning behavior in unlearning runs.
3. **Reproducible tuning:** small, fixed-budget sweeps reveal robust settings without confounding from variable compute spend.

