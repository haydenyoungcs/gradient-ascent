# SCRUB + SSD Runtime Diagnosis and Recovery Plan (ResNet-50, CIFAR-10 Frog Forgetting)

## Context and goal

This note explains why the current `SCRUB` and `SSD` implementations in this repository are much slower than expected, compares them to canonical implementations, and proposes a concrete plan to improve both runtime and unlearning quality for the *frog* forget set on CIFAR-10.

The objective is to return `SCRUB` and `SSD` to practical runtimes while improving forget quality (frog suppression) without excessive collateral damage on retained classes.

## What I compared

Local code examined:

- `src/gradient_ascent/unlearning/ssd.py`
- `src/gradient_ascent/unlearning/scrub.py`
- `src/gradient_ascent/experiments.py`
- `src/gradient_ascent/notebook_helpers.py`

Reference sources:

1. Foster et al., **Selective Synaptic Dampening (SSD)**, AAAI 2024 / arXiv:2308.07707.  
   URL: [https://arxiv.org/abs/2308.07707](https://arxiv.org/abs/2308.07707)
2. Official SSD repository (if-loops):  
   URL: [https://github.com/if-loops/selective-synaptic-dampening](https://github.com/if-loops/selective-synaptic-dampening)
3. Kurmanji et al., **SCRUB**, NeurIPS 2023 / arXiv:2302.09880.  
   URL: [https://arxiv.org/abs/2302.09880](https://arxiv.org/abs/2302.09880)
4. Official SCRUB repository (meghdadk):  
   URL: [https://github.com/meghdadk/SCRUB](https://github.com/meghdadk/SCRUB)

## Diagnosis: why runtime increased so much

### 1) SSD is currently doing expensive per-sample Fisher gradients

In local `estimate_empirical_fisher_diag(...)`, each selected sample in each batch calls `torch.autograd.grad(...)` separately, i.e. many backward-like passes per batch.

With typical settings:

- `fisher_batches = 80`
- `fisher_samples_per_batch = 16`
- Fisher for both forget and retain loaders

This gives roughly `80 * 16 * 2 = 2560` per-sample gradient evaluations.

By contrast, canonical SSD implementations typically accumulate squared gradients from **one batch backward pass** (or similarly cheap approximations) per batch, not one backward pass per sample. The official SSD code follows this cheaper batch-level pattern (`loss.backward()` then `grad^2` accumulation).

**Why this matters:** this is the dominant source of SSD runtime inflation.

### 2) SCRUB recovery phase still performs many retain updates

In local `run_scrub_unlearning(...)`:

- Forget phase loops over forget batches and paired retain batches.
- Recovery phase (default path when `recovery_beta_scale == 0`) still runs a full retain-loader training loop each recovery epoch unless capped.

For CIFAR-10 frog forgetting:

- Forget set size is ~5k examples (`~20` batches at batch size 256).
- Retain set size is ~45k examples (`~176` batches at batch size 256).

So recovery can become near fine-tuning on the whole retain set for multiple epochs. That substantially increases runtime versus methods like GA that use far fewer updates.

### 3) SCRUB has extra compute per update compared to GA

Each SCRUB update can include:

- student forward on retain,
- teacher forward on retain,
- (and in forget phase) student forward on forget,
- then backward on the combined objective.

That is inherently heavier than GA-style single-loss updates.

### 4) Runtime-sensitive settings may not match notebook “safe” defaults

`notebook_helpers.py` defines constrained defaults (`max_forget_batches_per_epoch`, `max_retain_batches_per_epoch`, reduced SSD sampling), but if experiments run through a different config path, those caps may not be applied. This can silently reintroduce long runs.

## Why quality may still be weak (even with high runtime)

### SSD quality

- Very aggressive or noisy Fisher estimates can damp too many parameters, harming retained classes.
- Using `selection_basis="retain"` is generally more stable in your setup than the union-based reference at 10% class forget ratios; this matches your local comments and is reasonable.
- Per-sample Fisher with limited random samples can be both expensive and noisy; slower does not always mean better here.

### SCRUB quality

- If recovery dominates (too many retain-only steps), forget signal can be undone.
- If forget pressure is too strong/unstable, collateral drops can occur in non-forgotten classes.
- Good behavior usually comes from balancing short destructive steps + constrained recovery, not simply “more epochs”.

## Plan to fix runtime first, then improve quality

## Phase A — instrument and confirm bottlenecks (1 run)

1. Add lightweight timers around:
   - SSD forget Fisher estimation
   - SSD retain/reference Fisher estimation
   - SCRUB forget phase
   - SCRUB recovery phase
2. Log:
   - number of backward/grad calls,
   - number of optimizer steps,
   - loader batch counts actually executed.
3. Keep one CSV per run in `out/` for reproducible reporting.

Success criterion: identify exact wall-clock contribution per stage.

## Phase B — SSD runtime correction (highest priority)

1. Implement a **batch-level Fisher mode** matching official SSD style:
   - one forward + one backward per batch,
   - accumulate `grad^2` per parameter tensor.
2. Keep current per-sample estimator as an optional “research mode”, but default to batch-level for main experiments.
3. Start with:
   - `fisher_batches`: 40, 60, 80 sweep
   - `fisher_samples_per_batch`: ignored in batch-level mode
4. Compare runtime + forget/retain metrics to current baseline.

Expected impact: major SSD speedup (often order-of-magnitude level).

## Phase C — SCRUB runtime correction

1. Enforce strict update budgets in all experiment entry points:
   - `max_forget_batches_per_epoch` set explicitly
   - `max_retain_batches_per_epoch` set explicitly
   - `recovery_max_forget_batches_per_epoch` set explicitly if forget term retained
2. Use a short schedule for ResNet-50 first:
   - 4–6 epochs total
   - 1–2 forget-phase epochs
   - capped recovery batches (e.g., 10–30 retain batches/epoch)
3. Keep teacher frozen and BN frozen as already implemented.

Expected impact: SCRUB runtime closer to GA/SalUn scale instead of near retraining.

## Phase D — quality tuning with fixed compute budget

Run small grid searches with **equalized runtime budgets**:

- SSD:
  - `alpha` in {6, 8, 10}
  - `lambda_` in {0.7, 0.9, 1.0}
  - Fisher batches in {40, 60, 80}
- SCRUB:
  - `beta` in {0.2, 0.4, 0.8}
  - `recovery_beta_scale` in {0.0, 0.1, 0.2}
  - retain batch cap in {10, 20, 30}

Track:

- frog class forgetting delta,
- mean retained-class accuracy,
- worst non-frog class drop,
- runtime (seconds),
- MIA signal if already in your pipeline.

Pick Pareto-optimal settings (best forgetting/utility at acceptable runtime).

## Phase E — dissertation-facing validation

For each chosen config, report:

1. exact hyperparameters,
2. runtime decomposition by stage,
3. classwise deltas (especially frog and worst collateral class),
4. comparison against retrain-from-scratch and strong baselines (GA/SalUn/certified),
5. at least 3 seeds for confidence intervals if budget allows.

## Practical next implementation steps

1. Add `fisher_mode: {"batch","per_sample"}` to `SSDConfig` (default `"batch"`).
2. Implement batch-level Fisher accumulation path in `ssd.py`.
3. Add stage timers/CSV logging for both `SSD` and `SCRUB`.
4. Ensure `experiments.py` always uses explicit scrub/ssd configs from notebook helper defaults (or explicitly prints effective config at runtime).

## Notes on methodological choices

- The SSD paper frames SSD as fast and retrain-free; matching that claim in this codebase requires Fisher estimation that is computationally light enough to preserve SSD’s intended advantage.
- SCRUB is iterative and expected to be slower than one-shot methods, but should still avoid drifting into near full fine-tuning unless intentionally configured for that.
- Equalizing runtime budgets across methods is important for fair comparison in the dissertation.

