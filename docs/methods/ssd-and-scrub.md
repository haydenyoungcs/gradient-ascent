# SSD and SCRUB in this repository

## Why SCRUB is included

SCRUB was in the original project proposal and is a standard **teacher–student** unlearning baseline (Kurmanji et al., *Towards Unbounded Machine Unlearning*). This repository’s SCRUB is an **adapted** implementation integrated with the shared evaluation pipeline, not a claim of exact paper reproduction. It complements:

- **Gradient ascent** — direct loss maximisation on the forget set  
- **SalUn** — saliency-masked updates  
- **SSD** — one-shot dampening  
- **Certified** — last-layer removal approximation  

SCRUB keeps a **frozen teacher** (original model) and trains a **student** from the same init: retain terms pull the student toward the teacher; forget terms push away (negative distillation). A **recovery** phase can restore retain utility after aggressive forgetting.

---

## Runtime context (what was slow)

Historical diagnosis (before fixes):

1. **SSD** — per-sample Fisher via many `autograd.grad` calls per batch dominated cost vs canonical **batch-level** squared-gradient accumulation.
2. **SCRUB** — recovery could approach **full retain-set** epochs; forget phase also does more forward/backward work per step than GA.

See `SSDConfig` / `SCRUBConfig` and `run_scrub_unlearning` in `src/gradient_ascent/unlearning/` for current behaviour.

---

## Methodological controls (current design)

### 1) SSD: batch-level Fisher by default

- `SSDConfig.fisher_mode` defaults to **`"batch"`** (one forward + backward per batch, accumulate `grad²`).
- **`"per_sample"`** remains an optional research mode.

**Rationale.** SSD is positioned as **fast and post-hoc**; batch-level Fisher matches common reference implementations.

**References:** Foster et al., *Fast Machine Unlearning Without Retraining Through Selective Synaptic Dampening*; [if-loops/selective-synaptic-dampening](https://github.com/if-loops/selective-synaptic-dampening).

### 2) SCRUB: hard update caps

- Explicit bounds: `max_forget_batches_per_epoch`, `max_retain_batches_per_epoch`, `recovery_max_forget_batches_per_epoch` (validated non-`None` on main entry path).

**Rationale.** SCRUB should not silently become unrestricted full fine-tuning; caps keep runtime and comparisons reproducible.

**References:** Kurmanji et al., *Towards Unbounded Machine Unlearning*; [meghdadk/SCRUB](https://github.com/meghdadk/SCRUB).

### 3) Fixed-budget SSD/SCRUB sweeps

**Helper:** `run_fixed_budget_ssd_scrub_sweeps(...)` writes e.g. `out/fixed_budget_ssd_scrub_sweeps.csv` under a fixed update budget.

**Rationale.** Tuning without a compute budget biases toward methods that can afford longer schedules.

---

## Practical interpretation for write-ups

1. **Defaults** aim to match algorithm intent (fast SSD; bounded SCRUB).  
2. **Caps** prevent accidental many-epoch retain training in SCRUB.  
3. **Fixed-budget sweeps** support fair hyperparameter search for the dissertation.

---

## Optional: staged recovery plan (from earlier notes)

If runtime or quality needs further tuning: instrument Fisher vs SCRUB phases; prefer batch Fisher for SSD; shorten SCRUB schedules (e.g. few epochs, capped recovery batches); compare on frog-forget and retained metrics before scaling to multitarget runs.
