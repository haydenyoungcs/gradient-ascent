# Membership inference evaluation

This document records the **trajectory MIA** design so dissertation text and plots stay aligned with the code (`src/gradient_ascent/data.py`, `experiments/trajectory.py`, `mia.py`).

## 1. Deterministic member-side preprocessing

**Issue.** Training uses random crop and horizontal flip. If MIA **member** probes are drawn from the training `Dataset` object as-is, each attack pass sees **stochastic** augmentations, while non-members use deterministic test preprocessing. That adds avoidable noise to the membership task.

**Change.** `clone_dataset_with_eval_transform(...)` builds an **eval-transform** clone of the training set for member probes; trajectory MIA uses that in `run_trajectory_analysis`.

**Rationale.** The attack compares member vs non-member statistics for a **fixed** checkpoint; probe-time preprocessing should be comparable and stable.

**Scope.** Training is unchanged; only probe extraction changes.

## 2. Fixed-fold CV across unlearning epochs

**Issue.** Resampling attacker CV folds at every epoch mixes **model change** with **split noise**.

**Change.** `TrajectoryExperimentConfig.mia_fixed_cv_across_epochs=True` (default): build folds once for forget and retain probes, reuse for every checkpoint.

**Rationale.** Paired protocol: each epoch is evaluated under the same attacker partitioning.

## 3. Metrics: AUC, advantage, bootstrap CIs

Logistic-regression MIA reports (among others):

- `auc`
- `advantage = max(TPR - FPR)`
- bootstrap 95% CIs for AUC and advantage

**Rationale.** AUC is threshold-agnostic; advantage captures best-case gap; CIs show uncertainty.

## 4. Optional MLP attacker

**Change.** `TrajectoryExperimentConfig.mia_include_mlp_attacker` defaults to **True**: nonlinear head alongside logistic regression, same CV protocol (`forget_mlp_*` / `retain_mlp_*`).

**Rationale.** Linear attacks can underfit nonlinear membership signal in confidence features.

## 5. Unbalanced probes and balanced attack training

**Sampling.** `mia_balance_probe_classes` (default `False`) can use all available class-matched train/test examples instead of downsampling to balanced counts.

**Attacker.** `mia_logreg` uses `class_weight="balanced"` so training is less sensitive to member/non-member count imbalance from unbalanced probes.

**Rationale.** Stronger attacks should reflect genuine membership signal, not base-rate artefacts.

## 6. Mapping to a public baseline (Google Research)

**Aligned with** the pattern in Google Research `learn_to_forget/membership_inference.py`: supervised membership labels, logistic regression, stratified CV.

**Extensions in this project:**

- Richer features (confidence, margin, entropy, top-k, etc.)
- Optional MLP attacker
- Trajectory protocol with **fixed** folds across steps
- Separate forget vs retain-control probes
- Bootstrap CIs
- Option for larger (unbalanced) probe sets

## 7. MLP hyperparameter sweep (nested in CV)

`mia_mlp` can run a small **fold-local** sweep over hidden sizes and L2 `alpha`, chosen on an inner validation split then fit on the outer training fold.

**Config:** `mia_mlp_sweep_enabled`, `mia_mlp_sweep_hidden_layer_sizes`, `mia_mlp_sweep_alphas`.

## References

- Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017). *Membership Inference Attacks Against Machine Learning Models.* IEEE S&P.
- Yeom, S., Giacomelli, I., Fredrikson, M., & Jha, S. (2018). *Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting.* IEEE CSF.
- Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., & Tramer, F. (2022). *Membership Inference Attacks From First Principles.* IEEE S&P.
- Google Research reference script: [learn_to_forget/membership_inference.py](https://github.com/google-research/google-research/blob/master/learn_to_forget/membership_inference.py)
