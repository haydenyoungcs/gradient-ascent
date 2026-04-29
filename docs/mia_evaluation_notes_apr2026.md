# MIA evaluation notes (April 2026)

This note records a methodological fix to make the trajectory MIA probe more reliable for dissertation reporting.

## 1. Deterministic member-side preprocessing for MIA

**Problem.** The training dataset uses random crop + horizontal flip. Before this fix, MIA member probes were sampled directly from that train dataset object, so each MIA pass saw stochastically augmented member images. Non-member probes came from the test dataset with deterministic preprocessing. This introduces avoidable distribution noise in the membership task itself.

**Change.** Added `clone_dataset_with_eval_transform(...)` in `src/gradient_ascent/data.py`, and updated trajectory MIA subset construction (`run_trajectory_analysis` in `experiments.py`) to draw member probes from an eval-transform clone of the training dataset.

**Why this is valid.** Membership inference compares confidence/feature statistics between member and non-member examples for the same model checkpoint. Making input preprocessing deterministic at probe time reduces nuisance variance and yields a cleaner estimate of membership signal strength.

## 2. Scope and expected impact

- Training remains unchanged (still uses random augmentations).
- Only MIA probe extraction changes.
- Expected effect: less run-to-run noise and, in many settings, stronger/more stable attack signal if one exists.

## 3. References

- Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017). *Membership Inference Attacks Against Machine Learning Models*. IEEE Symposium on Security and Privacy.
- Yeom, S., Giacomelli, I., Fredrikson, M., & Jha, S. (2018). *Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting*. IEEE Computer Security Foundations Symposium.

## 4. Fixed-fold CV across unlearning epochs

**Problem.** If attacker CV splits are re-sampled at every unlearning epoch, trajectory curves mix model-change effects with attack-split noise.

**Change.** The MIA trajectory path now supports fixed CV folds across epochs (`TrajectoryExperimentConfig.mia_fixed_cv_across_epochs=True` by default). Folds are built once for forget and retain probes and reused for every checkpoint.

**Why this is valid.** This is equivalent to a paired evaluation protocol: each epoch is tested with the same attacker partitioning, so observed differences better isolate model changes.

## 5. Additional reporting metrics: attack advantage + bootstrap CIs

**Change.** LogReg MIA now reports:

- `auc`,
- `advantage = max(TPR - FPR)`,
- bootstrap 95% confidence intervals for AUC and advantage (`*_ci_low`, `*_ci_high`),
- existing low-FPR and mean-member-probability summaries.

**Why this is useful.** AUC gives threshold-independent separability; advantage reflects best operating-point gap; confidence intervals expose uncertainty instead of only point estimates.

## 6. Optional MLP attacker

**Change.** Added an optional MLP attack head (`TrajectoryExperimentConfig.mia_include_mlp_attacker`, now default `True`) evaluated with the same fixed-fold CV protocol and exported as `forget_mlp_*` / `retain_mlp_*` metrics.

**Why this is useful.** Logistic regression is a strong linear baseline, but some membership signals are nonlinear in confidence features. The MLP option helps test whether attack underfitting is masking leakage.

## 7. April 2026 strengthening pass: unbalanced probe option + MLP default-on

**Change A (sampling protocol).** Added `sample_class_subsets(...)` in `src/gradient_ascent/data.py` with a `balance_classes` switch.  
`TrajectoryExperimentConfig` now exposes `mia_balance_probe_classes` (default `False`), so MIA probe construction can either:

- use balanced member/non-member counts per class (`True`, old behavior), or
- use all available class-matched train/test examples (`False`, new default).

**Why this helps.** In CIFAR-10 classwise MIA, balanced sampling forces member probes to be downsampled to test-set size (for frog: 5000 train vs 1000 test). Allowing unbalanced probes keeps more member-side evidence and can improve attack sensitivity, provided metrics are interpreted with care (AUC/ROC-style metrics remain robust to class prior shifts).

**How we knew to do this.** Prior MIA literature treats attack strength as dependent on both attack model and evaluation protocol. Since we already improved protocol stability (deterministic preprocessing and fixed CV folds), the next straightforward strengthening step is reducing unnecessary sample discard in the probe sets.

**Change B (attacker default).** `TrajectoryExperimentConfig.mia_include_mlp_attacker` now defaults to `True` so trajectory runs include both linear and nonlinear attack heads unless explicitly disabled.

**Why this helps.** A linear attack can miss nonlinear membership structure in confidence-derived features; enabling MLP by default reduces the risk of underestimating leakage due to attacker underfitting.

## 8. Additional references for this pass

- Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., & Tramer, F. (2022). *Membership Inference Attacks From First Principles*. IEEE Symposium on Security and Privacy.

## 9. April 2026 second pass: prior-robust attackers under unbalanced probes

**Problem.** After enabling unbalanced probe mode (`mia_balance_probe_classes=False`), member/non-member counts can differ. If the attack model is trained without class weighting, it can partially optimize to the observed class prior instead of only the conditional membership signal.

**Change.** Updated `src/gradient_ascent/mia.py` so attacker training is more prior-robust:

- `mia_logreg(...)` now uses `LogisticRegression(..., class_weight="balanced")`.
- `mia_mlp(...)` uses fixed-fold CV with the same protocol, and we keep the stronger MLP head enabled by default in trajectory runs.

**Why this is valid.** For the linear head, balanced weighting equalizes member/non-member loss contribution in CV training splits, reducing dependence on class priors introduced by unbalanced probe construction. The MLP head still provides nonlinear attack capacity under the same fixed-fold protocol.

**How we knew to do this.** This follows standard imbalanced-classification practice and is consistent with privacy evaluation goals: stronger attacks should capture genuine membership signal rather than base-rate artifacts.

## 10. Baseline parity vs project-specific extensions (Google Research mapping)

To make the attack design traceable to prior work, we map our implementation to the Google Research reference implementation and then state explicit extensions.

**Reference baseline we align to.** We follow the core attack pattern in Google Research `learn_to_forget/membership_inference.py`: supervised membership inference with logistic regression and stratified cross-validation over member/non-member labels ([source](https://github.com/google-research/google-research/blob/master/learn_to_forget/membership_inference.py)).

**Baseline parity (what is directly aligned).**

- Binary membership labels (`member` vs `non-member`) as the attack target.
- Logistic-regression attack head as a primary baseline.
- Stratified cross-validation protocol to evaluate attack performance out-of-sample.
- Loss/confidence-driven attack signal family (our threshold-loss path is directly in this category).

**Project-specific extensions (what we add beyond the reference baseline).**

- **Richer attack features:** confidence/margin/entropy/top-k probability features in addition to loss-only signals.
- **Additional attack head:** optional MLP attacker to test nonlinear leakage not captured by linear models.
- **Trajectory protocol:** fixed-fold CV reused across unlearning steps, so temporal comparisons isolate model updates rather than resampling noise.
- **Targeted probes:** separate forget-class and retain-control probes to evaluate selective unlearning behavior.
- **Uncertainty reporting:** bootstrap confidence intervals for AUC and ROC advantage.
- **Larger probe mode:** option to avoid truncating to balanced minima, reducing sample discard.

**Methodological claim used in this project.** Our pipeline is baseline-compatible with the Google logistic-CV MIA template, and extends it for (i) targeted unlearning evaluation, (ii) stronger attacker capacity checks, and (iii) uncertainty-aware reporting appropriate for dissertation-level experimental claims.

## 11. MLP hyperparameter sweep (nested within CV folds)

**Motivation.** A fixed MLP architecture can underfit or overfit depending on probe size and signal strength. To reduce architecture-selection bias, we tune MLP settings inside each outer CV fold.

**Change.** `mia_mlp(...)` now supports a lightweight fold-local sweep (enabled by default) over:

- hidden layer sizes: `((64, 32), (128, 64), (64,))`
- regularization `alpha`: `(1e-4, 1e-3)`

Selection is done on an inner validation split from the outer training fold, then the chosen configuration is fit on the full outer training fold and evaluated on the held-out outer fold.

**Config knobs (trajectory).**

- `mia_mlp_sweep_enabled`
- `mia_mlp_sweep_hidden_layer_sizes`
- `mia_mlp_sweep_alphas`

**Why this is valid.** This is a standard nested-model-selection pattern that reduces optimistic bias versus choosing one global architecture after observing test-fold outcomes.
