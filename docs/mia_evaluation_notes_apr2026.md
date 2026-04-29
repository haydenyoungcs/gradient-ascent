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

## 9. April 2026 second pass: class-weighted attackers under unbalanced probes

**Problem.** After enabling unbalanced probe mode (`mia_balance_probe_classes=False`), member/non-member counts can differ. If the attack model is trained without class weighting, it can partially optimize to the observed class prior instead of only the conditional membership signal.

**Change.** Updated `src/gradient_ascent/mia.py` so attacker training is explicitly prior-robust:

- `mia_logreg(...)` now uses `LogisticRegression(..., class_weight="balanced")`.
- `mia_mlp(...)` now computes fold-local balanced sample weights and passes them to `MLPClassifier.fit(...)`.

**Why this is valid.** Balanced weighting equalizes the loss contribution of member and non-member labels inside each CV training split. This keeps the attacker focused on separability in features rather than raw base-rate differences introduced by unbalanced probe construction.

**How we knew to do this.** This follows standard supervised-learning practice for imbalanced binary classification and is consistent with privacy evaluation goals: when strengthening MIA, we want improved detection of genuine membership signal, not an artifact of label frequency.
