## SCRUB Baseline Note

SCRUB was added to this repository because it was part of the original project
proposal and because it is a recognised machine unlearning baseline from the
literature. It is useful here because it is methodologically different from the
other baselines already in the project. Gradient ascent directly increases loss
on the forget set, SalUn uses a saliency mask, certified removal focuses on a
formal removal procedure for the last layer, and SSD dampens parameters using
importance estimates. SCRUB gives a fifth comparison point based on
teacher-student distillation.

The method comes from Kurmanji et al., *Towards Unbounded Machine Unlearning*,
and there is also an official public SCRUB implementation. That means the idea
is not something invented just for this repository: it is an established
baseline from the machine unlearning literature and one that was already known
to the project from the proposal stage.

In simple terms, SCRUB keeps a frozen copy of the original trained model as a
teacher and trains a student model to behave differently on the data that
should be forgotten while staying similar on the data that should be retained.
In this repository, the student starts from the original checkpoint. During
unlearning, it is pushed away from the teacher on the forget set using a
negative distillation term, and pulled back toward the teacher on the retain
set using KL-based distillation and retain-label cross-entropy. The current
implementation applies these retain and forget terms together within each
update step, which is simpler and noticeably more stable than separating the
two phases too aggressively.

This implementation is an adapted repository baseline, not a claim of exact
paper reproduction. The goal is to preserve SCRUB's main retain-versus-forget
teacher-student idea while integrating cleanly with the existing data splits,
evaluation code, checkpoints, plots, and comparison notebook. It should be
described as an approximate unlearning baseline rather than a method with exact
removal guarantees.
