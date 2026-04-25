from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..training import build_amp_config, build_grad_scaler, evaluate
from .common import (
    _distill_kl,
    _freeze_model,
    _infinite_loader,
    _optimizer_step_with_optional_clip,
    _save_snapshot,
    _set_bn_eval,
)


@dataclass(frozen=True)
class SCRUBConfig:
    """SCRUB: teacher-student approximate unlearning (Kurmanji et al., 2023).

    This repository uses a clean adaptation of SCRUB's central idea rather than
    a bit-for-bit recreation of the authors' notebook code. The original
    trained model is copied and frozen as the teacher, while a student
    initialised from the same checkpoint is updated to:

    * stay close to the teacher on ``D_r`` via temperature-scaled KL
      distillation plus optional retain-label cross-entropy; and
    * move away from the teacher on ``D_f`` by making the student's logits
      deliberately uninformative on forget examples.

    To keep the implementation easy to compare with the existing baselines and
    easy to explain in a dissertation, we use a paired-batch objective during a
    short initial scrub phase: every forget batch is matched with a retain batch
    and the update minimises

    ``alpha * KL(student_r || teacher_r) + gamma * CE(student_r, y_r)
      + beta * KL(U || student_f)``.

    Here ``U`` is the uniform class distribution. Using a uniform forget target
    gives a more controlled "be uncertain on forgotten data" signal than a raw
    negative teacher-KL term, which in practice can make the model latch onto an
    arbitrary wrong class and produce erratic spikes in unrelated classes.

    After that, the student enters a recovery phase where the forget term is
    not removed entirely, but reduced sharply. This avoids the failure mode
    where a single destructive scrub step is immediately undone by pure
    retain-only training, while still keeping later epochs much more
    retain-focused than the initial scrub phase.
    """

    lr: float = 5e-4
    epochs: int = 5
    forget_phase_epochs: int = 2
    max_forget_batches_per_epoch: Optional[int] = None
    recovery_beta_scale: float = 0.0
    recovery_max_forget_batches_per_epoch: Optional[int] = None
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    temperature: float = 2.0
    weight_decay: float = 1e-4
    batch_size: Optional[int] = None
    grad_clip_norm: Optional[float] = 1.0
    freeze_bn: bool = True
    reset_optimizer_after_forget: bool = True


def _build_scrub_optimizer(model: nn.Module, config: SCRUBConfig):
    return optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)


def _uniform_kl(student_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    student_logits = student_logits.float()
    log_probs_student = torch.log_softmax(student_logits / temperature, dim=1)
    num_classes = int(student_logits.shape[1])
    uniform_probs = torch.full_like(student_logits, 1.0 / float(num_classes))
    # F.kl_div(log q, p) computes KL(p || q), so this is KL(U || student).
    return torch.nn.functional.kl_div(log_probs_student, uniform_probs, reduction="batchmean") * (temperature ** 2)


def run_scrub_unlearning(
    model: nn.Module,
    forget_loader,
    retain_loader,
    testloader,
    device: torch.device,
    config: Optional[SCRUBConfig] = None,
    num_classes: int = 10,
    snapshot_dir: Optional[str] = None,
    teacher_model: Optional[nn.Module] = None,
):
    """SCRUB teacher-student unlearning adapted to this repository."""
    config = config or SCRUBConfig()
    if retain_loader is None:
        raise ValueError("SCRUB requires a retain_loader.")
    if teacher_model is model:
        raise ValueError("teacher_model must be distinct from the student model.")
    if config.forget_phase_epochs < 0 or config.forget_phase_epochs > config.epochs:
        raise ValueError(
            f"forget_phase_epochs must lie in [0, epochs], got {config.forget_phase_epochs} for epochs={config.epochs}."
        )
    if config.recovery_beta_scale < 0.0 or config.recovery_beta_scale > 1.0:
        raise ValueError(
            f"recovery_beta_scale must lie in [0, 1], got {config.recovery_beta_scale}."
        )

    amp = build_amp_config(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _build_scrub_optimizer(model, config)
    scaler = build_grad_scaler(device, enabled=amp.use_grad_scaler)
    trainable_params = [param for param in model.parameters() if param.requires_grad]

    teacher = copy.deepcopy(model) if teacher_model is None else teacher_model
    teacher.to(device)
    _freeze_model(teacher)

    history: List[np.ndarray] = []
    snapshot_paths: List[str] = []
    diagnostics: List[Dict[str, float]] = []

    initial = _save_snapshot(model, snapshot_dir, 0)
    if initial is not None:
        snapshot_paths.append(initial)
    _, per_class = evaluate(model, testloader, num_classes=num_classes, device=device)
    history.append(per_class)
    print(
        "[SCRUB] starting unlearning "
        f"({config.epochs} epochs; forget_phase_epochs={config.forget_phase_epochs})"
    )

    retain_iter = _infinite_loader(retain_loader)

    for epoch in range(1, config.epochs + 1):
        in_forget_phase = epoch <= config.forget_phase_epochs
        if epoch == config.forget_phase_epochs + 1 and config.reset_optimizer_after_forget:
            optimizer = _build_scrub_optimizer(model, config)
            scaler = build_grad_scaler(device, enabled=amp.use_grad_scaler)
            print("[SCRUB] switched to recovery phase (optimizer and scaler reset)")

        model.train()
        if config.freeze_bn:
            _set_bn_eval(model)

        forget_kl_sum = 0.0
        forget_batches = 0
        retain_kl_sum = 0.0
        retain_ce_sum = 0.0
        retain_batches = 0

        epoch_beta_scale = 1.0 if in_forget_phase else config.recovery_beta_scale
        max_forget_batches = (
            config.max_forget_batches_per_epoch
            if in_forget_phase
            else config.recovery_max_forget_batches_per_epoch
        )

        if epoch_beta_scale > 0.0:
            for batch_idx, (forget_inputs, _forget_labels) in enumerate(forget_loader):
                if max_forget_batches is not None and batch_idx >= max_forget_batches:
                    break
                retain_inputs, retain_labels = next(retain_iter)
                forget_inputs = forget_inputs.to(device, non_blocking=True)
                retain_inputs = retain_inputs.to(device, non_blocking=True)
                retain_labels = retain_labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type="cuda", dtype=amp.dtype, enabled=amp.enabled):
                    student_forget_logits = model(forget_inputs)
                    student_retain_logits = model(retain_inputs)
                    with torch.no_grad():
                        teacher_retain_logits = teacher(retain_inputs)

                    forget_kl = _uniform_kl(student_forget_logits, config.temperature)
                    retain_kl = _distill_kl(student_retain_logits, teacher_retain_logits, config.temperature)
                    retain_ce = criterion(student_retain_logits.float(), retain_labels)
                    loss = (
                        config.alpha * retain_kl
                        + config.gamma * retain_ce
                        - (config.beta * epoch_beta_scale) * forget_kl
                    )

                if not torch.isfinite(loss.detach()):
                    raise RuntimeError(f"Non-finite SCRUB loss at epoch {epoch}.")
                _optimizer_step_with_optional_clip(
                    loss,
                    optimizer,
                    scaler,
                    amp,
                    trainable_params,
                    config.grad_clip_norm,
                )
                forget_kl_sum += float(forget_kl.detach().item())
                forget_batches += 1
                retain_kl_sum += float(retain_kl.detach().item())
                retain_ce_sum += float(retain_ce.detach().item())
                retain_batches += 1

        if not in_forget_phase:
            for retain_inputs, retain_labels in retain_loader:
                retain_inputs = retain_inputs.to(device, non_blocking=True)
                retain_labels = retain_labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type="cuda", dtype=amp.dtype, enabled=amp.enabled):
                    student_retain_logits = model(retain_inputs)
                    with torch.no_grad():
                        teacher_retain_logits = teacher(retain_inputs)

                    retain_kl = _distill_kl(student_retain_logits, teacher_retain_logits, config.temperature)
                    retain_ce = criterion(student_retain_logits.float(), retain_labels)
                    loss = config.alpha * retain_kl + config.gamma * retain_ce

                if not torch.isfinite(loss.detach()):
                    raise RuntimeError(f"Non-finite SCRUB loss at epoch {epoch}.")
                _optimizer_step_with_optional_clip(
                    loss,
                    optimizer,
                    scaler,
                    amp,
                    trainable_params,
                    config.grad_clip_norm,
                )
                retain_kl_sum += float(retain_kl.detach().item())
                retain_ce_sum += float(retain_ce.detach().item())
                retain_batches += 1

        diagnostics.append(
            {
                "epoch": float(epoch),
                "forget_kl": forget_kl_sum / float(max(forget_batches, 1)),
                "retain_kl": retain_kl_sum / float(max(retain_batches, 1)),
                "retain_ce": retain_ce_sum / float(max(retain_batches, 1)),
            }
        )

        _, per_class = evaluate(model, testloader, num_classes=num_classes, device=device)
        history.append(per_class)
        phase_name = "forget" if in_forget_phase else "recovery"
        print(
            f"[SCRUB] epoch {epoch}/{config.epochs} complete "
            f"(phase={phase_name}, forget_batches={forget_batches}, retain_batches={retain_batches})"
        )
        path = _save_snapshot(model, snapshot_dir, epoch)
        if path is not None:
            snapshot_paths.append(path)

    print("[SCRUB] unlearning complete")
    return {
        "model": model,
        "classwise_history": history,
        "snapshot_paths": snapshot_paths,
        "diagnostics": diagnostics,
    }


def run_scrub_snapshots(
    model_factory: Callable[[], nn.Module],
    original_checkpoint_path: str,
    forget_loader,
    retain_loader,
    testloader,
    device: torch.device,
    snapshot_dir: str,
    final_checkpoint_path: Optional[str] = None,
    config: Optional[SCRUBConfig] = None,
    num_classes: int = 10,
) -> str:
    model = model_factory()
    model.load_state_dict(torch.load(original_checkpoint_path, map_location=device))
    result = run_scrub_unlearning(
        model,
        forget_loader,
        retain_loader,
        testloader,
        device,
        config=config,
        num_classes=num_classes,
        snapshot_dir=snapshot_dir,
    )
    if final_checkpoint_path is not None:
        torch.save(result["model"].state_dict(), final_checkpoint_path)
    return snapshot_dir
