from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def _save_snapshot(model: nn.Module, snapshot_dir: Optional[str], epoch: int) -> Optional[str]:
    if snapshot_dir is None:
        return None
    os.makedirs(snapshot_dir, exist_ok=True)
    path = os.path.join(snapshot_dir, f"epoch_{epoch:03d}.pt")
    torch.save(model.state_dict(), path)
    return path


def _infinite_loader(loader):
    """Endless iterator used to draw retain batches inside a forget-paced loop."""
    while True:
        for batch in loader:
            yield batch


def _set_bn_eval(model: nn.Module) -> None:
    """Put every BatchNorm layer into eval mode without touching other modules."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def _freeze_model(model: nn.Module) -> None:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)


def _distill_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    student_logits = student_logits.float()
    teacher_logits = teacher_logits.float()
    log_probs_student = F.log_softmax(student_logits / temperature, dim=1)
    probs_teacher = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(log_probs_student, probs_teacher, reduction="batchmean") * (temperature ** 2)


def _optimizer_step_with_optional_clip(
    loss: torch.Tensor,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    amp,
    parameters,
    grad_clip_norm: Optional[float],
) -> None:
    if amp.use_grad_scaler:
        scaler.scale(loss).backward()
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=grad_clip_norm)
        optimizer.step()


def _loader_dataset_size(loader) -> int:
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        raise AttributeError("Loader must expose a dataset.")
    return len(dataset)
