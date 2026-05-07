from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..training import evaluate
from .common import _loader_dataset_size, _save_snapshot


@dataclass(frozen=True)
class SSDConfig:
    """Selective Synaptic Dampening (Foster, Schoepf & Brintrup, 2023).

    One-shot dampening: for each weight whose forget-set Fisher dominates the
    reference Fisher (``I_forget > alpha * I_ref``), shrink it by
    ``beta = min(1, lambda * I_ref / I_forget)``. There is no fine-tuning, so
    the trajectory has just two steps (before and after dampening).

    ``selection_basis`` decides what ``I_ref`` is:

    * ``"retain"`` (default): Fisher on the retain set only. Makes ``alpha``
      interpretable regardless of the forget/retain split.
    * ``"union"``: Fisher on the full dataset, matching the reference
      implementation. Good when ``|D_f| << |D|``, but breaks down at the ~10%
      forget ratios used here.
    """

    alpha: float = 10.0
    lambda_: float = 1.0
    eps: float = 1e-12
    fisher_batches: int = 100
    fisher_samples_per_batch: int = 32
    selection_basis: str = "retain"
    fisher_mode: str = "batch"


def estimate_empirical_fisher_diag(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int,
    samples_per_batch: Optional[int] = None,
    fisher_mode: str = "batch",
) -> Dict[str, torch.Tensor]:
    params_named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    fisher = {name: torch.zeros_like(param, device=device) for name, param in params_named}

    model.eval()
    n_accum_steps = 0
    for batch_idx, (inputs, labels) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if fisher_mode == "batch":
            model.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            for name, param in params_named:
                if param.grad is not None:
                    fisher[name] += param.grad.detach() ** 2
            n_accum_steps += 1
        elif fisher_mode == "per_sample":
            logits = model(inputs)
            losses = F.cross_entropy(logits, labels, reduction="none")

            batch_size = int(labels.shape[0])
            if samples_per_batch is None or samples_per_batch >= batch_size:
                selected_indices = range(batch_size)
            else:
                selected_indices = torch.randperm(batch_size, device=labels.device)[:samples_per_batch].tolist()

            selected_count = len(selected_indices)
            for sample_pos, sample_idx in enumerate(selected_indices):
                grads = torch.autograd.grad(
                    losses[sample_idx],
                    [param for _, param in params_named],
                    retain_graph=sample_pos < selected_count - 1,
                    create_graph=False,
                )
                for (name, _), grad in zip(params_named, grads):
                    fisher[name] += grad.detach() ** 2
                n_accum_steps += 1
        else:
            raise ValueError(f"Unknown fisher_mode '{fisher_mode}'; expected 'batch' or 'per_sample'.")

    if n_accum_steps == 0:
        raise RuntimeError("No batches processed for Fisher estimation.")
    for name in fisher:
        fisher[name] /= float(n_accum_steps)
    return fisher


def _combine_fishers(
    fisher_forget: Dict[str, torch.Tensor],
    fisher_retain: Dict[str, torch.Tensor],
    forget_weight: float,
    retain_weight: float,
) -> Dict[str, torch.Tensor]:
    return {
        name: forget_weight * fisher_forget[name] + retain_weight * fisher_retain[name]
        for name in fisher_forget
    }


def _estimate_ssd_fishers(
    model: nn.Module,
    forget_loader,
    retain_loader,
    criterion: nn.Module,
    device: torch.device,
    fisher_batches: int,
    fisher_samples_per_batch: Optional[int] = None,
    selection_basis: str = "retain",
    fisher_mode: str = "batch",
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Estimate ``(I_forget, I_ref)`` according to ``selection_basis``."""
    max_forget_batches = min(fisher_batches, len(forget_loader))
    max_retain_batches = min(fisher_batches, len(retain_loader))
    fisher_forget = estimate_empirical_fisher_diag(
        model,
        forget_loader,
        criterion,
        device,
        max_forget_batches,
        samples_per_batch=fisher_samples_per_batch,
        fisher_mode=fisher_mode,
    )
    fisher_retain = estimate_empirical_fisher_diag(
        model,
        retain_loader,
        criterion,
        device,
        max_retain_batches,
        samples_per_batch=fisher_samples_per_batch,
        fisher_mode=fisher_mode,
    )

    if selection_basis == "retain":
        return fisher_forget, fisher_retain
    if selection_basis == "union":
        forget_size = _loader_dataset_size(forget_loader)
        retain_size = _loader_dataset_size(retain_loader)
        total_size = forget_size + retain_size
        fisher_full = _combine_fishers(
            fisher_forget,
            fisher_retain,
            forget_weight=float(forget_size) / float(total_size),
            retain_weight=float(retain_size) / float(total_size),
        )
        return fisher_forget, fisher_full
    raise ValueError(
        f"Unknown SSD selection_basis '{selection_basis}'; expected 'retain' or 'union'."
    )


def _apply_ssd_dampening(
    model: nn.Module,
    fisher_forget: Dict[str, torch.Tensor],
    fisher_ref: Dict[str, torch.Tensor],
    config: SSDConfig,
) -> Dict[str, float]:
    """Apply the SSD selection rule and shrink the chosen weights."""
    total_params = 0
    changed_params = 0
    mean_damp_sum = 0.0
    mean_ratio_sum = 0.0
    n_tensors = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            ff = fisher_forget[name]
            fr = fisher_ref[name]
            selected = ff > config.alpha * fr
            beta = torch.ones_like(param)
            if selected.any():
                scale = config.lambda_ * fr[selected] / (ff[selected] + config.eps)
                beta[selected] = torch.clamp(scale, max=1.0)
            param.mul_(beta)

            total_params += beta.numel()
            changed_params += int(selected.sum().item())
            mean_damp_sum += float(beta.mean().item())
            ratio = ff / (fr + config.eps)
            mean_ratio_sum += float(ratio.mean().item())
            n_tensors += 1

    if n_tensors == 0:
        return {"mean_damp": 1.0, "selected_fraction": 0.0, "mean_ratio": 0.0}
    return {
        "mean_damp": mean_damp_sum / float(n_tensors),
        "selected_fraction": changed_params / float(total_params) if total_params else 0.0,
        "mean_ratio": mean_ratio_sum / float(n_tensors),
    }


def run_ssd_unlearning(
    model: nn.Module,
    forget_loader,
    retain_loader,
    testloader,
    device: torch.device,
    config: Optional[SSDConfig] = None,
    num_classes: int = 10,
    snapshot_dir: Optional[str] = None,
):
    """Run SSD: estimate Fishers, then apply the dampening rule once."""
    config = config or SSDConfig()
    criterion = nn.CrossEntropyLoss()
    print("[SSD] starting unlearning (one-shot Fisher dampening)")

    history: List[np.ndarray] = []
    snapshot_paths: List[str] = []

    initial = _save_snapshot(model, snapshot_dir, 0)
    if initial is not None:
        snapshot_paths.append(initial)
    _, per_class = evaluate(model, testloader, num_classes=num_classes, device=device)
    history.append(per_class)

    fisher_forget, fisher_ref = _estimate_ssd_fishers(
        model,
        forget_loader,
        retain_loader,
        criterion,
        device,
        config.fisher_batches,
        fisher_samples_per_batch=config.fisher_samples_per_batch,
        selection_basis=config.selection_basis,
        fisher_mode=config.fisher_mode,
    )
    print("[SSD] Fisher estimation complete; applying dampening")
    diagnostics = _apply_ssd_dampening(model, fisher_forget, fisher_ref, config)

    _, per_class = evaluate(model, testloader, num_classes=num_classes, device=device)
    history.append(per_class)
    path = _save_snapshot(model, snapshot_dir, 1)
    if path is not None:
        snapshot_paths.append(path)

    print("[SSD] unlearning complete")
    return {
        "model": model,
        "classwise_history": history,
        "snapshot_paths": snapshot_paths,
        "diagnostics": diagnostics,
    }


