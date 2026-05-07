from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..training import build_amp_config, build_grad_scaler, evaluate
from .common import _infinite_loader, _loader_dataset_size, _save_snapshot


@dataclass(frozen=True)
class SalUnConfig:
    """SalUn: saliency-masked random-labelling unlearning (Fan et al., 2024).

    Builds a binary mask ``m_S`` of the most-salient weights from forget-set
    gradient magnitudes, then minimises
    ``CE(f(x_f), y'_f) + beta * CE(f(x_r), y_r)`` with updates restricted to
    ``m_S``. ``y'_f`` is a random wrong label.

    ``weight_decay`` defaults to 0: PyTorch's SGD adds it to the gradient
    *after* we zero it on non-salient weights, which would leak through the
    mask. Apply L2 regularisation in the loss instead.

    ``retain_weight`` (beta) compensates for our 1:1 forget-vs-retain pairing
    versus the paper's joint sampling. Default ``None`` rescales by
    ``|D_r|/|D_f|`` automatically; pass ``1.0`` to match Algorithm 1 literally.
    """

    lr: float = 5e-3
    epochs: int = 5
    mask_ratio: float = 0.5
    mask_batches: int = 10
    max_batches_per_epoch: Optional[int] = None
    num_classes: int = 10
    momentum: float = 0.9
    weight_decay: float = 0.0
    retain_weight: Optional[float] = None


def estimate_salun_importance(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int,
) -> Dict[str, torch.Tensor]:
    """Average ``|grad_theta L|`` over the forget set to score weight saliency."""
    amp = build_amp_config(device)
    params_named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    importance = {name: torch.zeros_like(param, device=device) for name, param in params_named}

    model.eval()
    n_batches = 0
    for batch_idx, (inputs, labels) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp.dtype, enabled=amp.enabled):
            logits = model(inputs)
            loss = criterion(logits, labels)

        grads = torch.autograd.grad(
            loss,
            [param for _, param in params_named],
            retain_graph=False,
            create_graph=False,
        )
        for (name, _), grad in zip(params_named, grads):
            importance[name] += grad.detach().abs()
        n_batches += 1

    if n_batches == 0:
        raise RuntimeError("No batches processed for SalUn mask estimation.")
    for name in importance:
        importance[name] /= float(n_batches)
    return importance


def build_salun_mask(importance: Dict[str, torch.Tensor], mask_ratio: float) -> Dict[str, torch.Tensor]:
    if mask_ratio <= 0.0 or mask_ratio > 1.0:
        raise ValueError(f"mask_ratio must be in (0, 1], got {mask_ratio}")

    flat_scores = torch.cat([vals.reshape(-1) for vals in importance.values()])
    k = max(1, int(mask_ratio * flat_scores.numel()))
    threshold = torch.topk(flat_scores, k=k, largest=True).values[-1]
    return {name: (vals >= threshold).to(vals.dtype) for name, vals in importance.items()}


def _random_wrong_labels(labels: torch.Tensor, num_classes: int, generator: torch.Generator) -> torch.Tensor:
    """Pick a random label per row that differs from the true label."""
    offsets = torch.randint(
        1,
        num_classes,
        labels.shape,
        device=labels.device,
        generator=generator,
    )
    return (labels + offsets) % num_classes


def run_salun_unlearning(
    model: nn.Module,
    forget_loader,
    retain_loader,
    testloader,
    device: torch.device,
    config: Optional[SalUnConfig] = None,
    num_classes: int = 10,
    snapshot_dir: Optional[str] = None,
):
    """Run SalUn unlearning: build a saliency mask, then masked SGD over forget+retain pairs."""
    config = config or SalUnConfig()
    if retain_loader is None:
        raise ValueError("SalUn requires a retain_loader.")

    amp = build_amp_config(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=0.0,
    )
    scaler = build_grad_scaler(device, enabled=amp.use_grad_scaler)

    history: List[np.ndarray] = []
    snapshot_paths: List[str] = []

    initial = _save_snapshot(model, snapshot_dir, 0)
    if initial is not None:
        snapshot_paths.append(initial)
    _, per_class = evaluate(model, testloader, num_classes=num_classes, device=device)
    history.append(per_class)
    print(f"[SalUn] starting unlearning ({config.epochs} epochs)")

    max_mask_batches = min(config.mask_batches, len(forget_loader))
    salun_mask = build_salun_mask(
        estimate_salun_importance(model, forget_loader, criterion, device, max_mask_batches),
        config.mask_ratio,
    )
    print(f"[SalUn] mask estimated ({max_mask_batches} batches, mask_ratio={config.mask_ratio:.2f})")

    retain_iter = _infinite_loader(retain_loader)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    wd = float(config.weight_decay)

    if config.retain_weight is None:
        forget_size = _loader_dataset_size(forget_loader)
        retain_size = _loader_dataset_size(retain_loader)
        retain_weight = float(retain_size) / float(max(forget_size, 1))
    else:
        retain_weight = float(config.retain_weight)

    rng = torch.Generator(device=device)
    rng.manual_seed(int(torch.initial_seed()) & 0xFFFFFFFF)

    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch_idx, (f_inputs, f_labels) in enumerate(forget_loader):
            if config.max_batches_per_epoch is not None and batch_idx >= config.max_batches_per_epoch:
                break
            f_inputs = f_inputs.to(device, non_blocking=True)
            f_labels = f_labels.to(device, non_blocking=True)
            random_labels = _random_wrong_labels(f_labels, config.num_classes, rng)

            r_inputs, r_labels = next(retain_iter)
            r_inputs = r_inputs.to(device, non_blocking=True)
            r_labels = r_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=amp.dtype, enabled=amp.enabled):
                f_logits = model(f_inputs)
                r_logits = model(r_inputs)
                loss = criterion(f_logits, random_labels) + retain_weight * criterion(r_logits, r_labels)
                if wd > 0.0:
                    l2 = sum((p * p).sum() for p in trainable_params)
                    loss = loss + 0.5 * wd * l2

            if amp.use_grad_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            for name, param in model.named_parameters():
                if param.grad is None or name not in salun_mask:
                    continue
                param.grad.mul_(salun_mask[name])

            if amp.use_grad_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        _, per_class = evaluate(model, testloader, num_classes=num_classes, device=device)
        history.append(per_class)
        print(f"[SalUn] epoch {epoch}/{config.epochs} complete")
        path = _save_snapshot(model, snapshot_dir, epoch)
        if path is not None:
            snapshot_paths.append(path)

    print("[SalUn] unlearning complete")
    return {"model": model, "classwise_history": history, "snapshot_paths": snapshot_paths}


