from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..training import build_amp_config, build_grad_scaler, evaluate
from .common import _save_snapshot, _set_bn_eval


@dataclass(frozen=True)
class GAConfig:
    """Pure gradient ascent on the forget set.

    Implements vanilla Gradient Ascent unlearning: a single SGD loop over the
    forget loader where each step moves against the cross-entropy gradient. No
    retain set is consulted. This is the definition used in e.g. Thudi et al.
    (2022) and the NegGrad baseline of Golatkar et al. (2020); there is no
    regularisation, no retain-side loss and no masking.
    """

    lr: float = 3e-6
    epochs: int = 10
    max_batches_per_epoch: Optional[int] = None
    freeze_bn: bool = False
    grad_clip_norm: Optional[float] = None


def run_ga_unlearning(
    model: nn.Module,
    forget_loader,
    testloader,
    device: torch.device,
    config: Optional[GAConfig] = None,
    num_classes: int = 10,
    snapshot_dir: Optional[str] = None,
):
    """Vanilla Gradient Ascent unlearning on the forget set only."""
    config = config or GAConfig()
    amp = build_amp_config(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.0)
    scaler = build_grad_scaler(device, enabled=amp.use_grad_scaler)

    history: List[np.ndarray] = []
    snapshot_paths: List[str] = []

    initial = _save_snapshot(model, snapshot_dir, 0)
    if initial is not None:
        snapshot_paths.append(initial)
    _, per_class = evaluate(model, testloader, num_classes=num_classes, device=device)
    history.append(per_class)
    print(f"[GA] starting unlearning ({config.epochs} epochs)")

    for epoch in range(1, config.epochs + 1):
        model.train()
        if config.freeze_bn:
            _set_bn_eval(model)
        for batch_idx, (inputs, labels) in enumerate(forget_loader):
            if config.max_batches_per_epoch is not None and batch_idx >= config.max_batches_per_epoch:
                break
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=amp.dtype, enabled=amp.enabled):
                logits = model(inputs)
                loss = criterion(logits, labels)

            if amp.use_grad_scaler:
                scaler.scale(-loss).backward()
                if config.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                (-loss).backward()
                if config.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
                optimizer.step()

        _, per_class = evaluate(model, testloader, num_classes=num_classes, device=device)
        history.append(per_class)
        print(f"[GA] epoch {epoch}/{config.epochs} complete")
        path = _save_snapshot(model, snapshot_dir, epoch)
        if path is not None:
            snapshot_paths.append(path)

    print("[GA] unlearning complete")
    return {"model": model, "classwise_history": history, "snapshot_paths": snapshot_paths}


def run_ga_snapshots(
    model_factory: Callable[[], nn.Module],
    original_checkpoint_path: str,
    forget_loader,
    testloader,
    device: torch.device,
    snapshot_dir: str,
    final_checkpoint_path: Optional[str] = None,
    config: Optional[GAConfig] = None,
    num_classes: int = 10,
) -> str:
    model = model_factory()
    model.load_state_dict(torch.load(original_checkpoint_path, map_location=device))
    result = run_ga_unlearning(
        model,
        forget_loader,
        testloader,
        device,
        config=config,
        num_classes=num_classes,
        snapshot_dir=snapshot_dir,
    )
    if final_checkpoint_path is not None:
        torch.save(result["model"].state_dict(), final_checkpoint_path)
    return snapshot_dir
