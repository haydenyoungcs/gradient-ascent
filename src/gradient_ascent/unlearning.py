from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .training import build_amp_config, evaluate


@dataclass(frozen=True)
class GAConfig:
    lr: float = 7e-6
    epochs: int = 10
    max_batches_per_epoch: Optional[int] = 6


@dataclass(frozen=True)
class SSDConfig:
    epochs: int = 9
    alpha: float = 1.8
    lambda_: float = 0.09
    eps: float = 1e-6
    min_scale: float = 0.65
    excess_cap: float = 8.0
    fisher_batches: int = 12


@dataclass(frozen=True)
class SalUnConfig:
    lr: float = 4e-6
    epochs: int = 10
    mask_ratio: float = 0.02
    mask_batches: int = 5
    max_batches_per_epoch: Optional[int] = 5


def _save_snapshot(model: nn.Module, snapshot_dir: Optional[str], epoch: int) -> Optional[str]:
    if snapshot_dir is None:
        return None
    os.makedirs(snapshot_dir, exist_ok=True)
    path = os.path.join(snapshot_dir, f"epoch_{epoch:03d}.pt")
    torch.save(model.state_dict(), path)
    return path


def estimate_empirical_fisher_diag(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int,
) -> Dict[str, torch.Tensor]:
    params_named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    fisher = {name: torch.zeros_like(param, device=device) for name, param in params_named}

    model.eval()
    n_batches = 0
    for batch_idx, (inputs, labels) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        model.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        grads = torch.autograd.grad(loss, [param for _, param in params_named], retain_graph=False, create_graph=False)
        for (name, _), grad in zip(params_named, grads):
            fisher[name] += grad.detach() ** 2
        n_batches += 1

    if n_batches == 0:
        raise RuntimeError("No batches processed for Fisher estimation.")
    for name in fisher:
        fisher[name] /= float(n_batches)
    return fisher


def estimate_salun_importance(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int,
) -> Dict[str, torch.Tensor]:
    amp = build_amp_config(device)
    params_named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    importance = {name: torch.zeros_like(param, device=device) for name, param in params_named}

    model.train()
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

        grads = torch.autograd.grad(loss, [param for _, param in params_named], retain_graph=False, create_graph=False)
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


def run_ga_unlearning(
    model: nn.Module,
    forget_loader,
    testloader,
    device: torch.device,
    config: Optional[GAConfig] = None,
    num_classes: int = 10,
    snapshot_dir: Optional[str] = None,
):
    config = config or GAConfig()
    amp = build_amp_config(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=amp.use_grad_scaler)

    history: List[np.ndarray] = []
    snapshot_paths: List[str] = []

    initial = _save_snapshot(model, snapshot_dir, 0)
    if initial is not None:
        snapshot_paths.append(initial)
    _, per_class = evaluate(model, testloader, num_classes=num_classes, device=device)
    history.append(per_class)

    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch_idx, (inputs, labels) in enumerate(forget_loader):
            if config.max_batches_per_epoch is not None and batch_idx >= config.max_batches_per_epoch:
                break
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=amp.dtype, enabled=amp.enabled):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            if amp.use_grad_scaler:
                scaler.scale(-loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                (-loss).backward()
                optimizer.step()

        _, per_class = evaluate(model, testloader, num_classes=num_classes, device=device)
        history.append(per_class)
        path = _save_snapshot(model, snapshot_dir, epoch)
        if path is not None:
            snapshot_paths.append(path)

    return {"model": model, "classwise_history": history, "snapshot_paths": snapshot_paths}


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
    config = config or SSDConfig()
    criterion = nn.CrossEntropyLoss()
    max_fisher_batches = min(config.fisher_batches, len(forget_loader), len(retain_loader))
    fisher_forget = estimate_empirical_fisher_diag(model, forget_loader, criterion, device, max_fisher_batches)
    fisher_retain = estimate_empirical_fisher_diag(model, retain_loader, criterion, device, max_fisher_batches)

    history: List[np.ndarray] = []
    snapshot_paths: List[str] = []

    initial = _save_snapshot(model, snapshot_dir, 0)
    if initial is not None:
        snapshot_paths.append(initial)
    _, per_class = evaluate(model, testloader, num_classes=num_classes, device=device)
    history.append(per_class)

    step_lambda = config.lambda_ / float(config.epochs)
    for epoch in range(1, config.epochs + 1):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                ratio = fisher_forget[name] / (fisher_retain[name] + config.eps)
                excess = torch.clamp(torch.relu(ratio - config.alpha), max=config.excess_cap)
                damp = 1.0 - step_lambda * excess
                damp = torch.clamp(damp, min=config.min_scale, max=1.0)
                param.mul_(damp)

        _, per_class = evaluate(model, testloader, num_classes=num_classes, device=device)
        history.append(per_class)
        path = _save_snapshot(model, snapshot_dir, epoch)
        if path is not None:
            snapshot_paths.append(path)

    return {"model": model, "classwise_history": history, "snapshot_paths": snapshot_paths}


def run_salun_unlearning(
    model: nn.Module,
    forget_loader,
    testloader,
    device: torch.device,
    config: Optional[SalUnConfig] = None,
    num_classes: int = 10,
    snapshot_dir: Optional[str] = None,
):
    config = config or SalUnConfig()
    amp = build_amp_config(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=amp.use_grad_scaler)

    max_mask_batches = min(config.mask_batches, len(forget_loader))
    salun_mask = build_salun_mask(
        estimate_salun_importance(model, forget_loader, criterion, device, max_mask_batches),
        config.mask_ratio,
    )

    history: List[np.ndarray] = []
    snapshot_paths: List[str] = []

    initial = _save_snapshot(model, snapshot_dir, 0)
    if initial is not None:
        snapshot_paths.append(initial)
    _, per_class = evaluate(model, testloader, num_classes=num_classes, device=device)
    history.append(per_class)

    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch_idx, (inputs, labels) in enumerate(forget_loader):
            if config.max_batches_per_epoch is not None and batch_idx >= config.max_batches_per_epoch:
                break
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=amp.dtype, enabled=amp.enabled):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            if amp.use_grad_scaler:
                scaler.scale(-loss).backward()
                scaler.unscale_(optimizer)
            else:
                (-loss).backward()

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
        path = _save_snapshot(model, snapshot_dir, epoch)
        if path is not None:
            snapshot_paths.append(path)

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


def run_ssd_snapshots(
    model_factory: Callable[[], nn.Module],
    original_checkpoint_path: str,
    forget_loader,
    retain_loader,
    testloader,
    device: torch.device,
    snapshot_dir: str,
    final_checkpoint_path: Optional[str] = None,
    config: Optional[SSDConfig] = None,
    num_classes: int = 10,
) -> str:
    model = model_factory()
    model.load_state_dict(torch.load(original_checkpoint_path, map_location=device))
    result = run_ssd_unlearning(
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


def run_salun_snapshots(
    model_factory: Callable[[], nn.Module],
    original_checkpoint_path: str,
    forget_loader,
    testloader,
    device: torch.device,
    snapshot_dir: str,
    final_checkpoint_path: Optional[str] = None,
    config: Optional[SalUnConfig] = None,
    num_classes: int = 10,
) -> str:
    model = model_factory()
    model.load_state_dict(torch.load(original_checkpoint_path, map_location=device))
    result = run_salun_unlearning(
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
