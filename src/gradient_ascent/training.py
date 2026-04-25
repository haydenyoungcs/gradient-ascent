from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass(frozen=True)
class AmpConfig:
    enabled: bool
    dtype: torch.dtype
    use_grad_scaler: bool


def configure_runtime() -> None:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def build_amp_config(device: torch.device) -> AmpConfig:
    enabled = device.type == "cuda"
    use_bf16 = enabled and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_grad_scaler = enabled and dtype == torch.float16
    return AmpConfig(enabled=enabled, dtype=dtype, use_grad_scaler=use_grad_scaler)


def build_grad_scaler(device: torch.device, enabled: bool) -> torch.amp.GradScaler:
    """Construct a GradScaler using the non-deprecated torch.amp API."""
    if device.type == "cuda":
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.amp.GradScaler("cpu", enabled=False)


@torch.inference_mode()
def evaluate(
    net: nn.Module,
    loader,
    num_classes: int = 10,
    device: torch.device | str = "cpu",
) -> Tuple[float, np.ndarray]:
    if isinstance(device, str):
        device = torch.device(device)

    amp = build_amp_config(device)
    net.eval()
    correct = 0
    total = 0
    correct_c = torch.zeros(num_classes, device=device)
    total_c = torch.zeros(num_classes, device=device)

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=amp.dtype, enabled=amp.enabled):
            outputs = net(inputs)
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.numel()

        for c in range(num_classes):
            mask = labels == c
            total_c[c] += mask.sum()
            correct_c[c] += (preds[mask] == labels[mask]).sum()

    overall_acc = correct / total if total > 0 else 0.0
    per_class_acc = (correct_c / torch.clamp(total_c, min=1)).detach().cpu().numpy()
    net.train()
    return overall_acc, per_class_acc


def train_model(
    net: nn.Module,
    trainloader,
    testloader,
    num_epochs: int,
    device: torch.device,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    save_path: Optional[str] = None,
    log_prefix: str = "",
    lr_milestones: Optional[List[int]] = None,
    lr_gamma: float = 0.1,
    num_classes: int = 10,
    wandb_run=None,
    wandb_prefix: Optional[str] = None,
    wandb_step_offset: int = 0,
) -> Tuple[List[float], List[float]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = None
    if lr_milestones is not None:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_gamma)

    amp = build_amp_config(device)
    scaler = build_grad_scaler(device, enabled=amp.use_grad_scaler)

    acc_history: List[float] = []
    time_history: List[float] = []

    for epoch in range(num_epochs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        epoch_loss = 0.0
        for inputs, labels in trainloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp.dtype, enabled=amp.enabled):
                outputs = net(inputs)
                loss = criterion(outputs, labels)

            if amp.use_grad_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        acc, _ = evaluate(net, testloader, num_classes=num_classes, device=device)
        acc_history.append(acc)

        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        time_history.append(dt)

        mean_loss = epoch_loss / len(trainloader)
        current_lr = optimizer.param_groups[0]["lr"]
        if wandb_run is not None and wandb_prefix is not None:
            wandb_run.log(
                {
                    f"{wandb_prefix}/train_loss": mean_loss,
                    f"{wandb_prefix}/test_acc": acc,
                    f"{wandb_prefix}/lr": current_lr,
                    f"{wandb_prefix}/epoch_time_s": dt,
                },
                step=wandb_step_offset + epoch,
            )

        print(
            f"{log_prefix}epoch {epoch + 1}/{num_epochs} loss {mean_loss:.3f} "
            f"test_acc {acc:.3f} lr {current_lr:.2e} epoch_s {dt:.2f}"
        )

    if save_path is not None:
        torch.save(net.state_dict(), save_path)
        print(f"Saved to {save_path}")

    return acc_history, time_history


def build_classwise_history(
    algorithm_key: str,
    history: Dict[str, List[np.ndarray]],
    per_class_acc: np.ndarray,
) -> Dict[str, List[np.ndarray]]:
    history.setdefault(algorithm_key, []).append(np.asarray(per_class_acc, dtype=np.float32))
    return history
