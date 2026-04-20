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
    """Pure gradient ascent on the forget set.

    Implements vanilla Gradient Ascent unlearning: a single SGD loop over the
    forget loader where each step moves against the cross-entropy gradient. No
    retain set is consulted. This is the definition used in e.g. Thudi et al.
    (2022) and the NegGrad baseline of Golatkar et al. (2020); there is no
    regularisation, no retain-side loss and no masking.
    """

    lr: float = 5e-6
    epochs: int = 10
    max_batches_per_epoch: Optional[int] = 6


@dataclass(frozen=True)
class SSDConfig:
    """Selective Synaptic Dampening (Foster, Schoepf & Brintrup, 2023).

    One-shot post-hoc dampening: for each parameter θ_i whose forget-set Fisher
    dominates the reference Fisher (``I_forget(θ_i) > α · I_ref(θ_i)``), apply
    ``θ_i ← β_i · θ_i`` with ``β_i = min(1, λ · I_ref(θ_i) / I_forget(θ_i))``.
    The paper performs this once — there is no fine-tuning step afterwards, so
    the returned trajectory contains exactly two steps (pre- and post-
    dampening).

    ``selection_basis`` chooses what ``I_ref`` is:

    * ``"retain"`` (default) computes the reference Fisher on ``D_r`` only.
      This is the canonical interpretation of "weights that are
      disproportionately important for the forget set vs. everything else we
      want to keep" and, crucially, makes ``alpha`` interpretable independently
      of the forget ratio. With the convex-combination form below, the
      selection condition ``I_f > α · (p I_f + (1-p) I_r)`` reduces to
      ``(1 - α p) I_f > α (1-p) I_r``; for any forget ratio ``p ≥ 1/α`` *no*
      parameter can ever be selected, which yields a no-op SSD pass.
    * ``"union"`` reproduces the Foster et al. reference implementation, where
      ``I_ref`` is the per-sample Fisher over ``D = D_f ∪ D_r``. This is a
      good approximation of ``I_retain`` when ``|D_f| ≪ |D|`` but is
      pathological at the 10% forget ratios used here.
    """

    alpha: float = 1.0
    lambda_: float = 1.0
    eps: float = 1e-12
    fisher_batches: int = 100
    selection_basis: str = "retain"


@dataclass(frozen=True)
class SalUnConfig:
    """SalUn: saliency-masked random-labeling unlearning (Fan et al., 2024).

    Matches Algorithm 1 of the paper: compute the weight-saliency mask ``m_S``
    from the forget-set gradient magnitudes at the pre-unlearning checkpoint,
    then minimise ``CE(f_θ(x_f), y'_f) + CE(f_θ(x_r), y_r)`` with gradients
    restricted to ``m_S``, where ``y'_f`` is a random label distinct from the
    true forget-set label. Both the forget and retain terms are part of the
    published objective.

    ``weight_decay`` defaults to 0 because PyTorch's ``SGD`` adds
    ``weight_decay * param`` to the gradient *inside* ``step()``, i.e. after
    we have zeroed the gradient on non-salient weights; any non-zero value
    would therefore leak through the saliency mask and violate the
    ``θ ← θ - η · m_S ⊙ ∇L`` update rule. Regularization, if desired, must
    be folded into the loss explicitly.
    """

    lr: float = 1e-2
    epochs: int = 10
    mask_ratio: float = 0.5
    mask_batches: int = 10
    max_batches_per_epoch: Optional[int] = None
    num_classes: int = 10
    momentum: float = 0.9
    weight_decay: float = 0.0


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
        grads = torch.autograd.grad(
            loss,
            [param for _, param in params_named],
            retain_graph=False,
            create_graph=False,
        )
        for (name, _), grad in zip(params_named, grads):
            fisher[name] += grad.detach() ** 2
        n_batches += 1

    if n_batches == 0:
        raise RuntimeError("No batches processed for Fisher estimation.")
    for name in fisher:
        fisher[name] /= float(n_batches)
    return fisher


def _loader_dataset_size(loader) -> int:
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        raise AttributeError("Loader must expose a dataset.")
    return len(dataset)


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
    selection_basis: str = "retain",
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Return ``(I_forget, I_ref)`` where ``I_ref`` is the reference Fisher.

    See :class:`SSDConfig` for what ``selection_basis`` controls.
    """
    max_forget_batches = min(fisher_batches, len(forget_loader))
    max_retain_batches = min(fisher_batches, len(retain_loader))
    fisher_forget = estimate_empirical_fisher_diag(model, forget_loader, criterion, device, max_forget_batches)
    fisher_retain = estimate_empirical_fisher_diag(model, retain_loader, criterion, device, max_retain_batches)

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
    """Apply the SSD selection-and-dampening rule from Foster et al. (2023).

    Selection: ``I_forget(θ_i) > α · I_ref(θ_i)``.
    Update:    ``θ_i ← min(1, λ · I_ref(θ_i) / I_forget(θ_i)) · θ_i`` on selected.
    """
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


def estimate_salun_importance(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int,
) -> Dict[str, torch.Tensor]:
    """Accumulate |∇_θ L(x, y; θ_o)| over the forget set for saliency scoring."""
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
    """Sample a label in ``[0, num_classes)`` distinct from each ``labels[i]``."""
    offsets = torch.randint(
        1,
        num_classes,
        labels.shape,
        device=labels.device,
        generator=generator,
    )
    return (labels + offsets) % num_classes


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
                logits = model(inputs)
                loss = criterion(logits, labels)

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
    """One-shot Selective Synaptic Dampening as specified in Foster et al. (2023).

    The returned ``classwise_history`` always contains exactly two entries —
    the model before dampening and the model after dampening — because SSD is
    a single algebraic intervention on the weights rather than an iterative
    procedure.
    """
    config = config or SSDConfig()
    criterion = nn.CrossEntropyLoss()

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
        selection_basis=config.selection_basis,
    )
    _apply_ssd_dampening(model, fisher_forget, fisher_ref, config)

    _, per_class = evaluate(model, testloader, num_classes=num_classes, device=device)
    history.append(per_class)
    path = _save_snapshot(model, snapshot_dir, 1)
    if path is not None:
        snapshot_paths.append(path)

    return {"model": model, "classwise_history": history, "snapshot_paths": snapshot_paths}


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
    """SalUn unlearning per Algorithm 1 of Fan et al. (2024).

    1. Build weight saliency mask from forget-set gradient magnitudes at θ_o.
    2. For each epoch, minimise ``CE(f_θ(x_f), y'_f) + CE(f_θ(x_r), y_r)`` with
       ``y'_f`` random wrong labels, masking gradients to the salient weights.
    """
    config = config or SalUnConfig()
    if retain_loader is None:
        raise ValueError("SalUn requires a retain_loader; see Fan et al. (2024) Algorithm 1.")

    amp = build_amp_config(device)
    criterion = nn.CrossEntropyLoss()
    # Weight decay is deliberately applied *inside* the loss (below) rather
    # than via the optimizer, so its contribution to the gradient is also
    # restricted to the saliency mask. Passing weight_decay to torch's SGD
    # would add wd*param *after* we mask the gradient and therefore leak an
    # update onto non-salient weights.
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=0.0,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp.use_grad_scaler)

    max_mask_batches = min(config.mask_batches, len(forget_loader))
    salun_mask = build_salun_mask(
        estimate_salun_importance(model, forget_loader, criterion, device, max_mask_batches),
        config.mask_ratio,
    )

    retain_iter = _infinite_loader(retain_loader)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    wd = float(config.weight_decay)

    rng = torch.Generator(device=device)
    rng.manual_seed(int(torch.initial_seed()) & 0xFFFFFFFF)

    history: List[np.ndarray] = []
    snapshot_paths: List[str] = []

    initial = _save_snapshot(model, snapshot_dir, 0)
    if initial is not None:
        snapshot_paths.append(initial)
    _, per_class = evaluate(model, testloader, num_classes=num_classes, device=device)
    history.append(per_class)

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
                loss = criterion(f_logits, random_labels) + criterion(r_logits, r_labels)
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
    retain_loader,
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
