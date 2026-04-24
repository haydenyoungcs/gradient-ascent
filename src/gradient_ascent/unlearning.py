from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    ``freeze_bn`` keeps every BatchNorm layer in eval mode throughout GA so
    ``running_mean`` / ``running_var`` are not EMA-updated from forget-only
    batches. With PyTorch's default BN momentum 0.1 and O(10²) steps on a
    single-class forget set the original buffers would otherwise be
    essentially overwritten by forget-class statistics, and subsequent
    ``evaluate()`` calls (which use eval-mode BN) would then read corrupted
    stats on *every* class. Weights still receive gradient updates — only the
    running buffers are frozen — so this is a measurement fix, not a change
    to the GA update rule itself.

    ``grad_clip_norm`` bounds the per-step update. Vanilla ``-CE`` is
    unbounded above, so once the model mis-classifies the forget class
    confidently the gradient norm grows without limit and GA diverges. A
    global L2 clip preserves the gradient ascent direction while preventing
    runaway steps; set to ``None`` to recover the unclipped Thudi et al.
    update.
    """

    lr: float = 3e-6
    epochs: int = 10
    max_batches_per_epoch: Optional[int] = 6
    freeze_bn: bool = True
    grad_clip_norm: Optional[float] = 1.0


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

    alpha: float = 10.0
    lambda_: float = 1.0
    eps: float = 1e-12
    fisher_batches: int = 100
    fisher_samples_per_batch: int = 32
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

    ``retain_weight`` (β) scales the retain-side cross-entropy term:
    ``L = CE(f_θ(x_f), y'_f) + β · CE(f_θ(x_r), y_r)``. Algorithm 1 in the
    paper assumes joint sampling from ``D_f ∪ D_r`` so the per-step ratio of
    forget to retain examples matches their dataset proportions. Our loop
    iterates the forget loader as the outer loop and draws one retain batch
    per forget batch (1:1), which over-weights the random-label term by a
    factor of ``|D_r|/|D_f|`` relative to the joint-sampling regime. Setting
    ``retain_weight=None`` (default) lets ``run_salun_unlearning`` measure
    that ratio at runtime and apply it as ``β``; pass an explicit float to
    override (use ``1.0`` to recover the literal Algorithm 1 schedule).
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


@dataclass(frozen=True)
class SCRUBConfig:
    """SCRUB: teacher-student approximate unlearning (Kurmanji et al., 2023).

    This repository uses a clean adaptation of SCRUB's central idea rather than
    a bit-for-bit recreation of the authors' notebook code. The original
    trained model is copied and frozen as the teacher, while a student
    initialised from the same checkpoint is updated to:

    * stay close to the teacher on ``D_r`` via temperature-scaled KL
      distillation plus optional retain-label cross-entropy; and
    * move away from the teacher on ``D_f`` via a negative KL term.

    To keep the implementation easy to compare with the existing baselines and
    easy to explain in a dissertation, we use a paired-batch objective within
    each epoch: every forget batch is matched with a retain batch and the
    update minimises

    ``alpha * KL(student_r || teacher_r) + gamma * CE(student_r, y_r)
      - beta * KL(student_f || teacher_f)``.

    This preserves SCRUB's retain-preservation / forget-divergence structure
    while avoiding the large oscillations that can appear if a whole forget pass
    is followed by a whole retain pass.
    """

    lr: float = 5e-4
    epochs: int = 5
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    temperature: float = 2.0
    weight_decay: float = 1e-4
    batch_size: Optional[int] = None
    grad_clip_norm: Optional[float] = 1.0
    freeze_bn: bool = True


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
    """Put every BatchNorm layer into eval mode without touching other modules.

    Called after ``model.train()`` so that weights still receive gradient
    updates but BN ``running_mean`` / ``running_var`` are not EMA-updated
    from the current batch.
    """
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
    scaler: torch.cuda.amp.GradScaler,
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


def estimate_empirical_fisher_diag(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int,
    samples_per_batch: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    params_named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    fisher = {name: torch.zeros_like(param, device=device) for name, param in params_named}

    model.eval()
    n_samples = 0
    for batch_idx, (inputs, labels) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(inputs)
        losses = F.cross_entropy(logits, labels, reduction="none")

        batch_size = int(labels.shape[0])
        if samples_per_batch is None or samples_per_batch >= batch_size:
            selected_indices = range(batch_size)
        else:
            selected_indices = range(samples_per_batch)

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
            n_samples += 1

    if n_samples == 0:
        raise RuntimeError("No batches processed for Fisher estimation.")
    for name in fisher:
        fisher[name] /= float(n_samples)
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
    fisher_samples_per_batch: Optional[int] = None,
    selection_basis: str = "retain",
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Return ``(I_forget, I_ref)`` where ``I_ref`` is the reference Fisher.

    See :class:`SSDConfig` for what ``selection_basis`` controls.
    """
    max_forget_batches = min(fisher_batches, len(forget_loader))
    max_retain_batches = min(fisher_batches, len(retain_loader))
    fisher_forget = estimate_empirical_fisher_diag(
        model,
        forget_loader,
        criterion,
        device,
        max_forget_batches,
        samples_per_batch=fisher_samples_per_batch,
    )
    fisher_retain = estimate_empirical_fisher_diag(
        model,
        retain_loader,
        criterion,
        device,
        max_retain_batches,
        samples_per_batch=fisher_samples_per_batch,
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

    # IMPORTANT: must be eval() — train() would EMA-update every BatchNorm
    # layer's running_mean / running_var with statistics from forget-only
    # batches, silently corrupting BN buffers before any later evaluate()
    # call. We only need autograd here, not BN's training-time behaviour.
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
                # Unscale before clipping so the clip threshold is in real-grad
                # units; scaler.step will then skip the unscale it would
                # otherwise perform internally.
                if config.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=config.grad_clip_norm
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                (-loss).backward()
                if config.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=config.grad_clip_norm
                    )
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
        fisher_samples_per_batch=config.fisher_samples_per_batch,
        selection_basis=config.selection_basis,
    )
    diagnostics = _apply_ssd_dampening(model, fisher_forget, fisher_ref, config)

    _, per_class = evaluate(model, testloader, num_classes=num_classes, device=device)
    history.append(per_class)
    path = _save_snapshot(model, snapshot_dir, 1)
    if path is not None:
        snapshot_paths.append(path)

    return {
        "model": model,
        "classwise_history": history,
        "snapshot_paths": snapshot_paths,
        "diagnostics": diagnostics,
    }


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

    history: List[np.ndarray] = []
    snapshot_paths: List[str] = []

    # Capture the *true* pre-unlearning state first, before any operation
    # that puts the model in train mode or could touch BN running buffers.
    # This guarantees step-0 accuracy matches the loaded checkpoint exactly,
    # so trajectories across algorithms share an identical starting point.
    initial = _save_snapshot(model, snapshot_dir, 0)
    if initial is not None:
        snapshot_paths.append(initial)
    _, per_class = evaluate(model, testloader, num_classes=num_classes, device=device)
    history.append(per_class)

    max_mask_batches = min(config.mask_batches, len(forget_loader))
    salun_mask = build_salun_mask(
        estimate_salun_importance(model, forget_loader, criterion, device, max_mask_batches),
        config.mask_ratio,
    )

    retain_iter = _infinite_loader(retain_loader)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    wd = float(config.weight_decay)

    # β balances retain-vs-forget per-step coverage. The default mirrors the
    # joint-sampling regime of Fan et al. (2024) Algorithm 1, where the ratio
    # of forget to retain examples per minibatch matches |D_f|:|D_r|. Our
    # outer-forget loop draws 1 retain batch per forget batch, so we scale
    # CE_retain by |D_r|/|D_f| to recover that balance.
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
        path = _save_snapshot(model, snapshot_dir, epoch)
        if path is not None:
            snapshot_paths.append(path)

    return {"model": model, "classwise_history": history, "snapshot_paths": snapshot_paths}


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
    """SCRUB teacher-student unlearning adapted to this repository.

    The student is updated while the teacher is frozen. Each optimisation step
    combines one forget batch and one retain batch:

    ``L = alpha * KL_T(retain) + gamma * CE(retain) - beta * KL_T(forget)``.

    This is a faithful implementation of SCRUB's retain-preservation /
    forget-divergence idea, adapted to the repository's per-epoch snapshot
    interface rather than the original authors' notebook training harness. In
    practice this paired-batch form is noticeably more stable than running an
    entire forget phase and then an entire retain phase.
    """
    config = config or SCRUBConfig()
    if retain_loader is None:
        raise ValueError("SCRUB requires a retain_loader.")
    if teacher_model is model:
        raise ValueError("teacher_model must be distinct from the student model.")

    amp = build_amp_config(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=amp.use_grad_scaler)
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

    retain_iter = _infinite_loader(retain_loader)

    for epoch in range(1, config.epochs + 1):
        model.train()
        if config.freeze_bn:
            _set_bn_eval(model)

        forget_kl_sum = 0.0
        forget_batches = 0
        retain_kl_sum = 0.0
        retain_ce_sum = 0.0
        retain_batches = 0

        for forget_inputs, _forget_labels in forget_loader:
            retain_inputs, retain_labels = next(retain_iter)
            forget_inputs = forget_inputs.to(device, non_blocking=True)
            retain_inputs = retain_inputs.to(device, non_blocking=True)
            retain_labels = retain_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=amp.dtype, enabled=amp.enabled):
                student_forget_logits = model(forget_inputs)
                student_retain_logits = model(retain_inputs)
                with torch.no_grad():
                    teacher_forget_logits = teacher(forget_inputs)
                    teacher_retain_logits = teacher(retain_inputs)

                forget_kl = _distill_kl(student_forget_logits, teacher_forget_logits, config.temperature)
                retain_kl = _distill_kl(student_retain_logits, teacher_retain_logits, config.temperature)
                retain_ce = criterion(student_retain_logits.float(), retain_labels)
                loss = config.alpha * retain_kl + config.gamma * retain_ce - config.beta * forget_kl

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
        path = _save_snapshot(model, snapshot_dir, epoch)
        if path is not None:
            snapshot_paths.append(path)

    return {
        "model": model,
        "classwise_history": history,
        "snapshot_paths": snapshot_paths,
        "diagnostics": diagnostics,
    }


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
