"""Last-layer certified data removal (Guo et al., 2020).

ResNet-50 is non-convex, so we freeze the feature extractor and only certify
the final linear classifier, which is a convex softmax regression problem.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..training import evaluate
from .common import _save_snapshot


@dataclass(frozen=True)
class CertifiedConfig:
    """Settings for last-layer certified removal (refit + influence correction + optional noise)."""

    l2_reg: float = 1e-3
    feature_clip: float = 1.0
    refit_max_iter: int = 200
    refit_tol: float = 1e-8
    cg_max_iter: int = 200
    cg_tol: float = 1e-8
    noise_sigma: float = 0.0
    epsilon: Optional[float] = None
    delta: float = 1e-5
    feature_batch_size: int = 512


def _get_linear_head(model: nn.Module) -> nn.Linear:
    """Return the model's final linear layer (``model.model.fc``)."""
    inner = getattr(model, "model", model)
    head = getattr(inner, "fc", None)
    if not isinstance(head, nn.Linear):
        raise TypeError(
            "Certified unlearning expects ``model.model.fc`` to be an nn.Linear; "
            f"got {type(head).__name__}."
        )
    return head


def _extract_penultimate_features(
    model: nn.Module,
    loader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward inputs through the backbone and collect penultimate features."""
    inner = getattr(model, "model", model)
    original_fc = inner.fc
    inner.fc = nn.Identity()
    model.eval()

    all_feats: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    try:
        with torch.inference_mode():
            for inputs, labels in loader:
                inputs = inputs.to(device, non_blocking=True)
                feats = model(inputs).float()
                all_feats.append(feats.detach().cpu())
                all_labels.append(labels.detach().cpu().long())
    finally:
        inner.fc = original_fc

    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)


def _clip_feature_norms(features: torch.Tensor, feature_clip: float) -> torch.Tensor:
    """Rescale each row so its L2 norm is at most ``feature_clip``."""
    if feature_clip <= 0.0:
        raise ValueError(f"feature_clip must be positive, got {feature_clip}")
    norms = features.norm(dim=1, keepdim=True).clamp(min=1e-12)
    scale = torch.where(norms > feature_clip, feature_clip / norms, torch.ones_like(norms))
    return features * scale


def _head_loss(
    W: torch.Tensor,
    b: torch.Tensor,
    features: torch.Tensor,
    labels: torch.Tensor,
    l2_reg: float,
) -> torch.Tensor:
    logits = features @ W.t() + b
    ce = F.cross_entropy(logits, labels, reduction="mean")
    reg = 0.5 * l2_reg * (W.pow(2).sum() + b.pow(2).sum())
    return ce + reg


def _head_sum_loss(
    W: torch.Tensor,
    b: torch.Tensor,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    logits = features @ W.t() + b
    return F.cross_entropy(logits, labels, reduction="sum")


def _refit_head(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    l2_reg: float,
    max_iter: int,
    tol: float,
    device: torch.device,
    init_W: Optional[torch.Tensor] = None,
    init_b: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    features = features.to(device)
    labels = labels.to(device)
    feature_dim = features.shape[1]

    W = (
        init_W.detach().to(device=device, dtype=torch.float32).clone()
        if init_W is not None
        else torch.zeros(num_classes, feature_dim, device=device, dtype=torch.float32)
    )
    b = (
        init_b.detach().to(device=device, dtype=torch.float32).clone()
        if init_b is not None
        else torch.zeros(num_classes, device=device, dtype=torch.float32)
    )
    W.requires_grad_(True)
    b.requires_grad_(True)

    optimizer = optim.LBFGS(
        [W, b],
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=tol,
        tolerance_change=tol,
        history_size=30,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = _head_loss(W, b, features, labels, l2_reg)
        loss.backward()
        return loss

    optimizer.step(closure)
    return W.detach(), b.detach()


def _sum_gradient_on_subset(
    W: torch.Tensor,
    b: torch.Tensor,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    W_v = W.detach().clone().requires_grad_(True)
    b_v = b.detach().clone().requires_grad_(True)
    loss = _head_sum_loss(W_v, b_v, features, labels)
    gW, gb = torch.autograd.grad(loss, [W_v, b_v])
    return gW.detach(), gb.detach()


def _retain_hessian_vector_product(
    W: torch.Tensor,
    b: torch.Tensor,
    features_retain: torch.Tensor,
    labels_retain: torch.Tensor,
    l2_reg: float,
    vW: torch.Tensor,
    vb: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    W_v = W.detach().clone().requires_grad_(True)
    b_v = b.detach().clone().requires_grad_(True)
    loss = _head_loss(W_v, b_v, features_retain, labels_retain, l2_reg)
    gW, gb = torch.autograd.grad(loss, [W_v, b_v], create_graph=True)
    dot = (gW * vW).sum() + (gb * vb).sum()
    hW, hb = torch.autograd.grad(dot, [W_v, b_v])
    return hW.detach(), hb.detach()


def _conjugate_gradient(
    matvec: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    rhs_W: torch.Tensor,
    rhs_b: torch.Tensor,
    max_iter: int,
    tol: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve ``H x = rhs`` via conjugate gradients (``H`` is SPD; only the matvec is needed)."""
    x_W = torch.zeros_like(rhs_W)
    x_b = torch.zeros_like(rhs_b)
    r_W = rhs_W.clone()
    r_b = rhs_b.clone()
    p_W = r_W.clone()
    p_b = r_b.clone()
    rs_old = (r_W.pow(2).sum() + r_b.pow(2).sum()).item()
    initial_rs = max(rs_old, 1e-30)

    for _ in range(max_iter):
        Hp_W, Hp_b = matvec(p_W, p_b)
        denom = (p_W * Hp_W).sum() + (p_b * Hp_b).sum()
        denom_val = float(denom.item())
        if not math.isfinite(denom_val) or abs(denom_val) < 1e-30:
            break
        alpha = rs_old / denom_val
        x_W = x_W + alpha * p_W
        x_b = x_b + alpha * p_b
        r_W = r_W - alpha * Hp_W
        r_b = r_b - alpha * Hp_b
        rs_new = (r_W.pow(2).sum() + r_b.pow(2).sum()).item()
        if rs_new / initial_rs < tol ** 2:
            break
        beta = rs_new / rs_old
        p_W = r_W + beta * p_W
        p_b = r_b + beta * p_b
        rs_old = rs_new

    return x_W, x_b


def _install_head(model: nn.Module, W: torch.Tensor, b: torch.Tensor) -> None:
    head = _get_linear_head(model)
    with torch.no_grad():
        head.weight.data.copy_(W.to(device=head.weight.device, dtype=head.weight.dtype))
        head.bias.data.copy_(b.to(device=head.bias.device, dtype=head.bias.dtype))


def _calibrated_noise_sigma(
    residual_norm: float,
    l2_reg: float,
    n_retain: int,
    epsilon: float,
    delta: float,
) -> float:
    if epsilon <= 0.0 or delta <= 0.0 or delta >= 1.0:
        raise ValueError(f"Require epsilon > 0 and 0 < delta < 1; got eps={epsilon}, delta={delta}")
    c = math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
    denom = max(l2_reg * float(n_retain), 1e-30)
    return c * float(residual_norm) / denom


def run_certified_unlearning(
    model: nn.Module,
    forget_loader,
    retain_loader,
    testloader,
    device: torch.device,
    config: Optional[CertifiedConfig] = None,
    num_classes: int = 10,
    snapshot_dir: Optional[str] = None,
) -> dict:
    """Run last-layer certified removal: refit on ``D``, then apply an influence correction."""
    config = config or CertifiedConfig()
    print("[Certified] starting unlearning (last-layer certified removal)")

    history: List[np.ndarray] = []
    snapshot_paths: List[str] = []

    _, per_class = evaluate(model, testloader, num_classes=num_classes, device=device)
    history.append(per_class)
    path = _save_snapshot(model, snapshot_dir, 0)
    if path is not None:
        snapshot_paths.append(path)

    forget_feats, forget_labels = _extract_penultimate_features(model, forget_loader, device)
    retain_feats, retain_labels = _extract_penultimate_features(model, retain_loader, device)
    print("[Certified] extracted forget/retain penultimate features")

    if config.feature_clip > 0.0:
        forget_feats = _clip_feature_norms(forget_feats, config.feature_clip)
        retain_feats = _clip_feature_norms(retain_feats, config.feature_clip)

    head = _get_linear_head(model)
    W_init = head.weight.detach().float().clone()
    b_init = head.bias.detach().float().clone()

    all_feats = torch.cat([forget_feats, retain_feats], dim=0)
    all_labels = torch.cat([forget_labels, retain_labels], dim=0)

    W_star, b_star = _refit_head(
        all_feats,
        all_labels,
        num_classes=num_classes,
        l2_reg=config.l2_reg,
        max_iter=config.refit_max_iter,
        tol=config.refit_tol,
        device=device,
        init_W=W_init,
        init_b=b_init,
    )
    _install_head(model, W_star, b_star)
    print("[Certified] refit complete; solving influence-system correction")

    _, per_class = evaluate(model, testloader, num_classes=num_classes, device=device)
    history.append(per_class)
    path = _save_snapshot(model, snapshot_dir, 1)
    if path is not None:
        snapshot_paths.append(path)

    forget_feats_dev = forget_feats.to(device)
    forget_labels_dev = forget_labels.to(device)
    retain_feats_dev = retain_feats.to(device)
    retain_labels_dev = retain_labels.to(device)

    n_forget = int(forget_feats.shape[0])
    n_retain = int(retain_feats.shape[0])

    sum_grad_W_f, sum_grad_b_f = _sum_gradient_on_subset(W_star, b_star, forget_feats_dev, forget_labels_dev)
    rhs_W = (config.l2_reg * float(n_forget) * W_star + sum_grad_W_f) / float(n_retain)
    rhs_b = (config.l2_reg * float(n_forget) * b_star + sum_grad_b_f) / float(n_retain)

    def matvec(vW: torch.Tensor, vb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return _retain_hessian_vector_product(
            W_star,
            b_star,
            retain_feats_dev,
            retain_labels_dev,
            config.l2_reg,
            vW,
            vb,
        )

    delta_W, delta_b = _conjugate_gradient(
        matvec,
        rhs_W,
        rhs_b,
        max_iter=config.cg_max_iter,
        tol=config.cg_tol,
    )

    W_unlearn = W_star + delta_W
    b_unlearn = b_star + delta_b

    sigma = float(config.noise_sigma)
    if config.epsilon is not None:
        residual_W, residual_b = _sum_gradient_on_subset(
            W_unlearn, b_unlearn, retain_feats_dev, retain_labels_dev
        )
        residual_W = residual_W / float(n_retain) + config.l2_reg * W_unlearn
        residual_b = residual_b / float(n_retain) + config.l2_reg * b_unlearn
        residual_norm = float(torch.sqrt(residual_W.pow(2).sum() + residual_b.pow(2).sum()).item())
        sigma_calibrated = _calibrated_noise_sigma(
            residual_norm,
            config.l2_reg,
            n_retain,
            epsilon=config.epsilon,
            delta=config.delta,
        )
        sigma = max(sigma, sigma_calibrated)

    if sigma > 0.0:
        W_unlearn = W_unlearn + sigma * torch.randn_like(W_unlearn)
        b_unlearn = b_unlearn + sigma * torch.randn_like(b_unlearn)

    _install_head(model, W_unlearn, b_unlearn)

    _, per_class = evaluate(model, testloader, num_classes=num_classes, device=device)
    history.append(per_class)
    path = _save_snapshot(model, snapshot_dir, 2)
    if path is not None:
        snapshot_paths.append(path)

    print("[Certified] unlearning complete")
    return {"model": model, "classwise_history": history, "snapshot_paths": snapshot_paths}


