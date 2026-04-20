"""Last-layer certified data removal.

Implements the non-convex adaptation of certified removal from Guo, Goldstein,
Hannun & van der Maaten (2020), *Certified Data Removal from Machine Learning
Models* (ICML 2020). Because ResNet-50 is non-convex, we freeze the feature
extractor ``phi`` and treat the final linear classifier as a convex softmax
regression head. The head is then subjected to the certified-removal recipe:

1. Fit ``w* = argmin_w (1/N) sum_{i in D} CE(f(x_i; w), y_i) + (lambda/2)||w||^2``
   to convex optimality on the penultimate features.
2. Apply a single Newton step to approximate the retain-set optimum
   ``w** = argmin_w (1/N_r) sum_{i in D_r} CE + (lambda/2)||w||^2``::

       w' = w* + (1/N_r) * H_r(w*)^{-1} * (lambda * N_f * w* + sum_{i in D_f} grad l_i(w*))

   where ``H_r(w*) = (1/N_r) sum_{i in D_r} nabla^2 l_i(w*) + lambda I``.
3. Optionally add Gaussian noise ``eta ~ N(0, sigma^2 I)`` for
   ``(epsilon, delta)``-certified removal; ``sigma`` can be supplied directly
   or calibrated from ``epsilon``, ``delta`` and a bound on the residual
   gradient norm.

Three snapshots are produced (step 0 = loaded model, step 1 = post-refit, step
2 = post Newton update) so the plotting pipeline used for GA / SSD / SalUn
applies unchanged.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .training import evaluate


@dataclass(frozen=True)
class CertifiedConfig:
    """Hyperparameters for last-layer certified unlearning.

    Attributes:
        l2_reg: Ridge strength ``lambda`` on the head; a larger value makes
            the convex problem more strongly convex and improves the Newton
            approximation but biases the classifier.
        feature_clip: Maximum L^2 norm for the penultimate features. Rows with
            a larger norm are rescaled to this value. Bounding ``||phi(x)||``
            is required for the theoretical residual-gradient bound.
        refit_max_iter: Maximum LBFGS iterations when fitting ``w*``.
        refit_tol: LBFGS tolerance (on gradient infinity norm).
        cg_max_iter: Maximum conjugate-gradient iterations when solving
            ``H_r * delta = g``.
        cg_tol: Conjugate-gradient relative-residual stopping criterion.
        noise_sigma: Explicit Gaussian noise standard deviation. If ``epsilon``
            is also set, ``sigma`` is auto-calibrated and this value is used
            as a floor.
        epsilon: Target differential-privacy-style epsilon for certification;
            when set to ``None`` the removal is deterministic (no noise).
        delta: Target delta for ``(epsilon, delta)``-certification.
        feature_batch_size: Batch size used during feature extraction.
    """

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


def _save_snapshot(model: nn.Module, snapshot_dir: Optional[str], epoch: int) -> Optional[str]:
    if snapshot_dir is None:
        return None
    os.makedirs(snapshot_dir, exist_ok=True)
    path = os.path.join(snapshot_dir, f"epoch_{epoch:03d}.pt")
    torch.save(model.state_dict(), path)
    return path


def _get_linear_head(model: nn.Module) -> nn.Linear:
    """Return the final linear layer of ``Net`` (``model.model.fc``)."""
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
    """Return ``(features, labels)`` by replacing the head with Identity.

    Features are returned on CPU as float32 to keep GPU memory free; the caller
    can move them back. Labels are int64.
    """
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
    """Row-wise rescale so that ``||features[i]||_2 <= feature_clip``."""
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
    """Mean-CE + (lambda/2)||w||^2 on the full feature matrix."""
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
    """Sum-CE only; used to form per-subset gradients before averaging."""
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
    """LBFGS refit of ``(W, b)`` on the full feature matrix."""
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
    """Return ``(sum_i grad_W l_i, sum_i grad_b l_i)`` without the L2 term."""
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
    """Compute ``H_r @ (vW, vb)`` where ``H_r = (1/N_r) sum nabla^2 l_i + lambda I``."""
    W_v = W.detach().clone().requires_grad_(True)
    b_v = b.detach().clone().requires_grad_(True)
    # Use the *mean*-CE Hessian plus l2_reg*I (matches the L_r(w) definition).
    loss = _head_loss(W_v, b_v, features_retain, labels_retain, l2_reg)
    gW, gb = torch.autograd.grad(loss, [W_v, b_v], create_graph=True)
    dot = (gW * vW).sum() + (gb * vb).sum()
    hW, hb = torch.autograd.grad(dot, [W_v, b_v])
    return hW.detach(), hb.detach()


def _flatten_pair(W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cat([W.reshape(-1), b.reshape(-1)])


def _unflatten_pair(v: torch.Tensor, W_shape, b_shape) -> Tuple[torch.Tensor, torch.Tensor]:
    nW = int(np.prod(W_shape))
    W = v[:nW].reshape(W_shape)
    b = v[nW:].reshape(b_shape)
    return W, b


def _conjugate_gradient(
    matvec: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    rhs_W: torch.Tensor,
    rhs_b: torch.Tensor,
    max_iter: int,
    tol: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve ``H @ x = rhs`` for symmetric positive-definite ``H`` via CG."""
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
    """Copy ``(W, b)`` into ``model.model.fc`` in-place."""
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
    """Guo et al. (2020) Theorem 1-style sigma from the residual gradient norm.

    Given a ``residual_norm``-approximate minimizer of the retain objective,
    perturbing it by ``N(0, sigma^2 I)`` yields ``(epsilon, delta)``-certified
    removal provided::

        sigma >= c(epsilon, delta) * residual_norm / (lambda * n_retain)

    where ``c(epsilon, delta) = sqrt(2 ln(1.25 / delta)) / epsilon`` is the
    Gaussian-mechanism constant.
    """
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
    """Last-layer certified data removal (Guo et al., 2020).

    Extracts penultimate features using the loaded feature extractor, refits
    the linear head to the convex L^2-regularized optimum, and replaces the
    head with the Newton-step approximation to the retain-only optimum. The
    feature extractor itself is left untouched.

    Returns a dict with the same ``{"model", "classwise_history",
    "snapshot_paths"}`` shape as the other baselines. ``classwise_history``
    contains three entries: the loaded model, the model with refit head
    (pre-removal baseline), and the model after the certified Newton step
    (optionally perturbed with Gaussian noise).
    """
    config = config or CertifiedConfig()

    history: List[np.ndarray] = []
    snapshot_paths: List[str] = []

    _, per_class = evaluate(model, testloader, num_classes=num_classes, device=device)
    history.append(per_class)
    path = _save_snapshot(model, snapshot_dir, 0)
    if path is not None:
        snapshot_paths.append(path)

    forget_feats, forget_labels = _extract_penultimate_features(model, forget_loader, device)
    retain_feats, retain_labels = _extract_penultimate_features(model, retain_loader, device)

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

    sum_grad_W_f, sum_grad_b_f = _sum_gradient_on_subset(
        W_star, b_star, forget_feats_dev, forget_labels_dev
    )
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
        residual_norm = float(
            torch.sqrt(residual_W.pow(2).sum() + residual_b.pow(2).sum()).item()
        )
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

    return {"model": model, "classwise_history": history, "snapshot_paths": snapshot_paths}


def run_certified_snapshots(
    model_factory: Callable[[], nn.Module],
    original_checkpoint_path: str,
    forget_loader,
    retain_loader,
    testloader,
    device: torch.device,
    snapshot_dir: str,
    final_checkpoint_path: Optional[str] = None,
    config: Optional[CertifiedConfig] = None,
    num_classes: int = 10,
) -> str:
    model = model_factory()
    model.load_state_dict(torch.load(original_checkpoint_path, map_location=device))
    result = run_certified_unlearning(
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
