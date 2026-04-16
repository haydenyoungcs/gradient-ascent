from __future__ import annotations

from typing import Callable, Dict, List, Mapping

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .training import build_amp_config
from .trajectories import list_snapshot_paths


def collect_attack_stats(model: nn.Module, loader, device: torch.device) -> Dict[str, np.ndarray]:
    amp = build_amp_config(device)
    criterion = nn.CrossEntropyLoss(reduction="none")

    true_prob_all = []
    max_prob_all = []
    loss_all = []
    entropy_all = []
    margin_all = []
    top5_all = []

    model.eval()
    with torch.inference_mode():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=amp.dtype, enabled=amp.enabled):
                logits = model(inputs)

            logits_f32 = logits.float()
            probs = torch.softmax(logits_f32, dim=1)
            losses = criterion(logits_f32, labels)

            true_prob = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            max_prob = torch.max(probs, dim=1).values

            topk = torch.topk(probs, k=min(5, probs.shape[1]), dim=1, largest=True, sorted=True).values
            if topk.shape[1] < 5:
                pad = torch.zeros((topk.shape[0], 5 - topk.shape[1]), device=topk.device, dtype=topk.dtype)
                topk = torch.cat([topk, pad], dim=1)

            margin = topk[:, 0] - topk[:, 1]
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)

            true_prob_all.append(true_prob.detach().cpu().numpy())
            max_prob_all.append(max_prob.detach().cpu().numpy())
            loss_all.append(losses.detach().cpu().numpy())
            entropy_all.append(entropy.detach().cpu().numpy())
            margin_all.append(margin.detach().cpu().numpy())
            top5_all.append(topk.detach().cpu().numpy())

    return {
        "true_prob": np.concatenate(true_prob_all, axis=0).astype(np.float64),
        "max_prob": np.concatenate(max_prob_all, axis=0).astype(np.float64),
        "loss": np.concatenate(loss_all, axis=0).astype(np.float64),
        "entropy": np.concatenate(entropy_all, axis=0).astype(np.float64),
        "margin": np.concatenate(margin_all, axis=0).astype(np.float64),
        "top5_probs": np.concatenate(top5_all, axis=0).astype(np.float64),
    }


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    valid = fpr <= float(target_fpr)
    if not np.any(valid):
        return 0.0
    return float(np.max(tpr[valid]))


def mia_threshold(member_scores, nonmember_scores) -> Dict[str, float]:
    y_true = np.concatenate(
        [
            np.ones(len(member_scores), dtype=np.int64),
            np.zeros(len(nonmember_scores), dtype=np.int64),
        ]
    )
    scores = np.concatenate(
        [
            np.asarray(member_scores, dtype=np.float64),
            np.asarray(nonmember_scores, dtype=np.float64),
        ]
    )
    return {
        "auc": float(roc_auc_score(y_true, scores)),
        "tpr_at_1pct": tpr_at_fpr(y_true, scores, 0.01),
        "tpr_at_0p1pct": tpr_at_fpr(y_true, scores, 0.001),
    }


def make_attack_features(stats: Dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack(
        [
            stats["true_prob"],
            stats["max_prob"],
            stats["loss"],
            stats["entropy"],
            stats["margin"],
            stats["top5_probs"],
        ]
    )


def mia_logreg(member_features, nonmember_features, seed: int = 0) -> Dict[str, float]:
    x_member = np.asarray(member_features, dtype=np.float64)
    x_nonmember = np.asarray(nonmember_features, dtype=np.float64)

    x = np.concatenate([x_member, x_nonmember], axis=0)
    y = np.concatenate(
        [
            np.ones(len(x_member), dtype=np.int64),
            np.zeros(len(x_nonmember), dtype=np.int64),
        ]
    )

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    n_splits = min(5, n_pos, n_neg)
    if n_splits < 2:
        raise RuntimeError(f"Not enough samples for stratified CV: pos={n_pos}, neg={n_neg}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    oof_scores = np.zeros(len(y), dtype=np.float64)

    for tr_idx, te_idx in skf.split(x, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(x[tr_idx], y[tr_idx])
        oof_scores[te_idx] = clf.predict_proba(x[te_idx])[:, 1]

    return {
        "auc": float(roc_auc_score(y, oof_scores)),
        "tpr_at_1pct": tpr_at_fpr(y, oof_scores, 0.01),
        "tpr_at_0p1pct": tpr_at_fpr(y, oof_scores, 0.001),
        "mean_member_prob": float(np.mean(oof_scores[y == 1])),
    }


def evaluate_probe_mia_metrics(
    model: nn.Module,
    member_loader,
    nonmember_loader,
    device: torch.device,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    member_stats = collect_attack_stats(model, member_loader, device)
    nonmember_stats = collect_attack_stats(model, nonmember_loader, device)

    loss_metrics = mia_threshold(-member_stats["loss"], -nonmember_stats["loss"])
    entropy_metrics = mia_threshold(-member_stats["entropy"], -nonmember_stats["entropy"])
    logreg_metrics = mia_logreg(
        make_attack_features(member_stats),
        make_attack_features(nonmember_stats),
        seed=seed,
    )

    return {
        "loss": loss_metrics,
        "entropy": entropy_metrics,
        "logreg": logreg_metrics,
    }


def flatten_mia_metrics_row(
    epoch: int,
    forget_metrics: Mapping[str, Mapping[str, float]],
    retain_metrics: Mapping[str, Mapping[str, float]],
) -> Dict[str, float]:
    return {
        "epoch": int(epoch),
        "forget_loss_auc": float(forget_metrics["loss"]["auc"]),
        "forget_loss_tpr_at_1pct": float(forget_metrics["loss"]["tpr_at_1pct"]),
        "forget_loss_tpr_at_0p1pct": float(forget_metrics["loss"]["tpr_at_0p1pct"]),
        "forget_entropy_auc": float(forget_metrics["entropy"]["auc"]),
        "forget_entropy_tpr_at_1pct": float(forget_metrics["entropy"]["tpr_at_1pct"]),
        "forget_entropy_tpr_at_0p1pct": float(forget_metrics["entropy"]["tpr_at_0p1pct"]),
        "forget_logreg_auc": float(forget_metrics["logreg"]["auc"]),
        "forget_logreg_tpr_at_1pct": float(forget_metrics["logreg"]["tpr_at_1pct"]),
        "forget_logreg_tpr_at_0p1pct": float(forget_metrics["logreg"]["tpr_at_0p1pct"]),
        "forget_logreg_mean_member_prob": float(forget_metrics["logreg"]["mean_member_prob"]),
        "retain_loss_auc": float(retain_metrics["loss"]["auc"]),
        "retain_loss_tpr_at_1pct": float(retain_metrics["loss"]["tpr_at_1pct"]),
        "retain_entropy_auc": float(retain_metrics["entropy"]["auc"]),
        "retain_logreg_auc": float(retain_metrics["logreg"]["auc"]),
        "retain_logreg_tpr_at_1pct": float(retain_metrics["logreg"]["tpr_at_1pct"]),
        "retain_logreg_mean_member_prob": float(retain_metrics["logreg"]["mean_member_prob"]),
    }


def compute_mia_baseline(
    model: nn.Module,
    forget_member_loader,
    forget_nonmember_loader,
    retain_member_loader,
    retain_nonmember_loader,
    device: torch.device,
    seed: int,
) -> Dict[str, float]:
    forget_metrics = evaluate_probe_mia_metrics(
        model,
        forget_member_loader,
        forget_nonmember_loader,
        device,
        seed=seed,
    )
    retain_metrics = evaluate_probe_mia_metrics(
        model,
        retain_member_loader,
        retain_nonmember_loader,
        device,
        seed=seed,
    )
    row = flatten_mia_metrics_row(0, forget_metrics, retain_metrics)
    return {
        "forget_loss_auc": row["forget_loss_auc"],
        "forget_logreg_auc": row["forget_logreg_auc"],
        "forget_logreg_tpr_at_1pct": row["forget_logreg_tpr_at_1pct"],
        "forget_logreg_mean_member_prob": row["forget_logreg_mean_member_prob"],
    }


def compute_mia_trajectory_rows(
    snapshot_dir: str,
    model_factory: Callable[[], nn.Module],
    forget_member_loader,
    forget_nonmember_loader,
    retain_member_loader,
    retain_nonmember_loader,
    device: torch.device,
    seed_base: int,
    map_location: torch.device | str,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for epoch_num, checkpoint_path in list_snapshot_paths(snapshot_dir):
        model = model_factory()
        model.load_state_dict(torch.load(checkpoint_path, map_location=map_location))
        model.eval()

        forget_metrics = evaluate_probe_mia_metrics(
            model,
            forget_member_loader,
            forget_nonmember_loader,
            device,
            seed=seed_base + int(epoch_num),
        )
        retain_metrics = evaluate_probe_mia_metrics(
            model,
            retain_member_loader,
            retain_nonmember_loader,
            device,
            seed=seed_base + int(epoch_num),
        )
        rows.append(flatten_mia_metrics_row(epoch_num, forget_metrics, retain_metrics))

    return rows
