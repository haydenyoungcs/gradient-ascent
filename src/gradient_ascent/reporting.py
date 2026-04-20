from __future__ import annotations

import csv
import os
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def save_similarity_trajectory_csv(
    epoch_rows: Sequence[Tuple[int, List[dict]]],
    out_path: str,
    metric_names: Iterable[str],
) -> None:
    metric_names = list(metric_names)
    header = ["epoch", "layer", "n_samples", "n_features"] + metric_names
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for epoch, rows in epoch_rows:
            for row in rows:
                writer.writerow({"epoch": epoch, **row})


def build_mean_metric_series(
    epoch_rows: Sequence[Tuple[int, List[dict]]],
    metric_names: Iterable[str],
) -> Tuple[List[int], Dict[str, List[float]]]:
    metric_names = list(metric_names)
    epochs = [epoch for epoch, _ in epoch_rows]
    series = {metric_name: [] for metric_name in metric_names}

    for _epoch, rows in epoch_rows:
        for metric_name in metric_names:
            series[metric_name].append(float(np.mean([row[metric_name] for row in rows])))

    return epochs, series


def save_metric_summary_plot(
    epoch_rows: Sequence[Tuple[int, List[dict]]],
    out_path: str,
    title: str,
    metric_names: Iterable[str],
    lower_better_metrics: Iterable[str],
) -> None:
    metric_names = list(metric_names)
    lower_better_metrics = set(lower_better_metrics)
    epochs, series = build_mean_metric_series(epoch_rows, metric_names)

    for metric_name in lower_better_metrics:
        vals = np.array(series[metric_name], dtype=np.float64)
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        if vmax > vmin:
            vals = (vals - vmin) / (vmax - vmin)
        else:
            vals = np.full_like(vals, 0.5)
        series[metric_name] = list(1.0 - vals)

    fig, ax = plt.subplots(1, 1, figsize=(16, 5), constrained_layout=True)
    for metric_name in metric_names:
        ax.plot(epochs, series[metric_name], marker="o", label=metric_name)

    ax.set_title(title)
    ax.set_xlabel("Unlearning step")
    ax.set_ylabel("Similarity (mean across layers)")
    ax.grid(alpha=0.3)
    ax.legend(ncol=3)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_accuracy_history_plot(
    series_by_label: Mapping[str, Sequence[float]],
    out_path: str,
    title: str,
    xlabel: str = "Epoch",
    ylabel: str = "Test accuracy",
) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)

    for label, values in series_by_label.items():
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, marker="o", linewidth=2, label=label)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def save_classwise_history_csv(
    epochs: Sequence[int],
    history: np.ndarray,
    class_names: Sequence[str],
    out_path: str,
) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fieldnames = ["epoch"] + list(class_names)
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for epoch_num, row in zip(epochs, history):
            writer.writerow({"epoch": int(epoch_num), **{class_name: float(val) for class_name, val in zip(class_names, row)}})
    return out_path


def save_classwise_percent_change_plot(
    epochs: Sequence[int],
    history: np.ndarray,
    class_names: Sequence[str],
    out_path: str,
    title: str,
) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    initial_acc = np.asarray(history[0], dtype=np.float64)
    denom = np.where(initial_acc == 0, 1e-12, initial_acc)
    percent_change = 100.0 * (np.asarray(history, dtype=np.float64) - initial_acc) / denom

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    for class_idx, class_name in enumerate(class_names):
        ax.plot(epochs, percent_change[:, class_idx], label=class_name)

    ax.set_xlabel("Unlearning step")
    ax.set_ylabel("Accuracy change (%)")
    ax.set_title(title)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.legend(loc="lower left", fontsize="small", ncol=2)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def save_classwise_absolute_accuracy_plot(
    epochs: Sequence[int],
    history: np.ndarray,
    class_names: Sequence[str],
    out_path: str,
    title: str,
) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    history = 100.0 * np.asarray(history, dtype=np.float64)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    for class_idx, class_name in enumerate(class_names):
        ax.plot(epochs, history[:, class_idx], label=class_name)

    ax.set_xlabel("Unlearning step")
    ax.set_ylabel("Class accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.legend(loc="lower left", fontsize="small", ncol=2)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def save_mia_baseline_csv(baseline: Mapping[str, float], out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in baseline.items():
            writer.writerow({"metric": key, "value": float(value)})
    return out_path


def save_mia_trajectory_csv(rows: Sequence[Mapping[str, float]], out_path: str) -> str:
    if not rows:
        raise RuntimeError("Cannot save empty MIA trajectory rows.")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def save_mia_metric_grid_plot(
    rows: Sequence[Mapping[str, float]],
    out_path: str,
    algorithm_label: str,
    panel_metrics: Sequence[Tuple[str, str]],
    baseline: Mapping[str, float] | None = None,
) -> str:
    if not rows:
        raise RuntimeError("Cannot plot empty MIA trajectory rows.")

    baseline = dict(baseline or {})
    epochs = [int(row["epoch"]) for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.reshape(-1)

    for idx, (metric_key, metric_label) in enumerate(panel_metrics):
        ax = axes[idx]
        vals = [float(row[metric_key]) for row in rows]
        ax.plot(epochs, vals, marker="o", linewidth=2, label=algorithm_label)
        if metric_key in baseline:
            ax.axhline(baseline[metric_key], linestyle="--", linewidth=1.8, color="black", label="Retrained baseline")
        ax.set_title(metric_label)
        ax.set_xlabel("Unlearning step")
        ax.set_ylabel("Attack metric")
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def save_mia_control_comparison_plot(
    rows: Sequence[Mapping[str, float]],
    out_path: str,
    algorithm_label: str,
    retain_control_label: int,
) -> str:
    if not rows:
        raise RuntimeError("Cannot plot empty MIA trajectory rows.")

    epochs = [int(row["epoch"]) for row in rows]
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
    ax.plot(
        epochs,
        [float(row["forget_logreg_mean_member_prob"]) for row in rows],
        marker="o",
        linewidth=2,
        label="Forget (frog)",
    )
    ax.plot(
        epochs,
        [float(row["retain_logreg_mean_member_prob"]) for row in rows],
        marker="s",
        linewidth=2,
        label=f"Retain (class={retain_control_label})",
    )
    ax.set_title(f"{algorithm_label} MIA control comparison (lower = less inferable)")
    ax.set_xlabel("Unlearning step")
    ax.set_ylabel("LogReg mean member probability")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def load_mean_series(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing trajectory CSV: {csv_path}")

    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        metrics_in_file = [key for key in fieldnames if key not in {"epoch", "layer", "n_samples", "n_features"}]

        by_epoch: Dict[int, Dict[str, List[float]]] = {}
        for row in reader:
            epoch = int(row["epoch"])
            by_epoch.setdefault(epoch, {metric_name: [] for metric_name in metrics_in_file})
            for metric_name in metrics_in_file:
                by_epoch[epoch][metric_name].append(float(row[metric_name]))

    epochs_sorted = sorted(by_epoch.keys())
    series = {
        metric_name: [float(np.mean(by_epoch[epoch][metric_name])) for epoch in epochs_sorted]
        for metric_name in metrics_in_file
    }
    return epochs_sorted, series, metrics_in_file


def load_mia_series(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing MIA trajectory CSV: {csv_path}")

    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if not rows:
            raise RuntimeError(f"Empty MIA trajectory CSV: {csv_path}")

        metrics = [key for key in (reader.fieldnames or []) if key != "epoch"]
        rows_sorted = sorted(rows, key=lambda row: int(row["epoch"]))
        epochs = [int(row["epoch"]) for row in rows_sorted]
        series = {metric: [float(row[metric]) for row in rows_sorted] for metric in metrics}
    return epochs, series, metrics


def load_mia_baseline(csv_path: str) -> Dict[str, float]:
    baseline: Dict[str, float] = {}
    if not os.path.exists(csv_path):
        return baseline
    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            baseline[row["metric"]] = float(row["value"])
    return baseline
