from __future__ import annotations

import csv
import os
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def min_max_to_similarity_01(
    values: np.ndarray,
    *,
    lower_is_more_similar: bool,
    range_from: np.ndarray | None = None,
) -> np.ndarray:
    """Min–max scale to [0, 1] so 1 = most similar, 0 = least.

    Range is taken from ``range_from`` if provided (useful for pooling across runs),
    otherwise from ``values``. Set ``lower_is_more_similar=True`` for metrics like
    Euclidean or KL where smaller is closer.
    """
    arr = np.asarray(values, dtype=np.float64)
    ref = np.asarray(range_from, dtype=np.float64) if range_from is not None else arr
    vmin = float(np.min(ref))
    vmax = float(np.max(ref))
    if vmax > vmin:
        if lower_is_more_similar:
            return (vmax - arr) / (vmax - vmin)
        return (arr - vmin) / (vmax - vmin)
    return np.full_like(arr, 0.5)


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


def orient_epoch_rows_for_similarity(
    epoch_rows: Sequence[Tuple[int, List[dict]]],
    metric_names: Iterable[str],
    lower_better_metrics: Iterable[str],
) -> Sequence[Tuple[int, List[dict]]]:
    """Rescale every metric across the (epoch, layer) cells to [0, 1] with 1 = most similar.

    Distance-like metrics (``euclidean``, ``kl_sym``) are inverted so they match
    CKA/CCA/cosine after rescaling.
    """
    metric_names = list(metric_names)
    lower_better_metrics = set(lower_better_metrics)
    oriented_rows = [(epoch, [{**row} for row in rows]) for epoch, rows in epoch_rows]

    for metric_name in metric_names:
        vals = np.array(
            [float(row[metric_name]) for _epoch, rows in oriented_rows for row in rows],
            dtype=np.float64,
        )
        scaled_all = min_max_to_similarity_01(vals, lower_is_more_similar=(metric_name in lower_better_metrics))
        flat_idx = 0
        for _epoch, rows in oriented_rows:
            for row in rows:
                row[metric_name] = float(scaled_all[flat_idx])
                flat_idx += 1
    return oriented_rows


def save_metric_summary_plot(
    epoch_rows: Sequence[Tuple[int, List[dict]]],
    out_path: str,
    title: str,
    metric_names: Iterable[str],
    lower_better_metrics: Iterable[str],
) -> None:
    metric_names = list(metric_names)
    oriented_rows = orient_epoch_rows_for_similarity(epoch_rows, metric_names, lower_better_metrics)
    epochs = [epoch for epoch, _ in oriented_rows]

    stats_by_metric = {metric_name: {"mean": [], "min": [], "max": []} for metric_name in metric_names}
    for _epoch, rows in oriented_rows:
        for metric_name in metric_names:
            vals = np.array([float(row[metric_name]) for row in rows], dtype=np.float64)
            stats_by_metric[metric_name]["mean"].append(float(np.mean(vals)))
            stats_by_metric[metric_name]["min"].append(float(np.min(vals)))
            stats_by_metric[metric_name]["max"].append(float(np.max(vals)))

    n_metrics = len(metric_names)
    n_cols = min(2, n_metrics)
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.2 * n_rows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx]
        metric_stats = stats_by_metric[metric_name]
        mean_vals = np.array(metric_stats["mean"], dtype=np.float64)
        min_vals = np.array(metric_stats["min"], dtype=np.float64)
        max_vals = np.array(metric_stats["max"], dtype=np.float64)
        ax.plot(epochs, mean_vals, marker="o", linewidth=2)
        ax.fill_between(epochs, min_vals, max_vals, alpha=0.2)
        ax.set_title(metric_name)
        ax.set_xlabel("Unlearning step")
        ax.set_ylabel("Similarity to reference")
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.3)

    for idx in range(n_metrics, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(title)
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
    history: np.ndarray,
    class_names: Sequence[str],
    out_path: str,
    title: str,
) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    initial_acc = np.asarray(history[0], dtype=np.float64)
    denom = np.where(initial_acc == 0, 1e-12, initial_acc)
    percent_change = 100.0 * (np.asarray(history, dtype=np.float64) - initial_acc) / denom

    final_change = percent_change[-1]
    order = np.argsort(final_change)
    ordered_classes = [class_names[idx] for idx in order]
    ordered_final_change = final_change[order]
    y_pos = np.arange(len(ordered_classes))
    colors = ["#c44e52" if val < 0 else "#4c72b0" for val in ordered_final_change]

    fig, ax = plt.subplots(1, 1, figsize=(12, 7), constrained_layout=True)
    ax.barh(y_pos, ordered_final_change, color=colors)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Accuracy change (%)")
    ax.set_ylabel("Class")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered_classes)
    ax.grid(axis="x", alpha=0.3)

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
    initial_acc = history[0]
    denom = np.where(initial_acc == 0, 1e-12, initial_acc)
    percent_change = 100.0 * (history - initial_acc) / denom

    final_acc = history[-1]
    initial_acc = history[0]
    delta = final_acc - initial_acc
    order = np.argsort(final_acc)
    ordered_classes = [class_names[idx] for idx in order]
    ordered_final_acc = final_acc[order]
    ordered_delta = delta[order]

    if len(epochs) > 3:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
        for class_idx, class_name in enumerate(class_names):
            ax.plot(epochs, percent_change[:, class_idx], label=class_name)

        ax.set_xlabel("Unlearning step")
        ax.set_ylabel("Accuracy change (%)")
        ax.set_title(title)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.legend(loc="lower left", fontsize="small", ncol=2)
        ax.grid(alpha=0.3)
    else:
        y_pos = np.arange(len(ordered_classes))
        colors = ["#c44e52" if val < 0 else "#4c72b0" for val in ordered_delta]

        fig, ax = plt.subplots(1, 1, figsize=(12, 7), constrained_layout=True)
        ax.barh(y_pos, ordered_final_acc, color=colors)
        ax.set_title(title)
        ax.set_xlabel("Accuracy (%)")
        ax.set_ylabel("Class")
        ax.set_xlim(0, 100)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ordered_classes)
        ax.grid(axis="x", alpha=0.3)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def save_classwise_accuracy_bar_plot(
    classwise_acc: Sequence[float],
    class_names: Sequence[str],
    out_path: str,
    title: str,
) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    values = 100.0 * np.asarray(classwise_acc, dtype=np.float64)
    x = np.arange(len(class_names))

    fig, ax = plt.subplots(1, 1, figsize=(12, 5), constrained_layout=True)
    ax.bar(x, values, color="#4c72b0")
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def save_classwise_percent_difference_bar_plot(
    original_classwise_acc: Sequence[float],
    retrained_classwise_acc: Sequence[float],
    class_names: Sequence[str],
    out_path: str,
    title: str,
) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    original = np.asarray(original_classwise_acc, dtype=np.float64)
    retrained = np.asarray(retrained_classwise_acc, dtype=np.float64)
    denom = np.where(np.abs(original) < 1e-12, 1e-12, np.abs(original))
    percent_diff = 100.0 * (retrained - original) / denom
    x = np.arange(len(class_names))
    colors = ["#c44e52" if value < 0 else "#4c72b0" for value in percent_diff]

    fig, ax = plt.subplots(1, 1, figsize=(12, 5), constrained_layout=True)
    ax.bar(x, percent_diff, color=colors)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Difference vs original (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)

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
        ci_low_key = f"{metric_key}_ci_low"
        ci_high_key = f"{metric_key}_ci_high"
        if ci_low_key in rows[0] and ci_high_key in rows[0]:
            ci_low = np.array([float(row[ci_low_key]) for row in rows], dtype=np.float64)
            ci_high = np.array([float(row[ci_high_key]) for row in rows], dtype=np.float64)
            if np.all(np.isfinite(ci_low)) and np.all(np.isfinite(ci_high)):
                ax.fill_between(epochs, ci_low, ci_high, alpha=0.2, label="95% CI")
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


def save_unlearning_runtime_bar_plot(
    runtime_seconds_by_algorithm: Mapping[str, float],
    out_path: str,
    title: str = "Unlearning runtime by algorithm",
) -> str:
    if not runtime_seconds_by_algorithm:
        raise RuntimeError("Cannot plot unlearning runtimes: input mapping is empty.")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    labels = list(runtime_seconds_by_algorithm.keys())
    values = np.asarray([float(runtime_seconds_by_algorithm[label]) for label in labels], dtype=np.float64)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
    bars = ax.bar(x, values, color="#4c72b0")
    ax.set_title(title)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.1f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

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
