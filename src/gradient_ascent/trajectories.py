from __future__ import annotations

import os
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np
import torch

from .reporting import orient_epoch_rows_for_similarity


EpochRows = Sequence[Tuple[int, List[dict]]]


def _activation_cache_path(activation_cache_dir: str, epoch_num: int) -> str:
    return os.path.join(activation_cache_dir, f"epoch_{epoch_num}.npz")


def _load_cached_activations(cache_path: str) -> Optional[Dict[str, np.ndarray]]:
    if not os.path.exists(cache_path):
        return None
    with np.load(cache_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _save_cached_activations(cache_path: str, acts: Dict[str, np.ndarray]) -> None:
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    # Uncompressed NPZ is intentionally faster to write/read for iterative
    # notebook reruns where activation extraction dominates wall-clock time.
    np.savez(cache_path, **acts)


def list_snapshot_paths(snapshot_dir: str) -> List[Tuple[int, str]]:
    if not os.path.isdir(snapshot_dir):
        raise FileNotFoundError(f"Missing snapshot directory: {snapshot_dir}")

    snapshot_files = sorted(
        filename for filename in os.listdir(snapshot_dir) if filename.startswith("epoch_") and filename.endswith(".pt")
    )
    if not snapshot_files:
        raise RuntimeError(f"No snapshot checkpoints found in {snapshot_dir}")

    return [
        (int(filename.replace("epoch_", "").replace(".pt", "")), os.path.join(snapshot_dir, filename))
        for filename in snapshot_files
    ]


def compute_epoch_rows_from_snapshots_multi_reference(
    snapshot_dir: str,
    model_factory: Callable[[], torch.nn.Module],
    reference_acts_map: Dict[str, Dict[str, object]],
    activation_collector: Callable[[torch.nn.Module], Dict[str, np.ndarray]],
    pair_evaluator: Callable[[Dict[str, object], Dict[str, object], Optional[str], Optional[Dict[str, float]]], List[dict]],
    map_location: torch.device | str,
    similarity_log_prefix_map: Optional[Dict[str, Optional[str]]] = None,
    activation_cache_dir: Optional[str] = None,
    activation_preparer: Optional[Callable[[Dict[str, np.ndarray]], Dict[str, object]]] = None,
    metric_timing_seconds: Optional[Dict[str, float]] = None,
) -> Dict[str, List[Tuple[int, List[dict]]]]:
    """Compute snapshot activations once, then score against multiple references."""
    rows_by_reference: Dict[str, List[Tuple[int, List[dict]]]] = {key: [] for key in reference_acts_map}
    for epoch_num, checkpoint_path in list_snapshot_paths(snapshot_dir):
        cache_path = (
            _activation_cache_path(activation_cache_dir, epoch_num) if activation_cache_dir is not None else None
        )
        acts_t = _load_cached_activations(cache_path) if cache_path is not None else None
        if acts_t is None:
            model = model_factory()
            model.load_state_dict(torch.load(checkpoint_path, map_location=map_location))
            model.eval()

            any_prefix = next(
                (prefix for prefix in (similarity_log_prefix_map or {}).values() if prefix is not None),
                None,
            )
            if any_prefix:
                print(f"{any_prefix} epoch={epoch_num} | collecting activations...", flush=True)
            acts_t = activation_collector(model)
            if cache_path is not None:
                _save_cached_activations(cache_path, acts_t)

        acts_for_evaluation: Dict[str, object]
        if activation_preparer is not None:
            acts_for_evaluation = activation_preparer(acts_t)
        else:
            acts_for_evaluation = acts_t

        for reference_key, reference_acts in reference_acts_map.items():
            similarity_log_prefix = (similarity_log_prefix_map or {}).get(reference_key)
            if similarity_log_prefix:
                print(
                    f"{similarity_log_prefix} epoch={epoch_num} | computing similarity scores...",
                    flush=True,
                )
            rows = pair_evaluator(acts_for_evaluation, reference_acts, similarity_log_prefix, metric_timing_seconds)
            rows_by_reference[reference_key].append((epoch_num, rows))

    for reference_key in rows_by_reference:
        rows_by_reference[reference_key] = sorted(rows_by_reference[reference_key], key=lambda item: item[0])
    return rows_by_reference


def save_similarity_metric_timing_plot(
    metric_timing_seconds: Mapping[str, float],
    out_path: str,
    algorithm_key: str,
) -> str:
    if not metric_timing_seconds:
        raise RuntimeError("Cannot create metric timing plot: metric_timing_seconds is empty.")

    metric_names = list(metric_timing_seconds.keys())
    values = np.array([float(metric_timing_seconds[name]) for name in metric_names], dtype=np.float64)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5), constrained_layout=True)
    y_positions = np.arange(len(metric_names))
    ax.barh(y_positions, values, color="#4c72b0")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(metric_names)
    ax.set_xlabel("Total compute time (s)")
    ax.set_title(f"{algorithm_key.upper()} similarity metric compute time totals")
    ax.grid(axis="x", alpha=0.3)

    for idx, value in enumerate(values):
        ax.text(float(value) + max(values) * 0.01 if float(max(values)) > 0 else 0.01, idx, f"{value:.2f}s", va="center")

    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def save_similarity_evolving_bar_plot(
    epoch_rows: EpochRows,
    out_path: str,
    algorithm_key: str,
    reference_key: str,
    metric_names: Iterable[str],
    lower_better_metrics: Iterable[str],
) -> str:
    metric_names = list(metric_names)
    oriented_rows = orient_epoch_rows_for_similarity(epoch_rows, metric_names, lower_better_metrics)
    if not oriented_rows:
        raise RuntimeError("Cannot create evolving similarity bar plot: epoch_rows is empty.")

    # Mean over layers at each step gives a compact per-metric trajectory frame.
    mean_values_by_epoch: list[tuple[int, np.ndarray]] = []
    for epoch_num, rows in oriented_rows:
        metric_means = []
        for metric_name in metric_names:
            values = np.array([float(row[metric_name]) for row in rows], dtype=np.float64)
            metric_means.append(float(np.mean(values)))
        mean_values_by_epoch.append((epoch_num, np.array(metric_means, dtype=np.float64)))

    x_positions = np.arange(len(metric_names))
    fig, ax = plt.subplots(1, 1, figsize=(12, 5), constrained_layout=True)
    writer = PillowWriter(fps=1)
    with writer.saving(fig, out_path, dpi=160):
        for epoch_num, mean_values in mean_values_by_epoch:
            ax.clear()
            colors = ["#4c72b0" for _ in metric_names]
            ax.bar(x_positions, mean_values, color=colors)
            ax.set_title(
                f"{algorithm_key.upper()} vs {reference_key.capitalize()} | "
                f"Unlearning step {epoch_num} (mean across layers)"
            )
            ax.set_xlabel("Similarity metric")
            ax.set_ylabel("Similarity to reference")
            ax.set_ylim(0.0, 1.02)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(metric_names, rotation=20, ha="right")
            ax.grid(axis="y", alpha=0.3)
            writer.grab_frame()

    plt.close(fig)
    return out_path


def save_similarity_evolving_grouped_bar_plot(
    epoch_rows: EpochRows,
    out_path: str,
    algorithm_key: str,
    reference_key: str,
    layer_names: Iterable[str],
    metric_names: Iterable[str],
    lower_better_metrics: Iterable[str],
) -> str:
    metric_names = list(metric_names)
    layer_names = list(layer_names)
    oriented_rows = orient_epoch_rows_for_similarity(epoch_rows, metric_names, lower_better_metrics)
    if not oriented_rows:
        raise RuntimeError("Cannot create evolving grouped bar plot: epoch_rows is empty.")

    layer_to_idx = {layer_name: idx for idx, layer_name in enumerate(layer_names)}
    grouped_values_by_epoch: list[tuple[int, np.ndarray]] = []
    for epoch_num, rows in oriented_rows:
        matrix = np.zeros((len(layer_names), len(metric_names)), dtype=np.float64)
        for row in rows:
            layer = row["layer"]
            if layer not in layer_to_idx:
                continue
            layer_idx = layer_to_idx[layer]
            for metric_idx, metric_name in enumerate(metric_names):
                matrix[layer_idx, metric_idx] = float(row[metric_name])
        grouped_values_by_epoch.append((epoch_num, matrix))

    x_positions = np.arange(len(metric_names), dtype=np.float64)
    n_layers = max(len(layer_names), 1)
    group_width = 0.8
    bar_width = group_width / n_layers
    color_map = plt.cm.get_cmap("tab10", n_layers)

    fig, ax = plt.subplots(1, 1, figsize=(14, 6), constrained_layout=True)
    writer = PillowWriter(fps=1)
    with writer.saving(fig, out_path, dpi=160):
        for epoch_num, matrix in grouped_values_by_epoch:
            ax.clear()
            for layer_idx, layer_name in enumerate(layer_names):
                offsets = x_positions - (group_width / 2.0) + (layer_idx + 0.5) * bar_width
                ax.bar(
                    offsets,
                    matrix[layer_idx],
                    width=bar_width * 0.95,
                    label=layer_name,
                    color=color_map(layer_idx),
                )

            ax.set_title(
                f"{algorithm_key.upper()} vs {reference_key.capitalize()} | "
                f"Unlearning step {epoch_num} (grouped by metric, bars = layers)"
            )
            ax.set_xlabel("Similarity metric")
            ax.set_ylabel("Similarity to reference")
            ax.set_ylim(0.0, 1.02)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(metric_names, rotation=20, ha="right")
            ax.grid(axis="y", alpha=0.3)
            ax.legend(title="Layer", ncol=2, fontsize="small")
            writer.grab_frame()

    plt.close(fig)
    return out_path


def save_combined_similarity_mia_plot(
    out_path: str,
    algo_keys: Sequence[str],
    algo_display: Dict[str, str],
    algo_epochs: Dict[str, Sequence[int]],
    algo_similarity_series: Dict[str, Dict[str, Sequence[float]]],
    all_similarity_metrics: Sequence[str],
    lower_better_metrics: Iterable[str],
    algo_mia_epochs: Dict[str, Sequence[int]],
    algo_mia_series: Dict[str, Dict[str, Sequence[float]]],
    mia_baseline: Dict[str, float],
    mia_panels: Sequence[Tuple[str, str]],
) -> str:
    lower_better_metrics = list(lower_better_metrics)

    for metric_name in lower_better_metrics:
        pooled = []
        for algo in algo_keys:
            pooled.extend(algo_similarity_series[algo][metric_name])
        pooled = np.array(pooled, dtype=np.float64)
        vmin = float(np.min(pooled))
        vmax = float(np.max(pooled))

        for algo in algo_keys:
            vals = np.array(algo_similarity_series[algo][metric_name], dtype=np.float64)
            if vmax > vmin:
                vals = (vals - vmin) / (vmax - vmin)
            else:
                vals = np.full_like(vals, 0.5)
            algo_similarity_series[algo][metric_name] = list(1.0 - vals)

    plot_metrics = [("similarity", metric, f"{metric} (scaled: higher=more similar)") for metric in all_similarity_metrics]
    plot_metrics += [("mia", metric, title) for metric, title in mia_panels]

    n_metrics = len(plot_metrics)
    n_cols = 3
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for idx, (metric_group, metric_key, title) in enumerate(plot_metrics):
        ax = axes[idx]

        for algo in algo_keys:
            if metric_group == "similarity":
                x_vals = algo_epochs[algo]
                y_vals = algo_similarity_series[algo][metric_key]
            else:
                x_vals = algo_mia_epochs[algo]
                y_vals = algo_mia_series[algo][metric_key]

            ax.plot(x_vals, y_vals, marker="o", linewidth=2, label=algo_display[algo])

        if metric_group == "mia" and metric_key in mia_baseline:
            ax.axhline(
                mia_baseline[metric_key],
                linestyle="--",
                linewidth=1.8,
                color="black",
                label="Retrained baseline",
            )

        ax.set_title(title)
        ax.set_xlabel("Unlearning step")
        ax.set_ylabel("Mean across layers" if metric_group == "similarity" else "Attack metric")
        ax.grid(alpha=0.3)

    for idx in range(n_metrics, len(axes)):
        axes[idx].axis("off")

    handles, labels = [], []
    for ax in axes[:n_metrics]:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path
