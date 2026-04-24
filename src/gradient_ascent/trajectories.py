from __future__ import annotations

import os
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from .reporting import orient_epoch_rows_for_similarity


EpochRows = Sequence[Tuple[int, List[dict]]]


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


def compute_epoch_rows_from_snapshots(
    snapshot_dir: str,
    model_factory: Callable[[], torch.nn.Module],
    reference_acts,
    activation_collector: Callable[[torch.nn.Module], Dict[str, np.ndarray]],
    pair_evaluator: Callable[[Dict[str, np.ndarray], Dict[str, np.ndarray]], List[dict]],
    map_location: torch.device | str,
) -> List[Tuple[int, List[dict]]]:
    epoch_rows = []
    for epoch_num, checkpoint_path in list_snapshot_paths(snapshot_dir):
        model = model_factory()
        model.load_state_dict(torch.load(checkpoint_path, map_location=map_location))
        model.eval()

        acts_t = activation_collector(model)
        rows = pair_evaluator(acts_t, reference_acts)
        epoch_rows.append((epoch_num, rows))

    return sorted(epoch_rows, key=lambda item: item[0])


def save_similarity_heatmap_grid(
    epoch_rows: EpochRows,
    out_path: str,
    algorithm_key: str,
    reference_key: str,
    layer_names: Iterable[str],
    metric_names: Iterable[str],
    lower_better_metrics: Iterable[str],
) -> str:
    layer_names = list(layer_names)
    metric_names = list(metric_names)
    oriented_rows = orient_epoch_rows_for_similarity(epoch_rows, metric_names, lower_better_metrics)
    epochs = [epoch for epoch, _ in oriented_rows]

    matrices = {metric_name: np.zeros((len(layer_names), len(epochs)), dtype=np.float64) for metric_name in metric_names}
    layer_to_idx = {layer_name: idx for idx, layer_name in enumerate(layer_names)}
    for epoch_idx, (_epoch, rows) in enumerate(oriented_rows):
        for row in rows:
            layer_idx = layer_to_idx[row["layer"]]
            for metric_name in metric_names:
                matrices[metric_name][layer_idx, epoch_idx] = float(row[metric_name])

    n_metrics = len(metric_names)
    n_cols = min(2, n_metrics)
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.3 * n_rows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx]
        im = ax.imshow(
            matrices[metric_name],
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            origin="lower",
            interpolation="nearest",
        )
        ax.set_title(metric_name)
        ax.set_xlabel("Unlearning step")
        ax.set_ylabel("Layer")
        ax.set_xticks(range(len(epochs)))
        ax.set_xticklabels(epochs)
        ax.set_yticks(range(len(layer_names)))
        ax.set_yticklabels(layer_names)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Similarity")

    for idx in range(n_metrics, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(
        f"{algorithm_key.upper()} vs {reference_key.capitalize()} across layers\n"
        "(all metrics scaled to higher = more similar)"
    )
    fig.savefig(out_path, dpi=180)
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
