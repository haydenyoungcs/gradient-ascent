from __future__ import annotations

import csv
import math
import os
import shutil
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import (
    default_num_workers,
    load_cifar10_datasets,
    make_forget_retain_subsets,
    make_loader,
    subset_for_class,
)
from .experiments import (
    CombinedComparisonConfig,
    CoreExperimentArtifacts,
    CoreExperimentConfig,
    TrajectoryExperimentArtifacts,
    TrajectoryExperimentConfig,
    ensure_wandb_run,
    run_core_checkpoints,
    run_trajectory_analysis,
    save_combined_trajectory_comparison,
)
from .models import Net
from .reporting import (
    orient_epoch_rows_for_similarity,
    save_mia_baseline_csv,
    save_mia_metric_grid_plot,
    save_mia_trajectory_csv,
    save_metric_summary_plot,
    save_similarity_trajectory_csv,
)
from .similarity import (
    CCA,
    DEFAULT_LAYER_NAMES,
    HIGHER_BETTER_METRICS,
    LOWER_BETTER_METRICS,
    build_default_metrics,
    collect_model_activations,
    evaluate_pair_rows_prepared,
    prepare_activations_for_evaluation,
)
from .trajectories import (
    save_similarity_before_after_grouped_bar_plot,
    save_similarity_evolving_bar_plot,
    save_similarity_evolving_grouped_bar_plot,
)
from .training import build_amp_config, configure_runtime
from .unlearning import GAConfig, SCRUBConfig, SSDConfig

ALGORITHM_ORDER = ["ga", "ssd", "salun", "certified", "scrub"]
SIMILARITY_REFERENCES = ["retrained", "original"]


@dataclass(frozen=True)
class NotebookRuntime:
    num_classes: int
    out_dir: str
    model_depth: int
    device: torch.device
    use_cuda: bool
    num_workers: int
    use_bf16: bool
    trainset: object
    testset: object
    model_factory: Callable[[], nn.Module]
    core_config: CoreExperimentConfig


@dataclass(frozen=True)
class SimilaritySetup:
    layer_names: list[str]
    metrics: dict[str, object]
    higher_better_metrics: list[str]
    lower_better_metrics: list[str]
    plot_metric_names: list[str]


@dataclass(frozen=True)
class MultiTargetAggregateArtifacts:
    utility_csv_paths: dict[str, str]
    utility_relative_plot_paths: dict[str, str]
    utility_absolute_plot_paths: dict[str, str]
    utility_final_bar_plot_paths: dict[str, str]
    similarity_csv_paths: dict[str, dict[str, str]]
    similarity_summary_plot_paths: dict[str, dict[str, str]]
    similarity_evolving_bar_paths: dict[str, dict[str, str]]
    similarity_grouped_evolving_bar_paths: dict[str, dict[str, str]]
    similarity_before_after_plot_paths: dict[str, dict[str, str]]
    mia_csv_paths: dict[str, str]
    mia_grid_plot_paths: dict[str, str]
    mia_control_plot_paths: dict[str, str]
    mia_baseline_csv_path: Optional[str]


def _load_similarity_epoch_rows_from_csv(csv_path: str) -> list[tuple[int, list[dict]]]:
    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return []
    by_epoch: dict[int, list[dict]] = {}
    for row in rows:
        epoch = int(row["epoch"])
        row_dict = {
            "layer": row["layer"],
            "n_samples": int(row["n_samples"]),
            "n_features": int(row["n_features"]),
        }
        for key, value in row.items():
            if key in {"epoch", "layer", "n_samples", "n_features"}:
                continue
            row_dict[key] = float(value)
        by_epoch.setdefault(epoch, []).append(row_dict)
    return sorted(by_epoch.items(), key=lambda item: item[0])


def _load_classwise_history_csv(csv_path: str) -> tuple[list[int], np.ndarray]:
    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if not rows:
            raise RuntimeError(f"Empty classwise history CSV: {csv_path}")
        class_names = [name for name in (reader.fieldnames or []) if name != "epoch"]
        epochs = [int(row["epoch"]) for row in rows]
        history = np.array(
            [[float(row[class_name]) for class_name in class_names] for row in rows],
            dtype=np.float64,
        )
    return epochs, history


def _line_plot_two_series(
    epochs: list[int],
    forget_vals: np.ndarray,
    retain_vals: np.ndarray,
    *,
    out_path: str,
    title: str,
    ylabel: str,
) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
    ax.plot(epochs, forget_vals, marker="o", linewidth=2, label="Forgotten class")
    ax.plot(epochs, retain_vals, marker="s", linewidth=2, label="Retained classes (mean)")
    ax.set_title(title)
    ax.set_xlabel("Unlearning step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _final_bar_two_series(
    forget_val: float,
    retain_val: float,
    *,
    out_path: str,
    title: str,
    ylabel: str,
) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    labels = ["Forgotten class", "Retained classes (mean)"]
    vals = np.array([forget_val, retain_val], dtype=np.float64)
    colors = ["#c44e52", "#4c72b0"]
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True)
    x = np.arange(len(labels))
    ax.bar(x, vals, color=colors)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _average_similarity_csvs(csv_paths: list[str]) -> list[tuple[int, list[dict]]]:
    if not csv_paths:
        raise RuntimeError("Cannot average similarity CSVs: csv_paths is empty.")
    runs = [_load_similarity_epoch_rows_from_csv(path) for path in csv_paths]
    if any(not rows for rows in runs):
        raise RuntimeError("Cannot average similarity CSVs: at least one CSV has no rows.")

    runs_by_epoch: list[dict[int, list[dict]]] = []
    for path, run in zip(csv_paths, runs):
        by_ep: dict[int, list[dict]] = {}
        for epoch, body in run:
            ep = int(epoch)
            if ep in by_ep:
                raise RuntimeError(f"Duplicate epoch {ep} block in similarity CSV: {path}")
            by_ep[ep] = body
        runs_by_epoch.append(by_ep)

    common_epochs = set(runs_by_epoch[0].keys())
    for by_ep in runs_by_epoch[1:]:
        common_epochs &= set(by_ep.keys())
    if not common_epochs:
        raise RuntimeError(
            "No epoch appears in all similarity trajectory CSVs; cannot average. "
            f"Epoch sets (per file): {[sorted(d.keys()) for d in runs_by_epoch]}"
        )

    sorted_epochs = sorted(common_epochs)
    max_len = max(len(d) for d in runs_by_epoch)
    if len(sorted_epochs) < max_len:
        warnings.warn(
            "Similarity trajectories differ in unlearning length or epoch sets; "
            f"averaging over the intersection only ({len(sorted_epochs)} steps, "
            f"longest single run {max_len} steps).",
            UserWarning,
            stacklevel=2,
        )

    averaged: list[tuple[int, list[dict]]] = []
    for epoch in sorted_epochs:
        first_rows = runs_by_epoch[0][epoch]
        layer_order = [str(row["layer"]) for row in first_rows]
        metric_names = [
            key for key in first_rows[0].keys() if key not in {"layer", "n_samples", "n_features"}
        ]

        by_run_layer: list[dict[str, dict]] = []
        for by_ep in runs_by_epoch:
            row_map = {str(row["layer"]): row for row in by_ep[epoch]}
            by_run_layer.append(row_map)

        epoch_rows: list[dict] = []
        for layer in layer_order:
            row_ref = by_run_layer[0][layer]
            out_row = {
                "layer": layer,
                "n_samples": int(row_ref["n_samples"]),
                "n_features": int(row_ref["n_features"]),
            }
            for metric_name in metric_names:
                vals = [float(run_rows[layer][metric_name]) for run_rows in by_run_layer]
                out_row[metric_name] = float(np.mean(np.asarray(vals, dtype=np.float64)))
            epoch_rows.append(out_row)
        averaged.append((int(epoch), epoch_rows))
    return averaged


def _aggregate_binary_unlearning_curves(
    classwise_csv_paths_by_target: dict[int, str],
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not classwise_csv_paths_by_target:
        raise RuntimeError("No classwise CSVs provided for aggregate utility curves.")
    target_labels = sorted(classwise_csv_paths_by_target.keys())
    per_target_epochs: list[list[int]] = []
    per_target_histories: dict[int, np.ndarray] = {}
    for target_label in target_labels:
        epochs, history = _load_classwise_history_csv(classwise_csv_paths_by_target[target_label])
        per_target_epochs.append(epochs)
        per_target_histories[target_label] = history

    if any(epochs != per_target_epochs[0] for epochs in per_target_epochs[1:]):
        raise RuntimeError("Epoch mismatch across classwise histories; cannot aggregate.")
    epochs = per_target_epochs[0]

    forget_abs_runs = []
    retain_abs_runs = []
    forget_rel_runs = []
    retain_rel_runs = []
    for target_label in target_labels:
        history = per_target_histories[target_label]
        forget_abs = history[:, target_label]
        retain_idx = [idx for idx in range(history.shape[1]) if idx != target_label]
        retain_abs = history[:, retain_idx].mean(axis=1)

        forget_rel = 100.0 * (forget_abs - forget_abs[0]) / max(float(forget_abs[0]), 1e-12)
        retain_rel = 100.0 * (retain_abs - retain_abs[0]) / max(float(retain_abs[0]), 1e-12)

        forget_abs_runs.append(forget_abs)
        retain_abs_runs.append(retain_abs)
        forget_rel_runs.append(forget_rel)
        retain_rel_runs.append(retain_rel)

    mean_forget_abs = np.mean(np.stack(forget_abs_runs, axis=0), axis=0)
    mean_retain_abs = np.mean(np.stack(retain_abs_runs, axis=0), axis=0)
    mean_forget_rel = np.mean(np.stack(forget_rel_runs, axis=0), axis=0)
    mean_retain_rel = np.mean(np.stack(retain_rel_runs, axis=0), axis=0)
    return epochs, mean_forget_abs, mean_retain_abs, mean_forget_rel, mean_retain_rel


def _load_mia_rows_from_csv(csv_path: str) -> list[dict[str, float]]:
    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if not rows:
            raise RuntimeError(f"Empty MIA trajectory CSV: {csv_path}")
        out = []
        for row in rows:
            converted = {}
            for key, value in row.items():
                if key == "epoch":
                    converted[key] = int(value)
                else:
                    converted[key] = float(value)
            out.append(converted)
    return out


def _average_mia_rows_csvs(csv_paths: list[str]) -> list[dict[str, float]]:
    if not csv_paths:
        raise RuntimeError("Cannot average MIA CSVs: csv_paths is empty.")
    runs = [_load_mia_rows_from_csv(path) for path in csv_paths]
    runs_by_epoch: list[dict[int, dict[str, float]]] = []
    for path, run in zip(csv_paths, runs):
        by_ep: dict[int, dict[str, float]] = {}
        for row in run:
            ep = int(row["epoch"])
            if ep in by_ep:
                raise RuntimeError(f"Duplicate epoch {ep} in MIA trajectory CSV: {path}")
            by_ep[ep] = row
        runs_by_epoch.append(by_ep)

    common_epochs = set(runs_by_epoch[0].keys())
    for by_ep in runs_by_epoch[1:]:
        common_epochs &= set(by_ep.keys())
    if not common_epochs:
        raise RuntimeError(
            "No epoch appears in all MIA trajectory CSVs; cannot average. "
            f"Epoch sets (per file): {[sorted(d.keys()) for d in runs_by_epoch]}"
        )

    sorted_epochs = sorted(common_epochs)
    max_len = max(len(d) for d in runs_by_epoch)
    if len(sorted_epochs) < max_len:
        warnings.warn(
            "MIA trajectories differ in unlearning length or epoch sets; "
            f"averaging over the intersection only ({len(sorted_epochs)} steps, "
            f"longest single run {max_len} steps).",
            UserWarning,
            stacklevel=2,
        )

    ref_row = runs_by_epoch[0][sorted_epochs[0]]
    metric_keys = [key for key in ref_row.keys() if key != "epoch"]
    averaged: list[dict[str, float]] = []
    for epoch in sorted_epochs:
        out_row: dict[str, float] = {"epoch": int(epoch)}
        for key in metric_keys:
            vals = [float(runs_by_epoch[i][epoch][key]) for i in range(len(runs_by_epoch))]
            out_row[key] = float(np.mean(np.asarray(vals, dtype=np.float64)))
        averaged.append(out_row)
    return averaged


def _average_mia_baseline_csvs(csv_paths: list[str]) -> dict[str, float]:
    if not csv_paths:
        raise RuntimeError("Cannot average MIA baselines: csv_paths is empty.")
    by_metric: dict[str, list[float]] = {}
    for csv_path in csv_paths:
        with open(csv_path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                by_metric.setdefault(str(row["metric"]), []).append(float(row["value"]))
    return {
        metric: float(np.mean(np.asarray(values, dtype=np.float64)))
        for metric, values in by_metric.items()
        if values
    }


def _save_mia_forget_vs_retain_mean_plot(
    rows: list[dict[str, float]],
    out_path: str,
    algorithm_label: str,
) -> str:
    if not rows:
        raise RuntimeError("Cannot plot averaged MIA control comparison: rows is empty.")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    epochs = [int(row["epoch"]) for row in rows]
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
    ax.plot(
        epochs,
        [float(row["forget_logreg_mean_member_prob"]) for row in rows],
        marker="o",
        linewidth=2,
        label="Forget class (mean over targets)",
    )
    ax.plot(
        epochs,
        [float(row["retain_logreg_mean_member_prob"]) for row in rows],
        marker="s",
        linewidth=2,
        label="Retain control (mean over targets)",
    )
    ax.set_title(f"{algorithm_label} MIA control comparison (mean over forget labels)")
    ax.set_xlabel("Unlearning step")
    ax.set_ylabel("LogReg mean member probability")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def display_similarity_epoch_slider(
    csv_path: str,
    algorithm_key: str,
    reference_key: str,
    layer_names: list[str],
    metric_names: list[str],
    lower_better_metrics: list[str],
) -> None:
    """Display an interactive slider to scrub similarity epochs in-notebook."""
    from IPython.display import clear_output, display
    import ipywidgets as widgets

    epoch_rows = _load_similarity_epoch_rows_from_csv(csv_path)
    oriented_rows = orient_epoch_rows_for_similarity(epoch_rows, metric_names, lower_better_metrics)
    if not oriented_rows:
        print(f"No similarity rows available for {algorithm_key} vs {reference_key}.")
        return

    epoch_options = [int(epoch) for epoch, _ in oriented_rows]
    mode = widgets.ToggleButtons(
        options=[("Mean across layers", "mean"), ("Grouped by layer", "grouped")],
        description="View:",
    )
    slider = widgets.SelectionSlider(
        options=epoch_options,
        value=epoch_options[0],
        description="Epoch:",
        continuous_update=False,
    )
    out = widgets.Output()

    layer_to_idx = {layer: idx for idx, layer in enumerate(layer_names)}

    def _render(*_args):
        epoch = int(slider.value)
        rows_for_epoch = next(rows for ep, rows in oriented_rows if int(ep) == epoch)
        with out:
            clear_output(wait=True)
            if mode.value == "mean":
                fig, ax = plt.subplots(1, 1, figsize=(10, 4.5), constrained_layout=True)
                values = []
                for metric_name in metric_names:
                    vals = np.array([float(row[metric_name]) for row in rows_for_epoch], dtype=np.float64)
                    values.append(float(np.mean(vals)))
                x_positions = np.arange(len(metric_names))
                ax.bar(x_positions, values, color="#4c72b0")
                ax.set_xticks(x_positions)
                ax.set_xticklabels(metric_names, rotation=20, ha="right")
                ax.set_ylim(0.0, 1.02)
                ax.set_ylabel("Similarity to reference")
                ax.set_title(
                    f"{algorithm_key.upper()} vs {reference_key.capitalize()} | "
                    f"step {epoch} (mean across layers)"
                )
                ax.grid(axis="y", alpha=0.3)
            else:
                fig, ax = plt.subplots(1, 1, figsize=(12, 5), constrained_layout=True)
                matrix = np.zeros((len(layer_names), len(metric_names)), dtype=np.float64)
                for row in rows_for_epoch:
                    layer = row["layer"]
                    if layer not in layer_to_idx:
                        continue
                    layer_idx = layer_to_idx[layer]
                    for metric_idx, metric_name in enumerate(metric_names):
                        matrix[layer_idx, metric_idx] = float(row[metric_name])
                x_positions = np.arange(len(metric_names), dtype=np.float64)
                n_layers = max(len(layer_names), 1)
                group_width = 0.8
                bar_width = group_width / n_layers
                color_map = plt.cm.get_cmap("tab10", n_layers)
                for layer_idx, layer_name in enumerate(layer_names):
                    offsets = x_positions - (group_width / 2.0) + (layer_idx + 0.5) * bar_width
                    ax.bar(
                        offsets,
                        matrix[layer_idx],
                        width=bar_width * 0.95,
                        label=layer_name,
                        color=color_map(layer_idx),
                    )
                ax.set_xticks(x_positions)
                ax.set_xticklabels(metric_names, rotation=20, ha="right")
                ax.set_ylim(0.0, 1.02)
                ax.set_ylabel("Similarity to reference")
                ax.set_title(
                    f"{algorithm_key.upper()} vs {reference_key.capitalize()} | "
                    f"step {epoch} (grouped by metric, bars = layers)"
                )
                ax.grid(axis="y", alpha=0.3)
                ax.legend(title="Layer", ncol=2, fontsize="small")
            display(fig)
            plt.close(fig)

    mode.observe(_render, names="value")
    slider.observe(_render, names="value")
    _render()
    display(widgets.VBox([widgets.HBox([mode, slider]), out]))


def build_default_core_config(
    num_classes: int = 10,
    model_depth: int = 50,
    out_dir: str = "out",
    use_bf16: bool = False,
) -> CoreExperimentConfig:
    """Return the notebook's default baseline configuration bundle."""
    # Keep GA close to the naive forget-only baseline: full passes over the
    # forget loader, no BatchNorm freezing, and no gradient clipping.
    ga_config = GAConfig(
        lr=1.1e-4,
        epochs=6,
        max_batches_per_epoch=None,
        freeze_bn=False,
        grad_clip_norm=None,
    )
    # SSD was previously over-aggressive and expensive in this notebook setup.
    # These defaults trade a bit of forgetting strength for much lower runtime
    # and substantially less collateral damage on non-forgotten classes.
    ssd_config = SSDConfig(
        alpha=8.0,
        lambda_=0.9,
        eps=1e-12,
        fisher_batches=80,
        fisher_samples_per_batch=16,
        selection_basis="retain",
        fisher_mode="batch",
    )
    # SCRUB preset tuned to keep frog forgetting stronger late in training:
    # - more forget batches in the scrub phase,
    # - non-trivial residual forget pressure in recovery,
    # - lower retain CE weight so recovery does not quickly relearn frogs.
    scrub_config = SCRUBConfig(
        lr=5e-5,
        epochs=6,
        forget_phase_epochs=3,
        max_forget_batches_per_epoch=6,
        recovery_beta_scale=0.4,
        recovery_max_forget_batches_per_epoch=4,
        max_retain_batches_per_epoch=12,
        alpha=2.5,
        beta=0.8,
        gamma=2.0,
        temperature=2.0,
        weight_decay=1e-4,
        grad_clip_norm=0.5,
    )
    return CoreExperimentConfig(
        num_classes=num_classes,
        model_depth=model_depth,
        out_dir=out_dir,
        num_epochs=120,
        batch_size=128,
        training_lr=0.1,
        unlearn_batch_size=512 if use_bf16 else 256,
        ga_config=ga_config,
        ssd_config=ssd_config,
        scrub_config=scrub_config,
    )


def build_core_wandb_config(config: CoreExperimentConfig) -> dict[str, object]:
    return {
        "target_label": config.target_label,
        "num_epochs": config.num_epochs,
        "batch_size": config.batch_size,
        "model_depth": config.model_depth,
        "ga_lr": config.ga_config.lr,
        "ga_epochs": config.ga_config.epochs,
        "ga_max_batches_per_epoch": config.ga_config.max_batches_per_epoch,
        "ga_grad_clip_norm": config.ga_config.grad_clip_norm,
        "ssd_alpha": config.ssd_config.alpha,
        "ssd_lambda": config.ssd_config.lambda_,
        "ssd_fisher_batches": config.ssd_config.fisher_batches,
        "ssd_fisher_mode": config.ssd_config.fisher_mode,
        "scrub_lr": config.scrub_config.lr,
        "scrub_epochs": config.scrub_config.epochs,
        "scrub_forget_phase_epochs": config.scrub_config.forget_phase_epochs,
        "scrub_max_forget_batches_per_epoch": config.scrub_config.max_forget_batches_per_epoch,
        "scrub_max_retain_batches_per_epoch": config.scrub_config.max_retain_batches_per_epoch,
        "scrub_recovery_beta_scale": config.scrub_config.recovery_beta_scale,
        "scrub_recovery_max_forget_batches_per_epoch": config.scrub_config.recovery_max_forget_batches_per_epoch,
        "scrub_alpha": config.scrub_config.alpha,
        "scrub_beta": config.scrub_config.beta,
        "scrub_gamma": config.scrub_config.gamma,
        "scrub_temperature": config.scrub_config.temperature,
    }


def prepare_notebook_runtime(
    num_classes: int = 10,
    out_dir: str = "out",
    model_depth: int = 50,
    data_root: str = "./data",
    cifar10_download: bool = True,
    cifar10_download_retries: int = 2,
) -> NotebookRuntime:
    configure_runtime()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_config = build_amp_config(device)
    use_bf16 = amp_config.dtype == torch.bfloat16

    trainset, testset = load_cifar10_datasets(
        data_root,
        download=cifar10_download,
        download_retries=cifar10_download_retries,
    )
    use_cuda = device.type == "cuda"
    num_workers = default_num_workers(use_cuda)
    model_factory = lambda: Net(num_classes=num_classes, pretrained=False, model_depth=model_depth).to(device)
    core_config = build_default_core_config(
        num_classes=num_classes,
        model_depth=model_depth,
        out_dir=out_dir,
        use_bf16=use_bf16,
    )
    return NotebookRuntime(
        num_classes=num_classes,
        out_dir=out_dir,
        model_depth=model_depth,
        device=device,
        use_cuda=use_cuda,
        num_workers=num_workers,
        use_bf16=use_bf16,
        trainset=trainset,
        testset=testset,
        model_factory=model_factory,
        core_config=core_config,
    )


def run_notebook_core_experiment(
    runtime: NotebookRuntime,
    wandb_module=None,
    wandb_project: str = "gradient-ascent",
    wandb_name: str = "core-checkpoints",
    reuse_existing_checkpoints: bool = False,
    reuse_original_checkpoint: Optional[bool] = None,
    reuse_retrained_checkpoint: Optional[bool] = None,
    reuse_unlearned_checkpoints: Optional[bool] = None,
) -> tuple[CoreExperimentArtifacts, object]:
    wandb_run = ensure_wandb_run(
        wandb_module,
        project=wandb_project,
        name=wandb_name,
        config=build_core_wandb_config(runtime.core_config),
    )
    artifacts = run_core_checkpoints(
        model_factory=runtime.model_factory,
        trainset=runtime.trainset,
        testset=runtime.testset,
        device=runtime.device,
        use_cuda=runtime.use_cuda,
        num_workers=runtime.num_workers,
        config=runtime.core_config,
        reuse_existing_checkpoints=reuse_existing_checkpoints,
        reuse_original_checkpoint=reuse_original_checkpoint,
        reuse_retrained_checkpoint=reuse_retrained_checkpoint,
        reuse_unlearned_checkpoints=reuse_unlearned_checkpoints,
        wandb_run=wandb_run,
        wandb_module=wandb_module,
    )
    return artifacts, wandb_run


def prepare_similarity_setup(
    *,
    cca_max_columns: Optional[int] = 256,
    cca_column_subsample_seed: int = 43,
) -> SimilaritySetup:
    """Return the notebook's default similarity-analysis configuration."""
    layer_names = list(DEFAULT_LAYER_NAMES)
    metrics = build_default_metrics(
        cca_max_columns=cca_max_columns,
        cca_column_subsample_seed=cca_column_subsample_seed,
    )
    higher_better_metrics = list(HIGHER_BETTER_METRICS)
    lower_better_metrics = list(LOWER_BETTER_METRICS)
    plot_metric_names = higher_better_metrics + lower_better_metrics
    return SimilaritySetup(
        layer_names=layer_names,
        metrics=metrics,
        higher_better_metrics=higher_better_metrics,
        lower_better_metrics=lower_better_metrics,
        plot_metric_names=plot_metric_names,
    )


def build_default_trajectory_config(runtime: NotebookRuntime) -> TrajectoryExperimentConfig:
    return TrajectoryExperimentConfig(
        num_classes=runtime.num_classes,
        model_depth=runtime.model_depth,
        out_dir=runtime.out_dir,
        trajectory_batch_size=512 if runtime.use_bf16 else 256,
    )


def _similarity_setup_with_trajectory_cca(
    setup: SimilaritySetup,
    trajectory_config: TrajectoryExperimentConfig,
) -> SimilaritySetup:
    """Align the CCA object with trajectory speed settings (single knob in the notebook)."""
    metrics = dict(setup.metrics)
    metrics["cca"] = CCA(
        max_columns=trajectory_config.cca_max_columns,
        column_subsample_seed=trajectory_config.cca_column_subsample_seed,
    )
    return SimilaritySetup(
        layer_names=setup.layer_names,
        metrics=metrics,
        higher_better_metrics=setup.higher_better_metrics,
        lower_better_metrics=setup.lower_better_metrics,
        plot_metric_names=setup.plot_metric_names,
    )


def run_notebook_trajectory_experiment(
    runtime: NotebookRuntime,
    core_artifacts: CoreExperimentArtifacts,
    similarity_setup: SimilaritySetup,
    wandb_module=None,
    wandb_project: str = "gradient-ascent",
    wandb_name: str = "unlearning-algorithm-comparison",
) -> tuple[TrajectoryExperimentArtifacts, object]:
    """Run the notebook's full trajectory/MIA/similarity pipeline."""
    trajectory_config = build_default_trajectory_config(runtime)
    similarity_setup = _similarity_setup_with_trajectory_cca(similarity_setup, trajectory_config)
    wandb_run = ensure_wandb_run(wandb_module, project=wandb_project, name=wandb_name)
    artifacts = run_trajectory_analysis(
        model_factory=runtime.model_factory,
        trainset=runtime.trainset,
        testset=runtime.testset,
        device=runtime.device,
        use_cuda=runtime.use_cuda,
        num_workers=runtime.num_workers,
        config=trajectory_config,
        original_checkpoint_path=f"{runtime.out_dir}/original_net.pt",
        retrained_checkpoint_path=f"{runtime.out_dir}/retrained_from_scratch_net.pt",
        snapshot_dirs={
            algorithm_key: core_artifacts.algorithm_artifacts[algorithm_key].snapshot_dir
            for algorithm_key in ALGORITHM_ORDER
        },
        layer_names=similarity_setup.layer_names,
        metric_names=similarity_setup.plot_metric_names,
        lower_better_metrics=similarity_setup.lower_better_metrics,
        activation_collector=lambda model, loader: collect_model_activations(
            model,
            loader,
            similarity_setup.layer_names,
            runtime.device,
            max_batches=trajectory_config.max_batches_for_similarity,
            max_activation_samples=trajectory_config.max_activation_samples,
            activation_subsample_seed=trajectory_config.activation_subsample_seed,
        ),
        reference_activation_preparer=lambda acts: prepare_activations_for_evaluation(
            acts,
            layers=similarity_setup.layer_names,
            metrics=similarity_setup.metrics,
            max_activation_samples=None,
            subsample_seed=trajectory_config.activation_subsample_seed,
            precompute_metric_reference_cache=True,
        ),
        snapshot_activation_preparer=lambda acts: prepare_activations_for_evaluation(
            acts,
            layers=similarity_setup.layer_names,
            metrics=similarity_setup.metrics,
            max_activation_samples=None,
            subsample_seed=trajectory_config.activation_subsample_seed,
            precompute_metric_reference_cache=False,
        ),
        pair_evaluator=lambda prepared_acts_a, prepared_reference_acts, _similarity_log_prefix=None, metric_timing_seconds=None: evaluate_pair_rows_prepared(
            prepared_acts_a,
            prepared_reference_acts,
            layers=similarity_setup.layer_names,
            metrics=similarity_setup.metrics,
            metric_timing_seconds=metric_timing_seconds,
        ),
        wandb_run=wandb_run,
        wandb_module=wandb_module,
    )
    return artifacts, wandb_run


def save_notebook_combined_comparison(
    runtime: NotebookRuntime,
    wandb_module=None,
    wandb_project: str = "gradient-ascent",
    wandb_name: str = "unlearning-algorithm-comparison",
) -> str:
    """Build the combined cross-algorithm similarity + MIA figure."""
    wandb_run = ensure_wandb_run(wandb_module, project=wandb_project, name=wandb_name)
    return save_combined_trajectory_comparison(
        CombinedComparisonConfig(out_dir=runtime.out_dir),
        wandb_run=wandb_run,
        wandb_module=wandb_module,
    )


def run_multitarget_averaged_experiment(
    runtime: NotebookRuntime,
    similarity_setup: SimilaritySetup,
    *,
    target_labels: Optional[list[int]] = None,
    out_dir: Optional[str] = None,
    run_similarity_stage: bool = True,
    wandb_module=None,
    reuse_existing_checkpoints: bool = False,
    reuse_original_checkpoint: Optional[bool] = None,
    reuse_retrained_checkpoint: Optional[bool] = None,
    reuse_unlearned_checkpoints: Optional[bool] = None,
    reuse_trajectory_outputs: bool = True,
    shared_original_checkpoint_path: Optional[str] = None,
) -> MultiTargetAggregateArtifacts:
    """Run forget-label sweeps (default 0..9) and build averaged utility/similarity plots.

    Multi-target runs are forced to use one shared original checkpoint. If the
    shared checkpoint is missing, this function raises instead of retraining an
    original model.
    """
    labels = target_labels or list(range(runtime.num_classes))
    aggregate_out_dir = out_dir or os.path.join(runtime.out_dir, "multitarget_aggregate")
    os.makedirs(aggregate_out_dir, exist_ok=True)
    shared_dir = os.path.join(aggregate_out_dir, "shared")
    os.makedirs(shared_dir, exist_ok=True)

    candidate_shared_paths = [
        shared_original_checkpoint_path,
        os.path.join(shared_dir, "original_net.pt"),
        os.path.join(runtime.out_dir, "original_net.pt"),
    ]
    resolved_shared_original = next(
        (path for path in candidate_shared_paths if path is not None and os.path.exists(path)),
        None,
    )
    if resolved_shared_original is None:
        raise FileNotFoundError(
            "Shared original checkpoint not found. Expected one of: "
            f"{[path for path in candidate_shared_paths if path is not None]}"
        )

    canonical_shared_original = os.path.join(shared_dir, "original_net.pt")
    if os.path.abspath(resolved_shared_original) != os.path.abspath(canonical_shared_original):
        if not os.path.exists(canonical_shared_original):
            shutil.copy2(resolved_shared_original, canonical_shared_original)
            print(
                "[MultiTarget] Copied shared original checkpoint to "
                f"{canonical_shared_original}"
            )
        resolved_shared_original = canonical_shared_original

    core_by_target: dict[int, CoreExperimentArtifacts] = {}
    for target_label in labels:
        target_out_dir = os.path.join(aggregate_out_dir, f"target_{target_label}")
        target_original_path = os.path.join(target_out_dir, "original_net.pt")
        os.makedirs(target_out_dir, exist_ok=True)
        if not os.path.exists(target_original_path):
            shutil.copy2(resolved_shared_original, target_original_path)
            print(
                "[MultiTarget] Copied shared original for target="
                f"{target_label} -> {target_original_path}"
            )

        target_core_config = replace(
            runtime.core_config,
            out_dir=target_out_dir,
            target_label=int(target_label),
        )
        wandb_run = ensure_wandb_run(
            wandb_module,
            project="gradient-ascent",
            name=f"core-target-{target_label}",
            config=build_core_wandb_config(target_core_config),
        )
        core_by_target[int(target_label)] = run_core_checkpoints(
            model_factory=runtime.model_factory,
            trainset=runtime.trainset,
            testset=runtime.testset,
            device=runtime.device,
            use_cuda=runtime.use_cuda,
            num_workers=runtime.num_workers,
            config=target_core_config,
            reuse_existing_checkpoints=reuse_existing_checkpoints,
            # Never retrain originals in multi-target mode.
            reuse_original_checkpoint=True,
            reuse_retrained_checkpoint=reuse_retrained_checkpoint,
            reuse_unlearned_checkpoints=reuse_unlearned_checkpoints,
            wandb_run=wandb_run,
            wandb_module=wandb_module,
        )

    if run_similarity_stage:
        for target_label in labels:
            target_out_dir = os.path.join(aggregate_out_dir, f"target_{target_label}")
            if reuse_trajectory_outputs:
                expected_similarity_csvs = [
                    os.path.join(
                        target_out_dir,
                        f"similarity_vs_unlearning_epoch_{algorithm_key}_vs_{reference_key}.csv",
                    )
                    for algorithm_key in ALGORITHM_ORDER
                    for reference_key in SIMILARITY_REFERENCES
                ]
                expected_mia_csvs = [
                    os.path.join(target_out_dir, f"mia_vs_unlearning_epoch_{algorithm_key}.csv")
                    for algorithm_key in ALGORITHM_ORDER
                ]
                expected_baseline_csv = os.path.join(target_out_dir, "mia_retrained_baseline.csv")
                if all(
                    os.path.exists(path)
                    for path in [*expected_similarity_csvs, *expected_mia_csvs, expected_baseline_csv]
                ):
                    print(
                        "[MultiTarget] Reusing existing trajectory outputs for "
                        f"target={target_label} at {target_out_dir}"
                    )
                    continue
            trajectory_config = replace(
                build_default_trajectory_config(runtime),
                out_dir=target_out_dir,
                target_label=int(target_label),
                retain_control_label=int((int(target_label) + 1) % runtime.num_classes),
            )
            similarity_setup_for_target = _similarity_setup_with_trajectory_cca(similarity_setup, trajectory_config)
            wandb_run = ensure_wandb_run(
                wandb_module,
                project="gradient-ascent",
                name=f"trajectory-target-{target_label}",
            )
            _ = run_trajectory_analysis(
                model_factory=runtime.model_factory,
                trainset=runtime.trainset,
                testset=runtime.testset,
                device=runtime.device,
                use_cuda=runtime.use_cuda,
                num_workers=runtime.num_workers,
                config=trajectory_config,
                original_checkpoint_path=f"{target_out_dir}/original_net.pt",
                retrained_checkpoint_path=f"{target_out_dir}/retrained_from_scratch_net.pt",
                snapshot_dirs={
                    algorithm_key: core_by_target[int(target_label)].algorithm_artifacts[algorithm_key].snapshot_dir
                    for algorithm_key in ALGORITHM_ORDER
                },
                layer_names=similarity_setup_for_target.layer_names,
                metric_names=similarity_setup_for_target.plot_metric_names,
                lower_better_metrics=similarity_setup_for_target.lower_better_metrics,
                activation_collector=lambda model, loader: collect_model_activations(
                    model,
                    loader,
                    similarity_setup_for_target.layer_names,
                    runtime.device,
                    max_batches=trajectory_config.max_batches_for_similarity,
                    max_activation_samples=trajectory_config.max_activation_samples,
                    activation_subsample_seed=trajectory_config.activation_subsample_seed,
                ),
                reference_activation_preparer=lambda acts: prepare_activations_for_evaluation(
                    acts,
                    layers=similarity_setup_for_target.layer_names,
                    metrics=similarity_setup_for_target.metrics,
                    max_activation_samples=None,
                    subsample_seed=trajectory_config.activation_subsample_seed,
                    precompute_metric_reference_cache=True,
                ),
                snapshot_activation_preparer=lambda acts: prepare_activations_for_evaluation(
                    acts,
                    layers=similarity_setup_for_target.layer_names,
                    metrics=similarity_setup_for_target.metrics,
                    max_activation_samples=None,
                    subsample_seed=trajectory_config.activation_subsample_seed,
                    precompute_metric_reference_cache=False,
                ),
                pair_evaluator=lambda prepared_acts_a, prepared_reference_acts, _similarity_log_prefix=None, metric_timing_seconds=None: evaluate_pair_rows_prepared(
                    prepared_acts_a,
                    prepared_reference_acts,
                    layers=similarity_setup_for_target.layer_names,
                    metrics=similarity_setup_for_target.metrics,
                    metric_timing_seconds=metric_timing_seconds,
                ),
                wandb_run=wandb_run,
                wandb_module=wandb_module,
            )

    utility_csv_paths: dict[str, str] = {}
    utility_relative_plot_paths: dict[str, str] = {}
    utility_absolute_plot_paths: dict[str, str] = {}
    utility_final_bar_plot_paths: dict[str, str] = {}
    for algorithm_key in ALGORITHM_ORDER:
        classwise_csvs = {
            int(target_label): core_by_target[int(target_label)].algorithm_artifacts[algorithm_key].classwise_history_csv_path
            for target_label in labels
        }
        epochs, forget_abs, retain_abs, forget_rel, retain_rel = _aggregate_binary_unlearning_curves(classwise_csvs)

        csv_path = os.path.join(aggregate_out_dir, f"classwise_binary_aggregate_{algorithm_key}.csv")
        with open(csv_path, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "epoch",
                    "forget_abs_acc",
                    "retain_abs_acc",
                    "forget_rel_change_pct",
                    "retain_rel_change_pct",
                ],
            )
            writer.writeheader()
            for idx, epoch in enumerate(epochs):
                writer.writerow(
                    {
                        "epoch": int(epoch),
                        "forget_abs_acc": float(forget_abs[idx]),
                        "retain_abs_acc": float(retain_abs[idx]),
                        "forget_rel_change_pct": float(forget_rel[idx]),
                        "retain_rel_change_pct": float(retain_rel[idx]),
                    }
                )
        utility_csv_paths[algorithm_key] = csv_path

        utility_relative_plot_paths[algorithm_key] = _line_plot_two_series(
            epochs,
            forget_rel,
            retain_rel,
            out_path=os.path.join(aggregate_out_dir, f"classwise_binary_relative_change_{algorithm_key}.png"),
            title=f"Relative change during unlearning ({algorithm_key.upper()}, mean over forget labels 0-9)",
            ylabel="Accuracy change (%)",
        )
        utility_absolute_plot_paths[algorithm_key] = _line_plot_two_series(
            epochs,
            100.0 * forget_abs,
            100.0 * retain_abs,
            out_path=os.path.join(aggregate_out_dir, f"classwise_binary_absolute_accuracy_{algorithm_key}.png"),
            title=f"Absolute accuracy during unlearning ({algorithm_key.upper()}, mean over forget labels 0-9)",
            ylabel="Accuracy (%)",
        )
        utility_final_bar_plot_paths[algorithm_key] = _final_bar_two_series(
            float(forget_rel[-1]),
            float(retain_rel[-1]),
            out_path=os.path.join(aggregate_out_dir, f"classwise_binary_final_change_bar_{algorithm_key}.png"),
            title=f"Final relative change ({algorithm_key.upper()}, mean over forget labels 0-9)",
            ylabel="Accuracy change (%)",
        )

    similarity_csv_paths: dict[str, dict[str, str]] = {algo: {} for algo in ALGORITHM_ORDER}
    similarity_summary_plot_paths: dict[str, dict[str, str]] = {algo: {} for algo in ALGORITHM_ORDER}
    similarity_evolving_bar_paths: dict[str, dict[str, str]] = {algo: {} for algo in ALGORITHM_ORDER}
    similarity_grouped_evolving_bar_paths: dict[str, dict[str, str]] = {algo: {} for algo in ALGORITHM_ORDER}
    similarity_before_after_plot_paths: dict[str, dict[str, str]] = {algo: {} for algo in ALGORITHM_ORDER}
    mia_csv_paths: dict[str, str] = {}
    mia_grid_plot_paths: dict[str, str] = {}
    mia_control_plot_paths: dict[str, str] = {}
    mia_baseline_csv_path: Optional[str] = None

    if run_similarity_stage:
        avg_mia_baseline = _average_mia_baseline_csvs(
            [
                os.path.join(
                    aggregate_out_dir,
                    f"target_{target_label}",
                    "mia_retrained_baseline.csv",
                )
                for target_label in labels
            ]
        )
        mia_baseline_csv_path = save_mia_baseline_csv(
            avg_mia_baseline,
            os.path.join(aggregate_out_dir, "mia_retrained_baseline_mean_over_targets.csv"),
        )

        for algorithm_key in ALGORITHM_ORDER:
            avg_mia_rows = _average_mia_rows_csvs(
                [
                    os.path.join(
                        aggregate_out_dir,
                        f"target_{target_label}",
                        f"mia_vs_unlearning_epoch_{algorithm_key}.csv",
                    )
                    for target_label in labels
                ]
            )
            mia_csv_paths[algorithm_key] = save_mia_trajectory_csv(
                avg_mia_rows,
                os.path.join(aggregate_out_dir, f"mia_vs_unlearning_epoch_{algorithm_key}_mean_over_targets.csv"),
            )
            if "forget_mlp_auc" in avg_mia_rows[0]:
                panel_metrics = [
                    ("forget_logreg_auc", "Forget LogReg AUC (mean over forget labels)"),
                    ("forget_logreg_advantage", "Forget LogReg advantage (mean over forget labels)"),
                    ("forget_mlp_auc", "Forget MLP AUC (mean over forget labels)"),
                    ("forget_mlp_advantage", "Forget MLP advantage (mean over forget labels)"),
                ]
            else:
                panel_metrics = [
                    ("forget_logreg_auc", "Forget LogReg AUC (mean over forget labels)"),
                    ("forget_logreg_advantage", "Forget LogReg advantage (mean over forget labels)"),
                    ("forget_loss_auc", "Forget loss-threshold AUC (mean over forget labels)"),
                    ("forget_logreg_mean_member_prob", "Forget LogReg member prob (mean over forget labels)"),
                ]
            mia_grid_plot_paths[algorithm_key] = save_mia_metric_grid_plot(
                avg_mia_rows,
                os.path.join(aggregate_out_dir, f"mia_frog_trajectory_{algorithm_key}_mean_over_targets.png"),
                f"{algorithm_key.upper()} (mean over forget labels)",
                panel_metrics,
                baseline=avg_mia_baseline,
            )
            mia_control_plot_paths[algorithm_key] = _save_mia_forget_vs_retain_mean_plot(
                avg_mia_rows,
                os.path.join(
                    aggregate_out_dir,
                    f"mia_forget_vs_retain_logreg_mean_member_prob_{algorithm_key}_mean_over_targets.png",
                ),
                f"{algorithm_key.upper()} (mean over forget labels)",
            )

        for algorithm_key in ALGORITHM_ORDER:
            for reference_key in SIMILARITY_REFERENCES:
                csv_paths = [
                    os.path.join(
                        aggregate_out_dir,
                        f"target_{target_label}",
                        f"similarity_vs_unlearning_epoch_{algorithm_key}_vs_{reference_key}.csv",
                    )
                    for target_label in labels
                ]
                avg_epoch_rows = _average_similarity_csvs(csv_paths)

                csv_out = os.path.join(
                    aggregate_out_dir,
                    f"similarity_vs_unlearning_epoch_{algorithm_key}_vs_{reference_key}_mean_over_targets.csv",
                )
                summary_out = os.path.join(
                    aggregate_out_dir,
                    f"similarity_vs_unlearning_epoch_{algorithm_key}_vs_{reference_key}_mean_over_targets_summary.png",
                )
                evolving_out = os.path.join(
                    aggregate_out_dir,
                    f"similarity_vs_unlearning_epoch_{algorithm_key}_vs_{reference_key}_mean_over_targets_evolving_bars.gif",
                )
                grouped_out = os.path.join(
                    aggregate_out_dir,
                    f"similarity_vs_unlearning_epoch_{algorithm_key}_vs_{reference_key}_mean_over_targets_grouped_evolving_bars.gif",
                )
                before_after_out = os.path.join(
                    aggregate_out_dir,
                    f"similarity_vs_unlearning_epoch_{algorithm_key}_vs_{reference_key}_mean_over_targets_before_after.png",
                )

                save_similarity_trajectory_csv(avg_epoch_rows, csv_out, similarity_setup.plot_metric_names)
                save_metric_summary_plot(
                    avg_epoch_rows,
                    summary_out,
                    f"{algorithm_key.upper()} vs {reference_key.capitalize()} (mean over forget labels 0-9)",
                    similarity_setup.plot_metric_names,
                    similarity_setup.lower_better_metrics,
                )
                save_similarity_evolving_bar_plot(
                    avg_epoch_rows,
                    evolving_out,
                    algorithm_key=f"{algorithm_key}_avg",
                    reference_key=reference_key,
                    metric_names=similarity_setup.plot_metric_names,
                    lower_better_metrics=similarity_setup.lower_better_metrics,
                )
                save_similarity_evolving_grouped_bar_plot(
                    avg_epoch_rows,
                    grouped_out,
                    algorithm_key=f"{algorithm_key}_avg",
                    reference_key=reference_key,
                    layer_names=similarity_setup.layer_names,
                    metric_names=similarity_setup.plot_metric_names,
                    lower_better_metrics=similarity_setup.lower_better_metrics,
                )
                save_similarity_before_after_grouped_bar_plot(
                    avg_epoch_rows,
                    before_after_out,
                    algorithm_key=f"{algorithm_key}_avg",
                    reference_key=reference_key,
                    layer_names=similarity_setup.layer_names,
                    metric_names=similarity_setup.plot_metric_names,
                    lower_better_metrics=similarity_setup.lower_better_metrics,
                )

                similarity_csv_paths[algorithm_key][reference_key] = csv_out
                similarity_summary_plot_paths[algorithm_key][reference_key] = summary_out
                similarity_evolving_bar_paths[algorithm_key][reference_key] = evolving_out
                similarity_grouped_evolving_bar_paths[algorithm_key][reference_key] = grouped_out
                similarity_before_after_plot_paths[algorithm_key][reference_key] = before_after_out

    return MultiTargetAggregateArtifacts(
        utility_csv_paths=utility_csv_paths,
        utility_relative_plot_paths=utility_relative_plot_paths,
        utility_absolute_plot_paths=utility_absolute_plot_paths,
        utility_final_bar_plot_paths=utility_final_bar_plot_paths,
        similarity_csv_paths=similarity_csv_paths,
        similarity_summary_plot_paths=similarity_summary_plot_paths,
        similarity_evolving_bar_paths=similarity_evolving_bar_paths,
        similarity_grouped_evolving_bar_paths=similarity_grouped_evolving_bar_paths,
        similarity_before_after_plot_paths=similarity_before_after_plot_paths,
        mia_csv_paths=mia_csv_paths,
        mia_grid_plot_paths=mia_grid_plot_paths,
        mia_control_plot_paths=mia_control_plot_paths,
        mia_baseline_csv_path=mia_baseline_csv_path,
    )


def run_and_display_notebook_core_pipeline(
    runtime: NotebookRuntime,
    wandb_module=None,
    reuse_existing_checkpoints: bool = False,
    reuse_original_checkpoint: Optional[bool] = None,
    reuse_retrained_checkpoint: Optional[bool] = None,
    reuse_unlearned_checkpoints: Optional[bool] = None,
    run_diagnostics: bool = True,
):
    """Run the full core experiment and render notebook outputs inline.

    This helper keeps notebook cells compact by wrapping:
    - core checkpoint/original/retrained training,
    - all five unlearning baselines,
    - runtime and summary prints,
    - optional GA/SCRUB diagnostic exports and plots.
    """
    from IPython.display import Image as IPyImage, display

    core_artifacts, wandb_run = run_notebook_core_experiment(
        runtime,
        wandb_module=wandb_module,
        reuse_existing_checkpoints=reuse_existing_checkpoints,
        reuse_original_checkpoint=reuse_original_checkpoint,
        reuse_retrained_checkpoint=reuse_retrained_checkpoint,
        reuse_unlearned_checkpoints=reuse_unlearned_checkpoints,
    )

    display(IPyImage(filename=core_artifacts.original_vs_retrain_plot_path))
    display(IPyImage(filename=core_artifacts.original_classwise_plot_path))
    display(IPyImage(filename=core_artifacts.retrained_classwise_plot_path))
    display(IPyImage(filename=core_artifacts.retrained_vs_original_percent_diff_plot_path))
    for algorithm_key in ALGORITHM_ORDER:
        artifact = core_artifacts.algorithm_artifacts[algorithm_key]
        display(IPyImage(filename=artifact.classwise_percent_plot_path))
        display(IPyImage(filename=artifact.classwise_absolute_plot_path))

    for key, value in core_artifacts.summary_metrics.items():
        print(f"{key}: {value:.3f}")

    print(f"Saved unlearning runtime CSV to {core_artifacts.unlearning_runtime_csv_path}")
    print(f"Saved unlearning runtime plot to {core_artifacts.unlearning_runtime_plot_path}")
    display(IPyImage(filename=core_artifacts.unlearning_runtime_plot_path))

    if run_diagnostics:
        ga_csv_path, ga_plot_path = save_ga_diagnostics(
            runtime,
            core_artifacts,
            wandb_run=wandb_run,
            wandb_module=wandb_module,
        )
        print(f"Saved GA diagnostics CSV to {ga_csv_path}")
        print(f"Saved GA diagnostics plot to {ga_plot_path}")
        display(IPyImage(filename=ga_plot_path))

        scrub_csv_path, scrub_plot_path = save_scrub_diagnostics(
            runtime,
            core_artifacts,
            wandb_run=wandb_run,
            wandb_module=wandb_module,
        )
        print(f"Saved SCRUB diagnostics CSV to {scrub_csv_path}")
        print(f"Saved SCRUB diagnostics plot to {scrub_plot_path}")
        display(IPyImage(filename=scrub_plot_path))
    else:
        print("Skipped GA/SCRUB diagnostics (run_diagnostics=False).")

    return core_artifacts, wandb_run


def run_and_display_notebook_trajectory_pipeline(
    runtime: NotebookRuntime,
    core_artifacts: CoreExperimentArtifacts,
    similarity_setup: SimilaritySetup,
    wandb_module=None,
):
    """Run trajectory/MIA/similarity analysis and display all generated figures."""
    from IPython.display import Image as IPyImage, clear_output, display
    import ipywidgets as widgets

    trajectory_artifacts, trajectory_wandb_run = run_notebook_trajectory_experiment(
        runtime,
        core_artifacts,
        similarity_setup,
        wandb_module=wandb_module,
    )

    for algorithm_key in ALGORITHM_ORDER:
        mia_artifact = trajectory_artifacts.mia_artifacts[algorithm_key]
        display(IPyImage(filename=mia_artifact.grid_plot_path))
        display(IPyImage(filename=mia_artifact.control_plot_path))
        display(IPyImage(filename=trajectory_artifacts.similarity_metric_timing_plot_paths[algorithm_key]))
        for reference_key in ["retrained", "original"]:
            similarity_artifact = trajectory_artifacts.similarity_artifacts[algorithm_key][reference_key]
            hide_gifs_checkbox = widgets.Checkbox(
                value=False,
                description=f"Hide GIFs ({algorithm_key.upper()} vs {reference_key})",
                indent=False,
            )
            gif_out = widgets.Output()

            def _render_gifs(*_args):
                with gif_out:
                    clear_output(wait=True)
                    if not bool(hide_gifs_checkbox.value):
                        display(IPyImage(filename=similarity_artifact.before_after_grouped_plot_path))
                        display(IPyImage(filename=similarity_artifact.evolving_bar_plot_path))
                        display(IPyImage(filename=similarity_artifact.grouped_evolving_bar_plot_path))

            hide_gifs_checkbox.observe(_render_gifs, names="value")
            _render_gifs()
            display(widgets.VBox([hide_gifs_checkbox, gif_out]))
            display_similarity_epoch_slider(
                similarity_artifact.csv_path,
                algorithm_key=algorithm_key,
                reference_key=reference_key,
                layer_names=similarity_setup.layer_names,
                metric_names=similarity_setup.plot_metric_names,
                lower_better_metrics=similarity_setup.lower_better_metrics,
            )

    print(f"Saved trajectory timing CSV to {trajectory_artifacts.timing_csv_path}")
    return trajectory_artifacts, trajectory_wandb_run


def run_and_display_notebook_combined_comparison(runtime: NotebookRuntime, wandb_module=None) -> str:
    """Generate and display the combined cross-algorithm comparison figure."""
    from IPython.display import Image as IPyImage, display

    combined_path = save_notebook_combined_comparison(runtime, wandb_module=wandb_module)
    print(f"Saved integrated comparison figure to {combined_path}")
    display(IPyImage(filename=combined_path))
    return combined_path


def _compute_mean_forget_loss_and_grad_norm(model, loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_samples = 0
    grad_norms = []
    params = [param for param in model.parameters() if param.requires_grad]

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        model.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)

        sq_norm = 0.0
        for grad in grads:
            sq_norm += grad.detach().float().pow(2).sum().item()
        grad_norms.append(math.sqrt(sq_norm))

        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return {
        "forget_loss": total_loss / max(total_samples, 1),
        "forget_grad_norm": sum(grad_norms) / max(len(grad_norms), 1),
    }


def save_ga_diagnostics(
    runtime: NotebookRuntime,
    core_artifacts: CoreExperimentArtifacts,
    wandb_run=None,
    wandb_module=None,
) -> tuple[str, str]:
    forget_subset, _ = make_forget_retain_subsets(runtime.trainset, runtime.core_config.target_label)
    diagnostic_loader = make_loader(
        forget_subset,
        batch_size=min(256, runtime.core_config.unlearn_batch_size or 256),
        shuffle=False,
        num_workers=runtime.num_workers,
        use_cuda=runtime.use_cuda,
    )

    snapshot_dir = Path(core_artifacts.algorithm_artifacts["ga"].snapshot_dir)
    snapshot_paths = sorted(snapshot_dir.glob("epoch_*.pt"))
    if not snapshot_paths:
        raise RuntimeError(f"No GA snapshots found under {snapshot_dir}")

    rows = []
    for snapshot_path in snapshot_paths:
        epoch = int(snapshot_path.stem.split("_")[-1])
        model = runtime.model_factory()
        model.load_state_dict(torch.load(snapshot_path, map_location=runtime.device))
        diagnostics = _compute_mean_forget_loss_and_grad_norm(model, diagnostic_loader, runtime.device)
        rows.append({"epoch": epoch, **diagnostics})

    csv_path = Path(runtime.out_dir) / "ga_diagnostics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "forget_loss", "forget_grad_norm"])
        writer.writeheader()
        writer.writerows(rows)

    plot_path = Path(runtime.out_dir) / "ga_diagnostics.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    epochs = [row["epoch"] for row in rows]
    axes[0].plot(epochs, [row["forget_loss"] for row in rows], marker="o", linewidth=2)
    axes[0].set_title("GA forget-set loss")
    axes[0].set_xlabel("Unlearning step")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [row["forget_grad_norm"] for row in rows], marker="o", linewidth=2)
    axes[1].set_title("GA forget-set gradient norm")
    axes[1].set_xlabel("Unlearning step")
    axes[1].set_ylabel("L2 norm")
    axes[1].grid(alpha=0.3)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    if wandb_run is not None and wandb_module is not None:
        wandb_run.log(
            {
                "plots/ga_diagnostics": wandb_module.Image(str(plot_path)),
                "tables/ga_diagnostics": wandb_module.Table(
                    data=[[row["epoch"], row["forget_loss"], row["forget_grad_norm"]] for row in rows],
                    columns=["epoch", "forget_loss", "forget_grad_norm"],
                ),
            }
        )

    return str(csv_path), str(plot_path)


def _temperature_kl(student_logits, teacher_logits, temperature):
    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits.float() / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="sum") * (temperature ** 2)


@torch.inference_mode()
def _compute_scrub_snapshot_diagnostics(student_model, teacher_model, forget_loader, retain_loader, device, temperature):
    student_model.eval()
    teacher_model.eval()

    forget_kl_sum = 0.0
    forget_correct = 0
    forget_total = 0
    for inputs, labels in forget_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        student_logits = student_model(inputs)
        teacher_logits = teacher_model(inputs)
        forget_kl_sum += float(_temperature_kl(student_logits, teacher_logits, temperature).item())
        forget_correct += int((student_logits.argmax(dim=1) == labels).sum().item())
        forget_total += int(labels.numel())

    retain_kl_sum = 0.0
    retain_ce_sum = 0.0
    retain_correct = 0
    retain_total = 0
    for inputs, labels in retain_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        student_logits = student_model(inputs)
        teacher_logits = teacher_model(inputs)
        retain_kl_sum += float(_temperature_kl(student_logits, teacher_logits, temperature).item())
        retain_ce_sum += float(F.cross_entropy(student_logits.float(), labels, reduction="sum").item())
        retain_correct += int((student_logits.argmax(dim=1) == labels).sum().item())
        retain_total += int(labels.numel())

    return {
        "forget_kl": forget_kl_sum / max(forget_total, 1),
        "retain_kl": retain_kl_sum / max(retain_total, 1),
        "retain_ce": retain_ce_sum / max(retain_total, 1),
        "forget_acc": forget_correct / max(forget_total, 1),
        "retain_acc": retain_correct / max(retain_total, 1),
    }


def save_scrub_diagnostics(
    runtime: NotebookRuntime,
    core_artifacts: CoreExperimentArtifacts,
    wandb_run=None,
    wandb_module=None,
) -> tuple[str, str]:
    forget_subset, _ = make_forget_retain_subsets(runtime.trainset, runtime.core_config.target_label)
    forget_loader = make_loader(
        forget_subset,
        batch_size=min(256, runtime.core_config.unlearn_batch_size or 256),
        shuffle=False,
        num_workers=runtime.num_workers,
        use_cuda=runtime.use_cuda,
    )

    retain_subset = subset_for_class(runtime.trainset, runtime.core_config.target_label, include=False)
    retain_loader = make_loader(
        retain_subset,
        batch_size=min(
            256,
            runtime.core_config.scrub_config.batch_size or runtime.core_config.unlearn_batch_size or 256,
        ),
        shuffle=False,
        num_workers=runtime.num_workers,
        use_cuda=runtime.use_cuda,
    )

    teacher_model = runtime.model_factory()
    teacher_model.load_state_dict(torch.load(core_artifacts.original_checkpoint_path, map_location=runtime.device))
    teacher_model.eval()

    snapshot_dir = Path(core_artifacts.algorithm_artifacts["scrub"].snapshot_dir)
    snapshot_paths = sorted(snapshot_dir.glob("epoch_*.pt"))
    if not snapshot_paths:
        raise RuntimeError(f"No SCRUB snapshots found under {snapshot_dir}")

    rows = []
    for snapshot_path in snapshot_paths:
        epoch = int(snapshot_path.stem.split("_")[-1])
        student_model = runtime.model_factory()
        student_model.load_state_dict(torch.load(snapshot_path, map_location=runtime.device))
        diagnostics = _compute_scrub_snapshot_diagnostics(
            student_model,
            teacher_model,
            forget_loader,
            retain_loader,
            runtime.device,
            runtime.core_config.scrub_config.temperature,
        )
        rows.append({"epoch": epoch, **diagnostics})

    csv_path = Path(runtime.out_dir) / "scrub_diagnostics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "forget_kl", "retain_kl", "retain_ce", "forget_acc", "retain_acc"],
        )
        writer.writeheader()
        writer.writerows(rows)

    plot_path = Path(runtime.out_dir) / "scrub_diagnostics.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    epochs = [row["epoch"] for row in rows]

    phase_boundary = runtime.core_config.scrub_config.forget_phase_epochs + 0.5

    axes[0].plot(
        epochs,
        [100.0 * row["retain_acc"] for row in rows],
        marker="o",
        linewidth=2,
        label="retain accuracy",
    )
    axes[0].plot(
        epochs,
        [100.0 * row["forget_acc"] for row in rows],
        marker="s",
        linewidth=2,
        label="forget accuracy",
    )
    axes[0].axvline(phase_boundary, color="black", linestyle="--", linewidth=1, alpha=0.7)
    axes[0].set_title("SCRUB retain vs forget accuracy")
    axes[0].set_xlabel("Unlearning step")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    forget_kl = np.array([row["forget_kl"] for row in rows], dtype=np.float64)
    retain_kl = np.array([row["retain_kl"] for row in rows], dtype=np.float64)
    eps = 1e-8
    axes[1].plot(
        epochs,
        np.maximum(forget_kl, eps),
        marker="o",
        linewidth=2,
        label="forget KL",
    )
    axes[1].plot(
        epochs,
        np.maximum(retain_kl, eps),
        marker="s",
        linewidth=2,
        label="retain KL",
    )
    axes[1].axvline(phase_boundary, color="black", linestyle="--", linewidth=1, alpha=0.7)
    axes[1].set_yscale("log")
    axes[1].set_title("SCRUB divergence from teacher")
    axes[1].set_xlabel("Unlearning step")
    axes[1].set_ylabel("KL (log scale)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    if wandb_run is not None and wandb_module is not None:
        wandb_run.log(
            {
                "plots/scrub_diagnostics": wandb_module.Image(str(plot_path)),
                "tables/scrub_diagnostics": wandb_module.Table(
                    data=[
                        [
                            row["epoch"],
                            row["forget_kl"],
                            row["retain_kl"],
                            row["retain_ce"],
                            row["forget_acc"],
                            row["retain_acc"],
                        ]
                        for row in rows
                    ],
                    columns=["epoch", "forget_kl", "retain_kl", "retain_ce", "forget_acc", "retain_acc"],
                ),
            }
        )

    return str(csv_path), str(plot_path)

