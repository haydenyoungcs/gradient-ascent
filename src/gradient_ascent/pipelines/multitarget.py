"""Forget-label sweep: train/retrain/unlearn per class, then average metrics (dissertation macro view)."""

from __future__ import annotations

import csv
import os
import shutil
import warnings
from dataclasses import dataclass, replace
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ..constants import ALGORITHM_ORDER, SIMILARITY_REFERENCES
from ..correlation import run_epochwise_similarity_outcome_correlation, run_similarity_mia_correlation_analysis
from ..experiments import (
    CoreExperimentArtifacts,
    TrajectoryExperimentConfig,
    ensure_wandb_run,
    run_core_checkpoints,
    run_trajectory_analysis,
)
from ..notebook_runtime import (
    NotebookRuntime,
    SimilaritySetup,
    build_core_wandb_config,
    build_default_trajectory_config,
    similarity_setup_with_trajectory_cca,
)
from ..reporting import (
    save_mia_baseline_csv,
    save_mia_metric_grid_plot,
    save_mia_trajectory_csv,
    save_metric_summary_plot,
    save_similarity_trajectory_csv,
)
from ..similarity import (
    collect_model_activations,
    evaluate_pair_rows_prepared,
    prepare_activations_for_evaluation,
)
from ..trajectories import (
    save_similarity_before_after_grouped_bar_plot,
    save_similarity_evolving_bar_plot,
    save_similarity_evolving_grouped_bar_plot,
)


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
            similarity_setup_for_target = similarity_setup_with_trajectory_cca(similarity_setup, trajectory_config)
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
