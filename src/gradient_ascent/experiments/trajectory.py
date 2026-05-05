from __future__ import annotations

import csv
import os
import time
from collections import defaultdict
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

from ..data import (
    clone_dataset_with_eval_transform,
    make_forget_retain_subsets,
    make_loader,
    sample_class_subsets,
)
from ..mia import compute_mia_baseline, compute_mia_trajectory_rows
from ..reporting import (
    save_metric_summary_plot,
    save_mia_baseline_csv,
    save_mia_control_comparison_plot,
    save_mia_metric_grid_plot,
    save_mia_trajectory_csv,
    save_similarity_trajectory_csv,
)
from ..trajectories import (
    compute_epoch_rows_from_snapshots_multi_reference,
    save_similarity_before_after_grouped_bar_plot,
    save_similarity_evolving_bar_plot,
    save_similarity_evolving_grouped_bar_plot,
    save_similarity_metric_timing_plot,
)

from .types import MIAArtifact, SimilarityArtifact, TrajectoryExperimentArtifacts, TrajectoryExperimentConfig
from .wandb_utils import log_media


def run_trajectory_analysis(
    model_factory: Callable[[], torch.nn.Module],
    trainset,
    testset,
    device: torch.device,
    use_cuda: bool,
    num_workers: int,
    config: TrajectoryExperimentConfig,
    original_checkpoint_path: str,
    retrained_checkpoint_path: str,
    snapshot_dirs: Mapping[str, str],
    layer_names: Sequence[str],
    metric_names: Iterable[str],
    lower_better_metrics: Iterable[str],
    activation_collector: Callable[[torch.nn.Module, object], Dict[str, object]],
    pair_evaluator: Callable[[Dict[str, object], Dict[str, object], Optional[str], Optional[Dict[str, float]]], list[dict]],
    reference_activation_preparer: Optional[Callable[[Dict[str, object]], Dict[str, object]]] = None,
    snapshot_activation_preparer: Optional[Callable[[Dict[str, object]], Dict[str, object]]] = None,
    wandb_run=None,
    wandb_module=None,
) -> TrajectoryExperimentArtifacts:
    overall_t0 = time.perf_counter()
    stage_timing_seconds: Dict[str, float] = {}
    print("[Trajectory] Starting trajectory analysis pipeline...")
    for path in [original_checkpoint_path, retrained_checkpoint_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing checkpoint: {path}")

    t0 = time.perf_counter()
    similarity_dataset = testset
    similarity_label = "test"
    if config.similarity_data_mode == "forget":
        similarity_dataset, _ = make_forget_retain_subsets(testset, config.target_label)
        similarity_label = f"forget(test, class={config.target_label})"
    elif config.similarity_data_mode == "retain":
        _, similarity_dataset = make_forget_retain_subsets(testset, config.target_label)
        similarity_label = f"retain(test, without class={config.target_label})"
    elif config.similarity_data_mode != "test":
        raise ValueError(
            f"Unsupported similarity_data_mode={config.similarity_data_mode!r}; expected one of "
            "('forget', 'retain', 'test')."
        )
    testloader = make_loader(similarity_dataset, config.trajectory_batch_size, False, num_workers, use_cuda)
    stage_timing_seconds["build_test_loader"] = time.perf_counter() - t0
    print(
        "[Trajectory] Built similarity loader "
        f"(source={similarity_label}, batch_size={config.trajectory_batch_size}) "
        f"in {stage_timing_seconds['build_test_loader']:.1f}s"
    )

    t0 = time.perf_counter()
    retrained_model = model_factory()
    retrained_model.load_state_dict(torch.load(retrained_checkpoint_path, map_location=device))
    retrained_model.eval()
    retrained_acts = activation_collector(retrained_model, testloader)
    stage_timing_seconds["collect_retrained_reference_activations"] = time.perf_counter() - t0
    print(
        "[Trajectory] Collected retrained reference activations in "
        f"{stage_timing_seconds['collect_retrained_reference_activations']:.1f}s"
    )

    t0 = time.perf_counter()
    original_model = model_factory()
    original_model.load_state_dict(torch.load(original_checkpoint_path, map_location=device))
    original_model.eval()
    original_acts = activation_collector(original_model, testloader)
    stage_timing_seconds["collect_original_reference_activations"] = time.perf_counter() - t0
    print(
        "[Trajectory] Collected original reference activations in "
        f"{stage_timing_seconds['collect_original_reference_activations']:.1f}s"
    )

    reference_map = {
        "retrained": retrained_acts,
        "original": original_acts,
    }
    if reference_activation_preparer is not None:
        prep_t0 = time.perf_counter()
        reference_map = {
            reference_key: reference_activation_preparer(reference_acts)
            for reference_key, reference_acts in reference_map.items()
        }
        stage_timing_seconds["prepare_reference_activations"] = time.perf_counter() - prep_t0

    similarity_artifacts: Dict[str, Dict[str, SimilarityArtifact]] = {}
    similarity_metric_timing_plot_paths: Dict[str, str] = {}
    metric_names = list(metric_names)
    lower_better_metrics = list(lower_better_metrics)
    for algorithm_key, snapshot_dir in snapshot_dirs.items():
        algo_t0 = time.perf_counter()
        print(f"[Trajectory] Similarity stage for {algorithm_key.upper()}...")
        similarity_artifacts[algorithm_key] = {}
        similarity_log_prefix_map = {
            reference_key: (
                f"[Similarity {algorithm_key.upper()} vs {reference_key.capitalize()}]"
                if config.log_similarity_progress
                else None
            )
            for reference_key in reference_map
        }
        metric_timing_seconds_algo: Dict[str, float] = defaultdict(float)
        scoring_t0 = time.perf_counter()
        epoch_rows_map = compute_epoch_rows_from_snapshots_multi_reference(
            snapshot_dir,
            model_factory,
            reference_map,
            lambda model: activation_collector(model, testloader),
            pair_evaluator,
            device,
            similarity_log_prefix_map=similarity_log_prefix_map,
            activation_cache_dir=(
                os.path.join(config.out_dir, config.similarity_activation_cache_subdir, algorithm_key)
                if config.cache_similarity_activations
                else None
            ),
            activation_preparer=snapshot_activation_preparer or reference_activation_preparer,
            metric_timing_seconds=metric_timing_seconds_algo,
            step_stride=config.similarity_step_stride,
        )
        scoring_elapsed = time.perf_counter() - scoring_t0
        stage_timing_seconds[f"similarity_{algorithm_key}_epoch_scoring"] = scoring_elapsed
        print(
            f"[Trajectory] Similarity epoch scoring for {algorithm_key.upper()} done "
            f"in {scoring_elapsed:.1f}s"
        )

        artifact_total_elapsed = 0.0
        for reference_key in reference_map:
            ref_t0 = time.perf_counter()
            print(f"[Trajectory]   {algorithm_key.upper()} vs {reference_key.capitalize()} starting...")
            epoch_rows = epoch_rows_map[reference_key]
            csv_path = f"{config.out_dir}/similarity_vs_unlearning_epoch_{algorithm_key}_vs_{reference_key}.csv"
            summary_plot_path = (
                f"{config.out_dir}/similarity_vs_unlearning_epoch_{algorithm_key}_vs_{reference_key}_summary.png"
            )
            evolving_bar_plot_path = (
                f"{config.out_dir}/similarity_vs_unlearning_epoch_{algorithm_key}_vs_{reference_key}_evolving_bars.gif"
            )
            grouped_evolving_bar_plot_path = (
                f"{config.out_dir}/similarity_vs_unlearning_epoch_{algorithm_key}_vs_{reference_key}"
                "_evolving_grouped_bars.gif"
            )
            before_after_grouped_plot_path = (
                f"{config.out_dir}/similarity_vs_unlearning_epoch_{algorithm_key}_vs_{reference_key}"
                "_before_after_grouped_bars.png"
            )

            save_similarity_trajectory_csv(epoch_rows, csv_path, metric_names)
            save_metric_summary_plot(
                epoch_rows,
                summary_plot_path,
                f"{algorithm_key.upper()} vs {reference_key.capitalize()} over unlearning steps "
                "(all metrics min–max rescaled to [0, 1] vs this run; 1 = most similar)",
                metric_names,
                lower_better_metrics,
            )
            save_similarity_evolving_bar_plot(
                epoch_rows,
                evolving_bar_plot_path,
                algorithm_key,
                reference_key,
                metric_names,
                lower_better_metrics,
            )
            save_similarity_evolving_grouped_bar_plot(
                epoch_rows,
                grouped_evolving_bar_plot_path,
                algorithm_key,
                reference_key,
                layer_names,
                metric_names,
                lower_better_metrics,
            )
            save_similarity_before_after_grouped_bar_plot(
                epoch_rows,
                before_after_grouped_plot_path,
                algorithm_key,
                reference_key,
                layer_names,
                metric_names,
                lower_better_metrics,
            )
            log_media(
                wandb_run,
                wandb_module,
                f"plots/similarity_vs_unlearning_epoch_{algorithm_key}_vs_{reference_key}_summary",
                summary_plot_path,
            )
            log_media(
                wandb_run,
                wandb_module,
                f"plots/similarity_vs_unlearning_epoch_{algorithm_key}_vs_{reference_key}_evolving_bars",
                evolving_bar_plot_path,
            )
            log_media(
                wandb_run,
                wandb_module,
                f"plots/similarity_vs_unlearning_epoch_{algorithm_key}_vs_{reference_key}_evolving_grouped_bars",
                grouped_evolving_bar_plot_path,
            )
            log_media(
                wandb_run,
                wandb_module,
                f"plots/similarity_vs_unlearning_epoch_{algorithm_key}_vs_{reference_key}_before_after_grouped_bars",
                before_after_grouped_plot_path,
            )
            similarity_artifacts[algorithm_key][reference_key] = SimilarityArtifact(
                csv_path=csv_path,
                summary_plot_path=summary_plot_path,
                evolving_bar_plot_path=evolving_bar_plot_path,
                grouped_evolving_bar_plot_path=grouped_evolving_bar_plot_path,
                before_after_grouped_plot_path=before_after_grouped_plot_path,
            )
            ref_elapsed = time.perf_counter() - ref_t0
            print(
                f"[Trajectory]   {algorithm_key.upper()} vs {reference_key.capitalize()} done "
                f"in {ref_elapsed:.1f}s"
            )
            stage_timing_seconds[f"similarity_{algorithm_key}_vs_{reference_key}"] = ref_elapsed
            stage_timing_seconds[f"similarity_{algorithm_key}_vs_{reference_key}_artifact_io"] = ref_elapsed
            artifact_total_elapsed += ref_elapsed
        stage_timing_seconds[f"similarity_{algorithm_key}_artifact_io_total"] = artifact_total_elapsed
        for metric_name, seconds in metric_timing_seconds_algo.items():
            stage_timing_seconds[f"similarity_{algorithm_key}_metric_{metric_name}_seconds"] = float(seconds)
        metric_timing_plot_path = (
            f"{config.out_dir}/similarity_metric_timing_{algorithm_key}.png"
        )
        save_similarity_metric_timing_plot(
            metric_timing_seconds_algo,
            metric_timing_plot_path,
            algorithm_key,
        )
        similarity_metric_timing_plot_paths[algorithm_key] = metric_timing_plot_path
        log_media(
            wandb_run,
            wandb_module,
            f"plots/similarity_metric_timing_{algorithm_key}",
            metric_timing_plot_path,
        )
        stage_timing_seconds[f"similarity_{algorithm_key}_total"] = time.perf_counter() - algo_t0
        print(
            f"[Trajectory] Similarity stage for {algorithm_key.upper()} finished "
            f"in {stage_timing_seconds[f'similarity_{algorithm_key}_total']:.1f}s "
            f"(epoch scoring {scoring_elapsed:.1f}s + artifact I/O {artifact_total_elapsed:.1f}s)"
        )

    t0 = time.perf_counter()
    forget_subset, _retain_subset = make_forget_retain_subsets(trainset, config.target_label)
    # MIA should probe fixed member/non-member distributions. Use an eval-view of
    # the train dataset so member examples are not randomly augmented each pass.
    mia_trainset = clone_dataset_with_eval_transform(trainset)
    forget_member_subset, forget_nonmember_subset, forget_n_member, forget_n_nonmember = sample_class_subsets(
        mia_trainset,
        testset,
        config.target_label,
        config.mia_seed,
        balance_classes=config.mia_balance_probe_classes,
    )
    retain_member_subset, retain_nonmember_subset, retain_n_member, retain_n_nonmember = sample_class_subsets(
        mia_trainset,
        testset,
        config.retain_control_label,
        config.mia_seed + 100,
        balance_classes=config.mia_balance_probe_classes,
    )
    del forget_subset

    forget_member_loader = make_loader(forget_member_subset, config.trajectory_batch_size, False, num_workers, use_cuda)
    forget_nonmember_loader = make_loader(
        forget_nonmember_subset,
        config.trajectory_batch_size,
        False,
        num_workers,
        use_cuda,
    )
    retain_member_loader = make_loader(retain_member_subset, config.trajectory_batch_size, False, num_workers, use_cuda)
    retain_nonmember_loader = make_loader(
        retain_nonmember_subset,
        config.trajectory_batch_size,
        False,
        num_workers,
        use_cuda,
    )
    print(
        f"MIA probes ready | balanced={config.mia_balance_probe_classes} | "
        f"forget(frog={config.target_label}): {forget_n_member} member + {forget_n_nonmember} non-member | "
        f"retain(label={config.retain_control_label}): {retain_n_member} member + {retain_n_nonmember} non-member"
    )
    stage_timing_seconds["prepare_mia_subsets_and_loaders"] = time.perf_counter() - t0
    print(
        "[Trajectory] Prepared MIA subsets/loaders in "
        f"{stage_timing_seconds['prepare_mia_subsets_and_loaders']:.1f}s"
    )

    t0 = time.perf_counter()
    retrained_mia_model = model_factory()
    retrained_mia_model.load_state_dict(torch.load(retrained_checkpoint_path, map_location=device))
    retrained_mia_model.eval()
    mia_baseline = compute_mia_baseline(
        retrained_mia_model,
        forget_member_loader,
        forget_nonmember_loader,
        retain_member_loader,
        retain_nonmember_loader,
        device,
        seed=config.mia_seed,
        include_mlp_attacker=config.mia_include_mlp_attacker,
        bootstrap_rounds=config.mia_bootstrap_rounds,
        mlp_sweep_enabled=config.mia_mlp_sweep_enabled,
        mlp_sweep_hidden_layer_sizes=config.mia_mlp_sweep_hidden_layer_sizes,
        mlp_sweep_alphas=config.mia_mlp_sweep_alphas,
    )
    mia_baseline_csv_path = save_mia_baseline_csv(mia_baseline, f"{config.out_dir}/mia_retrained_baseline.csv")

    if wandb_run is not None and wandb_module is not None:
        wandb_run.log(
            {
                "tables/mia_retrained_baseline": wandb_module.Table(
                    data=[[key, float(value)] for key, value in mia_baseline.items()],
                    columns=["metric", "value"],
                )
            }
        )
    stage_timing_seconds["compute_retrained_mia_baseline"] = time.perf_counter() - t0
    print(
        "[Trajectory] Computed/logged retrained MIA baseline in "
        f"{stage_timing_seconds['compute_retrained_mia_baseline']:.1f}s"
    )

    mia_panel_metrics = [
        ("forget_logreg_auc", "LogReg AUC (lower = less inferable)"),
        ("forget_logreg_advantage", "LogReg advantage (lower = less inferable)"),
    ]
    if config.mia_include_mlp_attacker:
        mia_panel_metrics += [
            ("forget_mlp_auc", "MLP AUC (lower = less inferable)"),
            ("forget_mlp_advantage", "MLP advantage (lower = less inferable)"),
        ]
    else:
        mia_panel_metrics += [
            ("forget_loss_auc", "Loss-threshold AUC (lower = less inferable)"),
            ("forget_logreg_mean_member_prob", "LogReg mean member prob (lower = less inferable)"),
        ]

    mia_artifacts: Dict[str, MIAArtifact] = {}
    for algorithm_key, snapshot_dir in snapshot_dirs.items():
        mia_t0 = time.perf_counter()
        print(f"[Trajectory] MIA trajectory for {algorithm_key.upper()}...")
        rows = compute_mia_trajectory_rows(
            snapshot_dir,
            model_factory,
            forget_member_loader,
            forget_nonmember_loader,
            retain_member_loader,
            retain_nonmember_loader,
            device,
            seed_base=config.mia_seed,
            map_location=device,
            fixed_cv_across_epochs=config.mia_fixed_cv_across_epochs,
            include_mlp_attacker=config.mia_include_mlp_attacker,
            bootstrap_rounds=config.mia_bootstrap_rounds,
            progress_log_prefix=f"[MIA {algorithm_key.upper()}]",
            mlp_sweep_enabled=config.mia_mlp_sweep_enabled,
            mlp_sweep_hidden_layer_sizes=config.mia_mlp_sweep_hidden_layer_sizes,
            mlp_sweep_alphas=config.mia_mlp_sweep_alphas,
        )
        csv_path = save_mia_trajectory_csv(rows, f"{config.out_dir}/mia_vs_unlearning_epoch_{algorithm_key}.csv")
        grid_plot_path = save_mia_metric_grid_plot(
            rows,
            f"{config.out_dir}/mia_frog_trajectory_{algorithm_key}.png",
            algorithm_key.upper(),
            mia_panel_metrics,
            baseline=mia_baseline,
        )
        control_plot_path = save_mia_control_comparison_plot(
            rows,
            f"{config.out_dir}/mia_forget_vs_retain_logreg_mean_member_prob_{algorithm_key}.png",
            algorithm_key.upper(),
            config.retain_control_label,
        )
        log_media(wandb_run, wandb_module, f"plots/mia_frog_trajectory_{algorithm_key}", grid_plot_path)
        log_media(
            wandb_run,
            wandb_module,
            f"plots/mia_forget_vs_retain_logreg_mean_member_prob_{algorithm_key}",
            control_plot_path,
        )
        mia_artifacts[algorithm_key] = MIAArtifact(
            csv_path=csv_path,
            grid_plot_path=grid_plot_path,
            control_plot_path=control_plot_path,
        )
        print(
            f"[Trajectory] MIA trajectory for {algorithm_key.upper()} finished "
            f"in {time.perf_counter() - mia_t0:.1f}s"
        )
        stage_timing_seconds[f"mia_{algorithm_key}_total"] = time.perf_counter() - mia_t0

    stage_timing_seconds["trajectory_total"] = time.perf_counter() - overall_t0
    timing_csv_path = f"{config.out_dir}/trajectory_timing_seconds.csv"
    with open(timing_csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "seconds"])
        writer.writeheader()
        for stage_name, seconds in sorted(stage_timing_seconds.items()):
            writer.writerow({"stage": stage_name, "seconds": float(seconds)})
    if wandb_run is not None and wandb_module is not None:
        wandb_run.log(
            {
                "tables/trajectory_timing_seconds": wandb_module.Table(
                    data=[[stage, float(seconds)] for stage, seconds in sorted(stage_timing_seconds.items())],
                    columns=["stage", "seconds"],
                )
            }
        )
    print(f"[Trajectory] Full trajectory analysis completed in {stage_timing_seconds['trajectory_total']:.1f}s")
    print(f"[Trajectory] Saved timing CSV to {timing_csv_path}")
    return TrajectoryExperimentArtifacts(
        similarity_artifacts=similarity_artifacts,
        similarity_metric_timing_plot_paths=similarity_metric_timing_plot_paths,
        mia_artifacts=mia_artifacts,
        mia_baseline_csv_path=mia_baseline_csv_path,
        timing_csv_path=timing_csv_path,
        timing_seconds={key: float(value) for key, value in stage_timing_seconds.items()},
    )
