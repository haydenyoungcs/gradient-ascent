from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence

import torch

from .data import CIFAR10_CLASSES, make_forget_retain_subsets, make_loader, sample_balanced_class_subsets, subset_for_class
from .mia import compute_mia_baseline, compute_mia_trajectory_rows
from .reporting import (
    load_mean_series,
    load_mia_baseline,
    load_mia_series,
    save_accuracy_history_plot,
    save_classwise_absolute_accuracy_plot,
    save_classwise_history_csv,
    save_classwise_percent_change_plot,
    save_metric_summary_plot,
    save_mia_baseline_csv,
    save_mia_control_comparison_plot,
    save_mia_metric_grid_plot,
    save_mia_trajectory_csv,
    save_similarity_trajectory_csv,
)
from .certified import CertifiedConfig, run_certified_unlearning
from .trajectories import compute_epoch_rows_from_snapshots, save_combined_similarity_mia_plot, save_similarity_heatmap_grid
from .training import evaluate, train_model
from .unlearning import (
    GAConfig,
    SCRUBConfig,
    SSDConfig,
    SalUnConfig,
    run_ga_unlearning,
    run_salun_unlearning,
    run_scrub_unlearning,
    run_ssd_unlearning,
)


DEFAULT_ALGO_KEYS = ("ga", "ssd", "salun", "certified", "scrub")
DEFAULT_ALGO_DISPLAY = {
    "ga": "Gradient Ascent (GA)",
    "ssd": "Selective Synaptic Dampening (SSD)",
    "salun": "SalUn",
    "certified": "Certified Removal",
    "scrub": "SCRUB",
}


@dataclass(frozen=True)
class CoreExperimentConfig:
    num_classes: int = 10
    model_depth: int = 50
    out_dir: str = "out"
    target_label: int = 6
    num_epochs: int = 20
    batch_size: int = 50
    training_lr: float = 0.01
    training_momentum: float = 0.9
    weight_decay: float = 1e-4
    unlearn_batch_size: Optional[int] = None
    ga_config: GAConfig = field(default_factory=GAConfig)
    ssd_config: SSDConfig = field(default_factory=SSDConfig)
    salun_config: SalUnConfig = field(default_factory=SalUnConfig)
    certified_config: CertifiedConfig = field(default_factory=CertifiedConfig)
    scrub_config: SCRUBConfig = field(default_factory=SCRUBConfig)


@dataclass(frozen=True)
class AlgorithmArtifacts:
    snapshot_dir: str
    final_checkpoint_path: str
    classwise_history_csv_path: str
    classwise_percent_plot_path: str
    classwise_absolute_plot_path: str


@dataclass(frozen=True)
class CoreExperimentArtifacts:
    original_checkpoint_path: str
    retrained_checkpoint_path: str
    original_vs_retrain_plot_path: str
    algorithm_artifacts: Dict[str, AlgorithmArtifacts]
    summary_metrics: Dict[str, float]


@dataclass(frozen=True)
class TrajectoryExperimentConfig:
    num_classes: int = 10
    model_depth: int = 50
    out_dir: str = "out"
    target_label: int = 6
    retain_control_label: int = 0
    trajectory_batch_size: int = 256
    max_batches_for_similarity: int = 8
    mia_seed: int = 1337


@dataclass(frozen=True)
class SimilarityArtifact:
    csv_path: str
    summary_plot_path: str
    heatmap_plot_path: str


@dataclass(frozen=True)
class MIAArtifact:
    csv_path: str
    grid_plot_path: str
    control_plot_path: str


@dataclass(frozen=True)
class TrajectoryExperimentArtifacts:
    similarity_artifacts: Dict[str, Dict[str, SimilarityArtifact]]
    mia_artifacts: Dict[str, MIAArtifact]
    mia_baseline_csv_path: str


@dataclass(frozen=True)
class CombinedComparisonConfig:
    out_dir: str = "out"
    algo_keys: Sequence[str] = DEFAULT_ALGO_KEYS
    algo_display: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_ALGO_DISPLAY))
    reference_key: str = "retrained"
    lower_better_metrics: Sequence[str] = ("euclidean", "kl_sym")
    mia_panels: Sequence[tuple[str, str]] = (
        ("forget_loss_auc", "Forget loss AUC (lower = less inferable)"),
        ("forget_logreg_auc", "Forget logreg AUC (lower = less inferable)"),
        ("forget_logreg_tpr_at_1pct", "Forget logreg TPR@1%FPR (lower = less inferable)"),
        ("forget_logreg_mean_member_prob", "Forget logreg mean member prob (lower = less inferable)"),
    )


def ensure_wandb_run(wandb_module, project: str, name: str, config: Optional[Mapping[str, object]] = None):
    if wandb_module is None:
        return None
    if wandb_module.run is None:
        init_kwargs = {"project": project, "name": name}
        if config is not None:
            init_kwargs["config"] = dict(config)
        wandb_module.init(**init_kwargs)
    return wandb_module.run


def _log_media(wandb_run, wandb_module, key: str, path: str) -> None:
    if wandb_run is None or wandb_module is None:
        return
    if path.endswith(".gif"):
        wandb_run.log({key: wandb_module.Video(path, format="gif")})
    else:
        wandb_run.log({key: wandb_module.Image(path)})


def _save_classwise_artifacts(
    algorithm_key: str,
    algorithm_label: str,
    history,
    config: CoreExperimentConfig,
) -> AlgorithmArtifacts:
    history_tensor = torch.as_tensor(history, dtype=torch.float32).cpu().numpy()
    epochs = list(range(history_tensor.shape[0]))
    csv_path = f"{config.out_dir}/classwise_accuracy_{algorithm_key}.csv"
    percent_plot_path = f"{config.out_dir}/classwise_percent_change_{algorithm_key}.png"
    absolute_plot_path = f"{config.out_dir}/classwise_absolute_accuracy_{algorithm_key}.png"

    save_classwise_history_csv(epochs, history_tensor, CIFAR10_CLASSES, csv_path)
    save_classwise_percent_change_plot(
        epochs,
        history_tensor,
        CIFAR10_CLASSES,
        percent_plot_path,
        f"Relative Change in Class Accuracy During Unlearning ({algorithm_label}, ResNet-{config.model_depth})",
    )
    save_classwise_absolute_accuracy_plot(
        epochs,
        history_tensor,
        CIFAR10_CLASSES,
        absolute_plot_path,
        f"Absolute Class Accuracy During Unlearning ({algorithm_label}, ResNet-{config.model_depth})",
    )

    snapshot_dir = f"{config.out_dir}/unlearning_snapshots_{algorithm_key}"
    final_checkpoint_path = f"{config.out_dir}/unlearned_net_{algorithm_key}.pt"
    return AlgorithmArtifacts(
        snapshot_dir=snapshot_dir,
        final_checkpoint_path=final_checkpoint_path,
        classwise_history_csv_path=csv_path,
        classwise_percent_plot_path=percent_plot_path,
        classwise_absolute_plot_path=absolute_plot_path,
    )


def run_core_checkpoints(
    model_factory: Callable[[], torch.nn.Module],
    trainset,
    testset,
    device: torch.device,
    use_cuda: bool,
    num_workers: int,
    config: Optional[CoreExperimentConfig] = None,
    wandb_run=None,
    wandb_module=None,
) -> CoreExperimentArtifacts:
    config = config or CoreExperimentConfig()
    os.makedirs(config.out_dir, exist_ok=True)

    trainloader = make_loader(trainset, config.batch_size, True, num_workers, use_cuda)
    testloader = make_loader(testset, config.batch_size, False, num_workers, use_cuda)

    original_path = f"{config.out_dir}/original_net.pt"
    retrained_path = f"{config.out_dir}/retrained_from_scratch_net.pt"

    original_model = model_factory()
    original_acc_history, _ = train_model(
        original_model,
        trainloader,
        testloader,
        num_epochs=config.num_epochs,
        device=device,
        lr=config.training_lr,
        momentum=config.training_momentum,
        weight_decay=config.weight_decay,
        save_path=original_path,
        log_prefix="original ",
        num_classes=config.num_classes,
        wandb_run=wandb_run,
        wandb_prefix="original",
        wandb_step_offset=0,
    )

    keep_subset = subset_for_class(trainset, config.target_label, include=False)
    keep_loader = make_loader(keep_subset, config.batch_size, True, num_workers, use_cuda)

    retrained_model = model_factory()
    retrained_acc_history, _ = train_model(
        retrained_model,
        keep_loader,
        testloader,
        num_epochs=config.num_epochs,
        device=device,
        lr=config.training_lr,
        momentum=config.training_momentum,
        weight_decay=config.weight_decay,
        save_path=retrained_path,
        log_prefix="retrain ",
        num_classes=config.num_classes,
        wandb_run=wandb_run,
        wandb_prefix="retrain",
        wandb_step_offset=config.num_epochs,
    )

    original_vs_retrain_plot_path = save_accuracy_history_plot(
        {
            "Original (all data)": original_acc_history,
            "Retrained from scratch": retrained_acc_history,
        },
        f"{config.out_dir}/original_vs_retrain_acc.png",
        f"Original and retrained-from-scratch test accuracy vs epoch (ResNet-{config.model_depth})",
    )
    _log_media(wandb_run, wandb_module, "plots/original_vs_retrain", original_vs_retrain_plot_path)

    forget_subset, retain_subset = make_forget_retain_subsets(trainset, config.target_label)
    unlearn_batch_size = config.unlearn_batch_size or 256
    forget_loader = make_loader(forget_subset, unlearn_batch_size, True, num_workers, use_cuda)
    retain_loader = make_loader(retain_subset, unlearn_batch_size, True, num_workers, use_cuda)
    scrub_batch_size = config.scrub_config.batch_size or unlearn_batch_size
    scrub_forget_loader = (
        forget_loader
        if scrub_batch_size == unlearn_batch_size
        else make_loader(forget_subset, scrub_batch_size, True, num_workers, use_cuda)
    )
    scrub_retain_loader = (
        retain_loader
        if scrub_batch_size == unlearn_batch_size
        else make_loader(retain_subset, scrub_batch_size, True, num_workers, use_cuda)
    )

    ga_model = model_factory()
    ga_model.load_state_dict(torch.load(original_path, map_location=device))
    ga_result = run_ga_unlearning(
        ga_model,
        forget_loader,
        testloader,
        device,
        config=config.ga_config,
        num_classes=config.num_classes,
        snapshot_dir=f"{config.out_dir}/unlearning_snapshots_ga",
    )
    torch.save(ga_result["model"].state_dict(), f"{config.out_dir}/unlearned_net.pt")
    torch.save(ga_result["model"].state_dict(), f"{config.out_dir}/unlearned_net_ga.pt")

    ssd_model = model_factory()
    ssd_model.load_state_dict(torch.load(original_path, map_location=device))
    ssd_result = run_ssd_unlearning(
        ssd_model,
        forget_loader,
        retain_loader,
        testloader,
        device,
        config=config.ssd_config,
        num_classes=config.num_classes,
        snapshot_dir=f"{config.out_dir}/unlearning_snapshots_ssd",
    )
    torch.save(ssd_result["model"].state_dict(), f"{config.out_dir}/unlearned_net_ssd.pt")

    salun_model = model_factory()
    salun_model.load_state_dict(torch.load(original_path, map_location=device))
    salun_result = run_salun_unlearning(
        salun_model,
        forget_loader,
        retain_loader,
        testloader,
        device,
        config=config.salun_config,
        num_classes=config.num_classes,
        snapshot_dir=f"{config.out_dir}/unlearning_snapshots_salun",
    )
    torch.save(salun_result["model"].state_dict(), f"{config.out_dir}/unlearned_net_salun.pt")

    certified_model = model_factory()
    certified_model.load_state_dict(torch.load(original_path, map_location=device))
    certified_result = run_certified_unlearning(
        certified_model,
        forget_loader,
        retain_loader,
        testloader,
        device,
        config=config.certified_config,
        num_classes=config.num_classes,
        snapshot_dir=f"{config.out_dir}/unlearning_snapshots_certified",
    )
    torch.save(certified_result["model"].state_dict(), f"{config.out_dir}/unlearned_net_certified.pt")

    scrub_model = model_factory()
    scrub_model.load_state_dict(torch.load(original_path, map_location=device))
    scrub_result = run_scrub_unlearning(
        scrub_model,
        scrub_forget_loader,
        scrub_retain_loader,
        testloader,
        device,
        config=config.scrub_config,
        num_classes=config.num_classes,
        snapshot_dir=f"{config.out_dir}/unlearning_snapshots_scrub",
    )
    torch.save(scrub_result["model"].state_dict(), f"{config.out_dir}/unlearned_net_scrub.pt")

    algorithm_artifacts = {
        "ga": _save_classwise_artifacts("ga", "GA", ga_result["classwise_history"], config),
        "ssd": _save_classwise_artifacts("ssd", "SSD", ssd_result["classwise_history"], config),
        "salun": _save_classwise_artifacts("salun", "SalUn", salun_result["classwise_history"], config),
        "certified": _save_classwise_artifacts(
            "certified",
            "Certified",
            certified_result["classwise_history"],
            config,
        ),
        "scrub": _save_classwise_artifacts("scrub", "SCRUB", scrub_result["classwise_history"], config),
    }
    for algorithm_key, artifact in algorithm_artifacts.items():
        _log_media(wandb_run, wandb_module, f"plots/classwise_percent_change_{algorithm_key}", artifact.classwise_percent_plot_path)
        _log_media(
            wandb_run,
            wandb_module,
            f"plots/classwise_absolute_accuracy_{algorithm_key}",
            artifact.classwise_absolute_plot_path,
        )

    original_overall, original_per = evaluate(original_model, testloader, num_classes=config.num_classes, device=device)
    retrained_overall, retrained_per = evaluate(retrained_model, testloader, num_classes=config.num_classes, device=device)
    summary_metrics = {
        "original_overall_acc": float(original_overall),
        "retrain_overall_acc": float(retrained_overall),
        f"class_{config.target_label}_original_acc": float(original_per[config.target_label]),
        f"class_{config.target_label}_retrain_acc": float(retrained_per[config.target_label]),
    }
    baseline_results = {
        "ga": ga_result,
        "ssd": ssd_result,
        "salun": salun_result,
        "certified": certified_result,
        "scrub": scrub_result,
    }
    for algorithm_key, result in baseline_results.items():
        overall_acc, per_class_acc = evaluate(
            result["model"], testloader, num_classes=config.num_classes, device=device
        )
        summary_metrics[f"unlearned_{algorithm_key}_overall_acc"] = float(overall_acc)
        summary_metrics[f"class_{config.target_label}_unlearned_{algorithm_key}_acc"] = float(
            per_class_acc[config.target_label]
        )
    if wandb_run is not None:
        wandb_run.log({f"summary/{key}": value for key, value in summary_metrics.items()})

    return CoreExperimentArtifacts(
        original_checkpoint_path=original_path,
        retrained_checkpoint_path=retrained_path,
        original_vs_retrain_plot_path=original_vs_retrain_plot_path,
        algorithm_artifacts=algorithm_artifacts,
        summary_metrics=summary_metrics,
    )


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
    pair_evaluator: Callable[[Dict[str, object], Dict[str, object]], list[dict]],
    transform_rows_for_plot: Callable[[list[dict]], list[dict]],
    wandb_run=None,
    wandb_module=None,
) -> TrajectoryExperimentArtifacts:
    for path in [original_checkpoint_path, retrained_checkpoint_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing checkpoint: {path}")

    testloader = make_loader(testset, config.trajectory_batch_size, False, num_workers, use_cuda)

    retrained_model = model_factory()
    retrained_model.load_state_dict(torch.load(retrained_checkpoint_path, map_location=device))
    retrained_model.eval()
    retrained_acts = activation_collector(retrained_model, testloader)

    original_model = model_factory()
    original_model.load_state_dict(torch.load(original_checkpoint_path, map_location=device))
    original_model.eval()
    original_acts = activation_collector(original_model, testloader)

    reference_map = {
        "retrained": retrained_acts,
        "original": original_acts,
    }

    similarity_artifacts: Dict[str, Dict[str, SimilarityArtifact]] = {}
    metric_names = list(metric_names)
    lower_better_metrics = list(lower_better_metrics)
    for algorithm_key, snapshot_dir in snapshot_dirs.items():
        similarity_artifacts[algorithm_key] = {}
        for reference_key, reference_acts in reference_map.items():
            epoch_rows = compute_epoch_rows_from_snapshots(
                snapshot_dir,
                model_factory,
                reference_acts,
                lambda model: activation_collector(model, testloader),
                pair_evaluator,
                device,
            )
            csv_path = f"{config.out_dir}/similarity_vs_unlearning_epoch_{algorithm_key}_vs_{reference_key}.csv"
            summary_plot_path = (
                f"{config.out_dir}/similarity_vs_unlearning_epoch_{algorithm_key}_vs_{reference_key}_summary.png"
            )
            heatmap_plot_path = (
                f"{config.out_dir}/similarity_vs_unlearning_epoch_{algorithm_key}_vs_{reference_key}_heatmap.png"
            )

            save_similarity_trajectory_csv(epoch_rows, csv_path, metric_names)
            save_metric_summary_plot(
                epoch_rows,
                summary_plot_path,
                f"{algorithm_key.upper()} vs {reference_key.capitalize()} over unlearning steps "
                "(all metrics scaled to higher = more similar)",
                metric_names,
                lower_better_metrics,
            )
            save_similarity_heatmap_grid(
                epoch_rows,
                heatmap_plot_path,
                algorithm_key,
                reference_key,
                layer_names,
                metric_names,
                lower_better_metrics,
            )
            _log_media(
                wandb_run,
                wandb_module,
                f"plots/similarity_vs_unlearning_epoch_{algorithm_key}_vs_{reference_key}_summary",
                summary_plot_path,
            )
            _log_media(
                wandb_run,
                wandb_module,
                f"plots/similarity_vs_unlearning_epoch_{algorithm_key}_vs_{reference_key}_heatmap",
                heatmap_plot_path,
            )
            similarity_artifacts[algorithm_key][reference_key] = SimilarityArtifact(
                csv_path=csv_path,
                summary_plot_path=summary_plot_path,
                heatmap_plot_path=heatmap_plot_path,
            )

    forget_subset, _retain_subset = make_forget_retain_subsets(trainset, config.target_label)
    forget_member_subset, forget_nonmember_subset, forget_n = sample_balanced_class_subsets(
        trainset,
        testset,
        config.target_label,
        config.mia_seed,
    )
    retain_member_subset, retain_nonmember_subset, retain_n = sample_balanced_class_subsets(
        trainset,
        testset,
        config.retain_control_label,
        config.mia_seed + 100,
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
        f"MIA probes ready | forget(frog={config.target_label}): {forget_n} member + {forget_n} non-member | "
        f"retain(label={config.retain_control_label}): {retain_n} member + {retain_n} non-member"
    )

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

    mia_panel_metrics = [
        ("forget_loss_auc", "Loss-threshold AUC (lower = less inferable)"),
        ("forget_logreg_auc", "LogReg AUC (lower = less inferable)"),
        ("forget_logreg_tpr_at_1pct", "LogReg TPR @ 1% FPR (lower = less inferable)"),
        ("forget_logreg_mean_member_prob", "LogReg mean member prob (lower = less inferable)"),
    ]

    mia_artifacts: Dict[str, MIAArtifact] = {}
    for algorithm_key, snapshot_dir in snapshot_dirs.items():
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
        _log_media(wandb_run, wandb_module, f"plots/mia_frog_trajectory_{algorithm_key}", grid_plot_path)
        _log_media(
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

    return TrajectoryExperimentArtifacts(
        similarity_artifacts=similarity_artifacts,
        mia_artifacts=mia_artifacts,
        mia_baseline_csv_path=mia_baseline_csv_path,
    )


def save_combined_trajectory_comparison(
    config: Optional[CombinedComparisonConfig] = None,
    wandb_run=None,
    wandb_module=None,
) -> str:
    config = config or CombinedComparisonConfig()
    algo_keys = list(config.algo_keys)

    algo_epochs = {}
    algo_similarity_series = {}
    metrics_sets = []
    for algo in algo_keys:
        path = f"{config.out_dir}/similarity_vs_unlearning_epoch_{algo}_vs_{config.reference_key}.csv"
        epochs, series, metrics_in_file = load_mean_series(path)
        algo_epochs[algo] = epochs
        algo_similarity_series[algo] = series
        metrics_sets.append(set(metrics_in_file))

    all_similarity_metrics = sorted(set.intersection(*metrics_sets)) if metrics_sets else []
    if not all_similarity_metrics:
        raise RuntimeError("No common similarity metrics found across trajectory CSVs.")

    algo_mia_epochs = {}
    algo_mia_series = {}
    for algo in algo_keys:
        mia_path = f"{config.out_dir}/mia_vs_unlearning_epoch_{algo}.csv"
        epochs, series, _ = load_mia_series(mia_path)
        algo_mia_epochs[algo] = epochs
        algo_mia_series[algo] = series

    mia_baseline = load_mia_baseline(f"{config.out_dir}/mia_retrained_baseline.csv")
    out_path = f"{config.out_dir}/similarity_and_mia_trajectory_combined_algorithms.png"
    save_combined_similarity_mia_plot(
        out_path,
        algo_keys,
        dict(config.algo_display),
        algo_epochs,
        algo_similarity_series,
        all_similarity_metrics,
        config.lower_better_metrics,
        algo_mia_epochs,
        algo_mia_series,
        mia_baseline,
        config.mia_panels,
    )
    _log_media(wandb_run, wandb_module, "plots/similarity_and_mia_trajectory_combined_algorithms", out_path)
    return out_path
