"""Notebook-facing runtime and similarity configuration (no IPython / pipeline orchestration)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch.nn as nn

from .experiments import CoreExperimentConfig, TrajectoryExperimentConfig
from .similarity import CCA


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


def build_core_wandb_config(config: CoreExperimentConfig) -> dict[str, object]:
    """Flatten core experiment settings for Weights & Biases run config."""
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


def build_default_trajectory_config(runtime: NotebookRuntime) -> TrajectoryExperimentConfig:
    return TrajectoryExperimentConfig(
        num_classes=runtime.num_classes,
        model_depth=runtime.model_depth,
        out_dir=runtime.out_dir,
        trajectory_batch_size=512 if runtime.use_bf16 else 256,
        similarity_data_mode="forget",
    )


def similarity_setup_with_trajectory_cca(
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
