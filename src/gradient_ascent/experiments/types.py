from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Mapping, Optional, Sequence

from ..unlearning import (
    CertifiedConfig,
    GAConfig,
    SCRUBConfig,
    SSDConfig,
    SalUnConfig,
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
    original_classwise_plot_path: str
    retrained_classwise_plot_path: str
    retrained_vs_original_percent_diff_plot_path: str
    algorithm_artifacts: Dict[str, AlgorithmArtifacts]
    unlearning_runtime_csv_path: str
    unlearning_runtime_plot_path: str
    unlearning_runtime_seconds: Dict[str, float]
    summary_metrics: Dict[str, float]


@dataclass(frozen=True)
class TrajectoryExperimentConfig:
    num_classes: int = 10
    model_depth: int = 50
    out_dir: str = "out"
    target_label: int = 6
    retain_control_label: int = 0
    trajectory_batch_size: int = 256
    similarity_data_mode: Literal["forget", "retain", "test"] = "forget"
    max_batches_for_similarity: int = 8
    max_activation_samples: Optional[int] = 1024
    activation_subsample_seed: int = 42
    log_similarity_progress: bool = True
    cache_similarity_activations: bool = True
    similarity_activation_cache_subdir: str = "similarity_activation_cache"
    cca_max_columns: Optional[int] = 256
    cca_column_subsample_seed: int = 43
    similarity_step_stride: int = 1
    mia_seed: int = 1337
    mia_fixed_cv_across_epochs: bool = True
    mia_include_mlp_attacker: bool = True
    mia_balance_probe_classes: bool = False
    mia_bootstrap_rounds: int = 200
    mia_mlp_sweep_enabled: bool = True
    mia_mlp_sweep_hidden_layer_sizes: tuple[tuple[int, ...], ...] = ((64, 32), (128, 64), (64,))
    mia_mlp_sweep_alphas: tuple[float, ...] = (1e-4, 1e-3)


@dataclass(frozen=True)
class SimilarityArtifact:
    csv_path: str
    summary_plot_path: str
    evolving_bar_plot_path: str
    grouped_evolving_bar_plot_path: str
    before_after_grouped_plot_path: str


@dataclass(frozen=True)
class MIAArtifact:
    csv_path: str
    grid_plot_path: str
    control_plot_path: str


@dataclass(frozen=True)
class TrajectoryExperimentArtifacts:
    similarity_artifacts: Dict[str, Dict[str, SimilarityArtifact]]
    similarity_metric_timing_plot_paths: Dict[str, str]
    mia_artifacts: Dict[str, MIAArtifact]
    mia_baseline_csv_path: str
    timing_csv_path: str
    timing_seconds: Dict[str, float]


@dataclass(frozen=True)
class CombinedComparisonConfig:
    out_dir: str = "out"
    algo_keys: Sequence[str] = DEFAULT_ALGO_KEYS
    algo_display: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_ALGO_DISPLAY))
    reference_key: str = "retrained"
    lower_better_metrics: Sequence[str] = ("euclidean", "kl_sym")
    mia_panels: Sequence[tuple[str, str]] = (
        ("forget_logreg_auc", "Forget LogReg AUC (lower = less inferable)"),
        ("forget_logreg_advantage", "Forget LogReg advantage (lower = less inferable)"),
        ("forget_mlp_auc", "Forget MLP AUC (lower = less inferable)"),
        ("forget_mlp_advantage", "Forget MLP advantage (lower = less inferable)"),
    )
