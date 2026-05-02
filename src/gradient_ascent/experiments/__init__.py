"""Experiment orchestration: core training/unlearning, trajectory evaluation, combined figures."""

from .combined import save_combined_trajectory_comparison
from .core import run_core_checkpoints
from .trajectory import run_trajectory_analysis
from .types import (
    DEFAULT_ALGO_DISPLAY,
    DEFAULT_ALGO_KEYS,
    AlgorithmArtifacts,
    CombinedComparisonConfig,
    CoreExperimentArtifacts,
    CoreExperimentConfig,
    MIAArtifact,
    SimilarityArtifact,
    TrajectoryExperimentArtifacts,
    TrajectoryExperimentConfig,
)
from .wandb_utils import ensure_wandb_run, log_media

__all__ = [
    "DEFAULT_ALGO_DISPLAY",
    "DEFAULT_ALGO_KEYS",
    "AlgorithmArtifacts",
    "CombinedComparisonConfig",
    "CoreExperimentArtifacts",
    "CoreExperimentConfig",
    "MIAArtifact",
    "SimilarityArtifact",
    "TrajectoryExperimentArtifacts",
    "TrajectoryExperimentConfig",
    "ensure_wandb_run",
    "log_media",
    "run_core_checkpoints",
    "run_trajectory_analysis",
    "save_combined_trajectory_comparison",
]
