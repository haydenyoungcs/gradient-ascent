"""Thin drivers for `notebooks/experiments.ipynb` — keep the notebook narrative-only.

All imperative setup and section pipelines live here so the notebook stays short.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .notebook_bootstrap import NotebookBootstrapResult, bootstrap_notebook_environment
from .pipelines.multitarget import MultiTargetAggregateArtifacts

REPO_URL = "https://github.com/haydenyoungcs/gradient-ascent.git"
REPO_DIR = Path("/content/gradient-ascent")
DEFAULT_COLAB_OUT_DIR = Path("/content/drive/MyDrive/gradient-ascent-out")


@dataclass(frozen=True)
class ExperimentsBootstrapContext:
    project_root: Path
    default_out_dir: str
    wandb: Any
    in_colab: bool


def _find_project_root_from_cwd() -> Path | None:
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    return None


def experiments_bootstrap() -> ExperimentsBootstrapContext:
    """Colab/local path setup, optional clone, then `bootstrap_notebook_environment` (pip, wandb)."""
    github_token = os.environ.get("GITHUB_TOKEN")
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    in_colab = False

    try:
        from google.colab import drive, userdata  # type: ignore

        in_colab = True
        drive.mount("/content/drive", force_remount=False)
        if github_token is None:
            github_token = userdata.get("GITHUB_TOKEN")
        if wandb_api_key is None:
            wandb_api_key = userdata.get("WANDB_API_KEY")
    except Exception:
        pass

    project_root = _find_project_root_from_cwd() or REPO_DIR
    if not project_root.exists():
        if github_token:
            clone_url = REPO_URL.replace("https://", f"https://{github_token}@")
            subprocess.run(["git", "clone", clone_url], check=True)
        else:
            raise RuntimeError(
                "Repo checkout not found. Add GITHUB_TOKEN as env var/Colab secret, or clone manually."
            )

    if not (project_root / "pyproject.toml").exists():
        raise FileNotFoundError(f"Expected pyproject.toml under {project_root}, but it was not found.")

    repo_src = project_root / "src"
    for path in (project_root, repo_src):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    bootstrap: NotebookBootstrapResult = bootstrap_notebook_environment(
        repo_url=REPO_URL,
        repo_dir=REPO_DIR,
        default_colab_out_dir=DEFAULT_COLAB_OUT_DIR,
        github_token=github_token,
        wandb_api_key=wandb_api_key,
    )
    print(f"Changed working directory to {bootstrap.project_root}")
    print(f"Default OUT_DIR: {bootstrap.default_out_dir}")
    return ExperimentsBootstrapContext(
        project_root=Path(bootstrap.project_root),
        default_out_dir=bootstrap.default_out_dir,
        wandb=bootstrap.wandb,
        in_colab=bootstrap.in_colab,
    )


@dataclass
class CoreSectionConfig:
    num_classes: int = 10
    resnet_model_depth: int = 50
    reuse_existing_checkpoints: bool = True
    reuse_original_checkpoint: bool = True
    reuse_retrained_checkpoint: bool = True
    reuse_unlearned_checkpoints: bool = True
    run_core_diagnostics: bool = False


def experiments_section_core(
    *,
    out_dir: str,
    wandb_module: Any,
    config: Optional[CoreSectionConfig] = None,
) -> tuple[Any, Any, CoreSectionConfig]:
    """Section 1 — original/retrain/unlearn checkpoints and optional diagnostics.

    Returns the ``CoreSectionConfig`` instance that was applied so Section 4 can reuse reuse flags.
    """
    import warnings

    from .notebook_helpers import prepare_notebook_runtime, run_and_display_notebook_core_pipeline

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    cfg = config or CoreSectionConfig()
    runtime = prepare_notebook_runtime(
        num_classes=cfg.num_classes,
        out_dir=out_dir,
        model_depth=cfg.resnet_model_depth,
    )
    print(f"Runtime prepared on device: {runtime.device}")
    core_artifacts, _wandb_run = run_and_display_notebook_core_pipeline(
        runtime,
        wandb_module=wandb_module,
        reuse_existing_checkpoints=cfg.reuse_existing_checkpoints,
        reuse_original_checkpoint=cfg.reuse_original_checkpoint,
        reuse_retrained_checkpoint=cfg.reuse_retrained_checkpoint,
        reuse_unlearned_checkpoints=cfg.reuse_unlearned_checkpoints,
        run_diagnostics=cfg.run_core_diagnostics,
    )
    return runtime, core_artifacts, cfg


def experiments_section_trajectory(
    runtime: Any,
    core_artifacts: Any,
    *,
    wandb_module: Any,
) -> tuple[Any, Any, Any]:
    """Section 2 — similarity + MIA trajectories for all baselines."""
    from .notebook_helpers import prepare_similarity_setup, run_and_display_notebook_trajectory_pipeline

    similarity_setup = prepare_similarity_setup()
    trajectory_artifacts, trajectory_wandb_run = run_and_display_notebook_trajectory_pipeline(
        runtime,
        core_artifacts,
        similarity_setup,
        wandb_module=wandb_module,
    )
    return similarity_setup, trajectory_artifacts, trajectory_wandb_run


def experiments_section_combined(runtime: Any, *, wandb_module: Any) -> str:
    """Section 3 — single overlay figure."""
    from .notebook_helpers import run_and_display_notebook_combined_comparison

    return run_and_display_notebook_combined_comparison(runtime, wandb_module=wandb_module)


def _seed_frog_target_from_single_run(*, single_out: Path, multi_root: Path, target_label: int = 6) -> None:
    """Optional: copy prior single-output-dir artefacts into ``target_<label>/`` for reuse."""
    shared_dir = multi_root / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    shared_original = shared_dir / "original_net.pt"

    src_original = single_out / "original_net.pt"
    if src_original.exists() and not shared_original.exists():
        shutil.copy2(src_original, shared_original)
        print(f"Copied shared original checkpoint to: {shared_original}")

    if not shared_original.exists():
        raise FileNotFoundError(
            f"Missing shared original checkpoint at {shared_original}. "
            "Place/copy your pretrained original model there before running Section 4."
        )

    target_out = multi_root / f"target_{target_label}"
    target_out.mkdir(parents=True, exist_ok=True)

    file_mappings: dict[str, str] = {
        "original_net.pt": "original_net.pt",
        "retrained_from_scratch_net.pt": "retrained_from_scratch_net.pt",
        "original_vs_retrain_acc.png": "original_vs_retrain_acc.png",
        "classwise_accuracy_original.png": "classwise_accuracy_original.png",
        "classwise_accuracy_retrained.png": "classwise_accuracy_retrained.png",
        "classwise_percent_diff_retrained_vs_original.png": "classwise_percent_diff_retrained_vs_original.png",
        "unlearning_runtime_seconds.csv": "unlearning_runtime_seconds.csv",
        "unlearning_runtime_seconds.png": "unlearning_runtime_seconds.png",
        "mia_retrained_baseline.csv": "mia_retrained_baseline.csv",
        "trajectory_timing_seconds.csv": "trajectory_timing_seconds.csv",
    }
    for algo in ("ga", "ssd", "salun", "certified", "scrub"):
        file_mappings[f"unlearned_net_{algo}.pt"] = f"unlearned_net_{algo}.pt"
        file_mappings[f"classwise_accuracy_{algo}.csv"] = f"classwise_accuracy_{algo}.csv"
        file_mappings[f"classwise_percent_change_{algo}.png"] = f"classwise_percent_change_{algo}.png"
        file_mappings[f"classwise_absolute_accuracy_{algo}.png"] = f"classwise_absolute_accuracy_{algo}.png"
        file_mappings[f"mia_vs_unlearning_epoch_{algo}.csv"] = f"mia_vs_unlearning_epoch_{algo}.csv"
        file_mappings[f"mia_frog_trajectory_{algo}.png"] = f"mia_frog_trajectory_{algo}.png"
        file_mappings[f"mia_forget_vs_retain_logreg_mean_member_prob_{algo}.png"] = (
            f"mia_forget_vs_retain_logreg_mean_member_prob_{algo}.png"
        )
        file_mappings[f"similarity_metric_timing_{algo}.png"] = f"similarity_metric_timing_{algo}.png"
        for ref in ("retrained", "original"):
            prefix = f"similarity_vs_unlearning_epoch_{algo}_vs_{ref}"
            file_mappings[f"{prefix}.csv"] = f"{prefix}.csv"
            file_mappings[f"{prefix}_summary.png"] = f"{prefix}_summary.png"
            file_mappings[f"{prefix}_evolving_bars.gif"] = f"{prefix}_evolving_bars.gif"
            file_mappings[f"{prefix}_evolving_grouped_bars.gif"] = f"{prefix}_evolving_grouped_bars.gif"
            file_mappings[f"{prefix}_before_after_grouped_bars.png"] = f"{prefix}_before_after_grouped_bars.png"

    dir_mappings = {
        "unlearning_snapshots_ga": "unlearning_snapshots_ga",
        "unlearning_snapshots_ssd": "unlearning_snapshots_ssd",
        "unlearning_snapshots_salun": "unlearning_snapshots_salun",
        "unlearning_snapshots_certified": "unlearning_snapshots_certified",
        "unlearning_snapshots_scrub": "unlearning_snapshots_scrub",
        "similarity_activation_cache": "similarity_activation_cache",
    }

    copied_files = skipped_files = 0
    for src_name, dst_name in file_mappings.items():
        src = single_out / src_name
        dst = target_out / dst_name
        if not src.exists():
            continue
        if dst.exists():
            skipped_files += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied_files += 1

    copied_dirs = skipped_dirs = 0
    for src_name, dst_name in dir_mappings.items():
        src = single_out / src_name
        dst = target_out / dst_name
        if not src.exists() or not src.is_dir():
            continue
        if dst.exists():
            skipped_dirs += 1
            continue
        shutil.copytree(src, dst)
        copied_dirs += 1

    print(
        "Frog migration complete: "
        f"copied_files={copied_files}, skipped_files={skipped_files}, "
        f"copied_dirs={copied_dirs}, skipped_dirs={skipped_dirs}"
    )
    print(f"Target folder: {target_out}")


@dataclass
class MultitargetSectionConfig:
    run_similarity_stage: bool = True
    reuse_trajectory_outputs: bool = True
    seed_frog_target_from_single_run: bool = True
    frog_target_label: int = 6


def experiments_section_multitarget(
    runtime: Any,
    *,
    out_dir: str,
    wandb_module: Any,
    core_config: CoreSectionConfig,
    multitarget_config: Optional[MultitargetSectionConfig] = None,
) -> MultiTargetAggregateArtifacts:
    """Section 4 — optional frog seeding + forget-label sweep and aggregates."""
    from .notebook_helpers import prepare_similarity_setup, run_multitarget_averaged_experiment

    cfg = multitarget_config or MultitargetSectionConfig()
    single_out = Path(out_dir)
    multi_root = single_out / "multitarget_aggregate"

    if cfg.seed_frog_target_from_single_run:
        _seed_frog_target_from_single_run(single_out=single_out, multi_root=multi_root, target_label=cfg.frog_target_label)

    similarity_setup = prepare_similarity_setup()
    multi_target_artifacts = run_multitarget_averaged_experiment(
        runtime,
        similarity_setup,
        target_labels=list(range(10)),
        out_dir=str(multi_root),
        run_similarity_stage=cfg.run_similarity_stage,
        wandb_module=wandb_module,
        reuse_existing_checkpoints=core_config.reuse_existing_checkpoints,
        reuse_original_checkpoint=core_config.reuse_original_checkpoint,
        reuse_retrained_checkpoint=core_config.reuse_retrained_checkpoint,
        reuse_unlearned_checkpoints=core_config.reuse_unlearned_checkpoints,
        reuse_trajectory_outputs=cfg.reuse_trajectory_outputs,
        shared_original_checkpoint_path=str(multi_root / "shared" / "original_net.pt"),
    )
    print("Saved multi-target aggregate artefacts under:", str(multi_root))
    print("Utility aggregate CSVs:")
    for algorithm_key, csv_path in multi_target_artifacts.utility_csv_paths.items():
        print(f"  {algorithm_key}: {csv_path}")
    if cfg.run_similarity_stage:
        print("Averaged MIA baseline CSV:", multi_target_artifacts.mia_baseline_csv_path)
        print("Averaged MIA trajectory CSVs:")
        for algorithm_key, csv_path in multi_target_artifacts.mia_csv_paths.items():
            print(f"  {algorithm_key}: {csv_path}")
    return multi_target_artifacts


def experiments_section_correlation(
    runtime: Any,
    *,
    out_dir: str,
) -> dict[str, Path]:
    """Section 5 — run-level and epoch-wise similarity vs MIA/accuracy correlations."""
    from .notebook_helpers import (
        run_epochwise_similarity_proxy_analysis_from_notebook,
        run_similarity_mia_correlation_from_notebook,
    )

    multi_base = Path(out_dir) / "multitarget_aggregate"
    paths_run = run_similarity_mia_correlation_from_notebook(
        runtime,
        multitarget_aggregate_dir=str(multi_base),
        references=("retrained", "original"),
    )
    print("Run-level correlation outputs:")
    for key, path in sorted(paths_run.items()):
        print(f"  {key}: {path}")

    paths_epoch = run_epochwise_similarity_proxy_analysis_from_notebook(
        runtime,
        multitarget_aggregate_dir=str(multi_base),
        references=("retrained", "original"),
        min_n=15,
    )
    print("Epoch-wise correlation outputs:")
    for key, path in sorted(paths_epoch.items()):
        print(f"  {key}: {path}")
    return {**paths_run, **{f"epochwise:{k}": v for k, v in paths_epoch.items()}}


# Default section configs (edit attributes in-notebook if needed before calling).
DEFAULT_CORE_CONFIG = CoreSectionConfig()
DEFAULT_MULTITARGET_CONFIG = MultitargetSectionConfig()
