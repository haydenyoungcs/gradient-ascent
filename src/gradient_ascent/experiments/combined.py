from __future__ import annotations

from typing import Optional

from ..reporting import load_mean_series, load_mia_baseline, load_mia_series
from ..trajectories import save_combined_similarity_mia_plot

from .types import CombinedComparisonConfig
from .wandb_utils import log_media


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
    log_media(wandb_run, wandb_module, "plots/similarity_and_mia_trajectory_combined_algorithms", out_path)
    return out_path
