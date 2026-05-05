"""Scalar similarity change vs MIA reduction for dissertation-style exploratory analysis.

Similarity trajectory CSVs store **raw** metric values. Run-level endpoint
features in this module use those raw values directly (unscaled deltas).
Epoch-wise analyses still use oriented layer means to match trajectory plots.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .constants import ALGORITHM_ORDER
from .data import CIFAR10_CLASSES
from .reporting import orient_epoch_rows_for_similarity
from .similarity import LOWER_BETTER_METRICS

# Default trajectory metrics (wide columns in saved CSVs).
DEFAULT_SIMILARITY_METRICS: tuple[str, ...] = ("cka_linear", "cca", "cosine", "euclidean", "kl_sym")

REQUIRED_SIMILARITY_BASE_COLS = frozenset({"epoch", "layer", "n_samples", "n_features"})


def _similarity_wide_to_epoch_rows(df: pd.DataFrame, metric_cols: list[str]) -> list[tuple[int, list[dict]]]:
    """Convert a wide similarity dataframe to the nested structure used by orientation helpers."""
    out: dict[int, list[dict]] = {}
    for epoch in sorted(df["epoch"].unique()):
        sub = df.loc[df["epoch"] == epoch].sort_values("layer")
        rows: list[dict] = []
        for _, row in sub.iterrows():
            d = {
                "layer": str(row["layer"]),
                "n_samples": int(row["n_samples"]),
                "n_features": int(row["n_features"]),
            }
            for m in metric_cols:
                d[m] = float(row[m])
            rows.append(d)
        out[int(epoch)] = rows
    return sorted(out.items(), key=lambda x: x[0])


def _epoch_rows_to_wide_dataframe(epoch_rows: Sequence[tuple[int, list[dict]]]) -> pd.DataFrame:
    flat: list[dict] = []
    for epoch, rows in epoch_rows:
        for r in rows:
            flat.append({"epoch": int(epoch), **r})
    return pd.DataFrame(flat)


def load_similarity_csv(
    path: Path,
    *,
    apply_orientation: bool = True,
    lower_better_metrics: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load a similarity trajectory CSV written by ``save_similarity_trajectory_csv``.

    On-disk files use **raw** scores. When ``apply_orientation`` is True (default),
    rows are re-scaled with ``orient_epoch_rows_for_similarity`` so that, for every
    metric, larger values mean closer match to the reference (consistent with plots).

    Parameters
    ----------
    apply_orientation
        If False, return raw CSV values (use only when you will orient downstream).
    lower_better_metrics
        Passed to ``orient_epoch_rows_for_similarity``; defaults to ``LOWER_BETTER_METRICS``.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Similarity CSV not found: {path}")
    df = pd.read_csv(path)
    missing = REQUIRED_SIMILARITY_BASE_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Similarity CSV missing columns {sorted(missing)}: {path}")

    metric_cols = [c for c in df.columns if c not in REQUIRED_SIMILARITY_BASE_COLS]
    if not metric_cols:
        raise ValueError(f"No metric columns found in similarity CSV: {path}")

    if not apply_orientation:
        return df

    lb = list(lower_better_metrics) if lower_better_metrics is not None else list(LOWER_BETTER_METRICS)
    epoch_rows = _similarity_wide_to_epoch_rows(df, metric_cols)
    oriented = orient_epoch_rows_for_similarity(epoch_rows, metric_cols, lb)
    return _epoch_rows_to_wide_dataframe(oriented)


def compute_topk_layer_delta(
    similarity_df: pd.DataFrame,
    metric: str,
    k: int = 2,
    lower_is_more_similar: bool = False,
    epoch_col: str = "epoch",
    layer_col: str = "layer",
    value_col: str | None = None,
) -> pd.DataFrame:
    """Pick top-k layers by absolute similarity change (end − start) for one metric.

    The wide metric column is named ``metric`` unless ``value_col`` is set.

    Returns a one-row DataFrame with diagnostics.

    In addition to top-k means, this always reports the single-layer endpoint
    change for the most-shifted layer (largest ``abs_delta``):
    ``max_changed_layer_delta`` (signed) and ``max_changed_layer_abs_delta``.
    If ``lower_is_more_similar`` is True (e.g., Euclidean/KL), delta signs are
    flipped so that positive always means "more similar".
    """
    if value_col is None:
        value_col = metric
    if value_col not in similarity_df.columns:
        raise KeyError(f"Metric column {value_col!r} not in similarity dataframe.")

    by_layer = similarity_df[[epoch_col, layer_col, value_col]].copy()
    by_layer = by_layer.rename(columns={value_col: "_val"})
    by_layer[epoch_col] = by_layer[epoch_col].astype(int)

    layer_stats: list[dict] = []
    for layer, g in by_layer.groupby(layer_col, sort=False):
        g2 = g.sort_values(epoch_col)
        epochs = g2[epoch_col].to_numpy()
        vals = g2["_val"].to_numpy(dtype=np.float64)
        if len(vals) < 2:
            start_v = end_v = float(vals[0]) if len(vals) else float("nan")
            start_e = end_e = int(epochs[0]) if len(epochs) else -1
        else:
            start_v, end_v = float(vals[0]), float(vals[-1])
            start_e, end_e = int(epochs[0]), int(epochs[-1])
        raw_delta = end_v - start_v
        delta = -raw_delta if lower_is_more_similar else raw_delta
        layer_stats.append(
            {
                layer_col: str(layer),
                "start_epoch": start_e,
                "end_epoch": end_e,
                "start_value": start_v,
                "end_value": end_v,
                "delta": delta,
                "abs_delta": abs(delta),
            }
        )

    if not layer_stats:
        return pd.DataFrame(
            [
                {
                    "metric": metric,
                    "k": k,
                    "selected_layers": "",
                    "topk_mean_delta": np.nan,
                    "topk_mean_abs_delta": np.nan,
                    "max_abs_delta": np.nan,
                    "max_changed_layer_delta": np.nan,
                    "max_changed_layer_abs_delta": np.nan,
                    "start_epoch": np.nan,
                    "end_epoch": np.nan,
                    "n_layers_available": 0,
                    "selected_layer_deltas": "",
                }
            ]
        )

    stats_df = pd.DataFrame(layer_stats)
    stats_df = stats_df.sort_values(
        ["abs_delta", layer_col],
        ascending=[False, False],
        kind="mergesort",
    )
    kk = min(int(k), len(stats_df))
    top = stats_df.head(kk)
    selected_layers = "|".join(top[layer_col].tolist())
    deltas_str = "|".join(f"{row[layer_col]}:{row['delta']:+.4g}" for _, row in top.iterrows())

    return pd.DataFrame(
        [
            {
                "metric": metric,
                "k": int(k),
                "selected_layers": selected_layers,
                "topk_mean_delta": float(top["delta"].mean()),
                "topk_mean_abs_delta": float(top["abs_delta"].mean()),
                "max_abs_delta": float(stats_df["abs_delta"].max()),
                "max_changed_layer_delta": float(stats_df["delta"].iloc[0]),
                "max_changed_layer_abs_delta": float(stats_df["abs_delta"].iloc[0]),
                "start_epoch": int(top["start_epoch"].iloc[0]),
                "end_epoch": int(top["end_epoch"].iloc[0]),
                "n_layers_available": int(len(stats_df)),
                "selected_layer_deltas": deltas_str,
            }
        ]
    )


def summarise_similarity_deltas(
    similarity_df: pd.DataFrame,
    metrics: Optional[list[str]] = None,
    k: int = 2,
    lower_better_metrics: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Top-k layer deltas for every metric column present (unknown metrics are skipped)."""
    want = list(metrics) if metrics is not None else list(DEFAULT_SIMILARITY_METRICS)
    lower = set(lower_better_metrics) if lower_better_metrics is not None else set(LOWER_BETTER_METRICS)
    frames: list[pd.DataFrame] = []
    for m in want:
        if m not in similarity_df.columns:
            continue
        frames.append(compute_topk_layer_delta(similarity_df, m, k=k, lower_is_more_similar=(m in lower)))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_mia_csv(path: Path, *, mia_value_col: Optional[str] = None) -> tuple[pd.DataFrame, str]:
    """Load MIA trajectory CSV; return dataframe and resolved scalar column name.

    Default scalar for correlation is ``forget_logreg_mean_member_prob`` (forget-set
    logistic-regression mean member probability). Pass ``mia_value_col`` to override.
    """
    if not path.is_file():
        raise FileNotFoundError(f"MIA CSV not found: {path}")
    df = pd.read_csv(path)
    if "epoch" not in df.columns:
        raise ValueError(f"MIA CSV missing 'epoch' column: {path}")

    default_col = "forget_logreg_mean_member_prob"
    if mia_value_col is not None:
        if mia_value_col not in df.columns:
            raise KeyError(f"mia_value_col {mia_value_col!r} not in {path}")
        resolved = mia_value_col
    else:
        if default_col not in df.columns:
            candidates = [c for c in df.columns if c.startswith("forget_") and "mean_member_prob" in c]
            if len(candidates) == 1:
                resolved = candidates[0]
            elif not candidates:
                raise ValueError(
                    f"No default MIA column {default_col!r} and no forget mean_member_prob* column in {path}"
                )
            else:
                raise ValueError(
                    f"Ambiguous MIA value columns {candidates}; pass mia_value_col explicitly for {path}"
                )
        else:
            resolved = default_col
    return df, resolved


def compute_mia_delta(
    mia_df: pd.DataFrame,
    mia_value_col: str,
    epoch_col: str = "epoch",
) -> dict[str, float | int]:
    """Start/end statistics for one scalar MIA trajectory.

    Sign convention
    ---------------
    ``mia_delta`` = end − start (raw change in the attack score).

    ``mia_reduction`` = start − end so that **positive** values mean the attack
    became **less** confident on the forget set (privacy improvement for this score).
    """
    g = mia_df.sort_values(epoch_col)
    starts = g.iloc[0]
    ends = g.iloc[-1]
    start_e, end_e = int(starts[epoch_col]), int(ends[epoch_col])
    start_v, end_v = float(starts[mia_value_col]), float(ends[mia_value_col])
    return {
        "mia_start": start_v,
        "mia_end": end_v,
        "mia_delta": end_v - start_v,
        "mia_abs_delta": abs(end_v - start_v),
        "mia_reduction": start_v - end_v,
        "start_epoch": start_e,
        "end_epoch": end_e,
    }


def _utility_deltas_from_classwise(
    classwise_path: Path,
    forget_label: int,
) -> dict[str, float]:
    """Forget / retain accuracy deltas from classwise history CSV (optional fields)."""
    out = {
        "forget_accuracy_reduction": np.nan,
        "retain_accuracy_delta": np.nan,
    }
    if not classwise_path.is_file():
        return out
    df = pd.read_csv(classwise_path)
    if "epoch" not in df.columns or len(df) < 2:
        return out
    forget_name = CIFAR10_CLASSES[int(forget_label)]
    if forget_name not in df.columns:
        return out
    class_cols = [c for c in df.columns if c != "epoch"]
    g = df.sort_values("epoch")
    forget_s = float(g.iloc[0][forget_name])
    forget_e = float(g.iloc[-1][forget_name])
    retain_cols = [c for c in class_cols if c != forget_name]
    if not retain_cols:
        return out
    retain_s = float(g.iloc[0][retain_cols].mean())
    retain_e = float(g.iloc[-1][retain_cols].mean())
    out["forget_accuracy_reduction"] = forget_s - forget_e
    out["retain_accuracy_delta"] = retain_e - retain_s
    return out


def _discover_multitarget_labels(base_dir: Path) -> list[int]:
    labels: list[int] = []
    for p in sorted(base_dir.iterdir()):
        if not p.is_dir():
            continue
        m = re.fullmatch(r"target_(\d+)", p.name)
        if m:
            labels.append(int(m.group(1)))
    return sorted(labels)


def build_similarity_mia_correlation_table(
    base_dir: Path,
    *,
    algorithms: Optional[Sequence[str]] = None,
    forget_labels: Optional[Sequence[int]] = None,
    references: tuple[str, ...] = ("retrained",),
    similarity_metrics: Optional[list[str]] = None,
    top_k_layers: int = 2,
    mia_value_col: Optional[str] = None,
    lower_better_metrics: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Join per-target similarity summaries with MIA deltas (one row per run × metric × reference)."""
    algos = list(algorithms) if algorithms is not None else list(ALGORITHM_ORDER)
    metrics_filter = list(similarity_metrics) if similarity_metrics is not None else list(DEFAULT_SIMILARITY_METRICS)

    discovered = _discover_multitarget_labels(base_dir)
    if forget_labels is not None:
        targets = [int(x) for x in forget_labels]
    elif discovered:
        targets = discovered
    else:
        targets = [0]
        warnings.warn(
            f"No target_* directories under {base_dir}; assuming flat layout with forget_label=0.",
            UserWarning,
            stacklevel=2,
        )

    rows: list[dict] = []
    for tgt in targets:
        if discovered and tgt not in discovered:
            warnings.warn(f"target_{tgt} not found under {base_dir}; skipping.", UserWarning, stacklevel=2)
            continue
        target_dir = base_dir / f"target_{tgt}" if discovered else base_dir

        for algo in algos:
            sim_paths = {ref: target_dir / f"similarity_vs_unlearning_epoch_{algo}_vs_{ref}.csv" for ref in references}
            mia_path = target_dir / f"mia_vs_unlearning_epoch_{algo}.csv"
            classwise_path = target_dir / f"classwise_accuracy_{algo}.csv"

            if not mia_path.is_file():
                warnings.warn(f"Missing MIA CSV, skipping {algo} target={tgt}: {mia_path}", UserWarning, stacklevel=2)
                continue

            try:
                mia_df, resolved_mia_col = load_mia_csv(mia_path, mia_value_col=mia_value_col)
                mia_stats = compute_mia_delta(mia_df, resolved_mia_col)
            except (FileNotFoundError, ValueError, KeyError) as exc:
                warnings.warn(f"MIA load failed for {mia_path}: {exc}", UserWarning, stacklevel=2)
                continue

            util = _utility_deltas_from_classwise(classwise_path, tgt)

            for ref in references:
                sp = sim_paths[ref]
                if not sp.is_file():
                    warnings.warn(f"Missing similarity CSV, skipping {algo} vs {ref} target={tgt}: {sp}", UserWarning, stacklevel=2)
                    continue
                try:
                    sim_df = load_similarity_csv(
                        sp,
                        apply_orientation=False,
                        lower_better_metrics=lower_better_metrics,
                    )
                except (FileNotFoundError, ValueError) as exc:
                    warnings.warn(f"Similarity load failed for {sp}: {exc}", UserWarning, stacklevel=2)
                    continue

                summ = summarise_similarity_deltas(
                    sim_df,
                    metrics=metrics_filter,
                    k=top_k_layers,
                    lower_better_metrics=lower_better_metrics,
                )
                if summ.empty:
                    warnings.warn(f"No overlap metrics for {sp}; skipping.", UserWarning, stacklevel=2)
                    continue

                for _, r in summ.iterrows():
                    kk = int(r["k"])
                    row = {
                        "algorithm": algo,
                        "forget_label": int(tgt),
                        "reference": ref,
                        "similarity_metric": str(r["metric"]),
                        "mia_value_col": resolved_mia_col,
                        "selected_layers": str(r["selected_layers"]),
                        "max_abs_delta": float(r["max_abs_delta"]),
                        "max_changed_layer_delta": float(r["max_changed_layer_delta"]),
                        "max_changed_layer_abs_delta": float(r["max_changed_layer_abs_delta"]),
                        "k": kk,
                        "mia_start": mia_stats["mia_start"],
                        "mia_end": mia_stats["mia_end"],
                        "mia_delta": mia_stats["mia_delta"],
                        "mia_abs_delta": mia_stats["mia_abs_delta"],
                        "mia_reduction": mia_stats["mia_reduction"],
                        "similarity_start_epoch": int(r["start_epoch"]),
                        "similarity_end_epoch": int(r["end_epoch"]),
                        "selected_layer_deltas": str(r["selected_layer_deltas"]),
                        "forget_accuracy_reduction": util["forget_accuracy_reduction"],
                        "retain_accuracy_delta": util["retain_accuracy_delta"],
                    }
                    if kk == 2:
                        row["top2_mean_delta"] = float(r["topk_mean_delta"])
                        row["top2_mean_abs_delta"] = float(r["topk_mean_abs_delta"])
                    else:
                        row["topk_mean_delta"] = float(r["topk_mean_delta"])
                        row["topk_mean_abs_delta"] = float(r["topk_mean_abs_delta"])
                    rows.append(row)

    return pd.DataFrame(rows)


def compute_correlations(
    correlation_table: pd.DataFrame,
    *,
    y_col: str = "mia_reduction",
) -> pd.DataFrame:
    """Pooled Pearson/Spearman correlations across all rows in the table.

    One block of outputs per (reference, similarity_metric). Preferred x feature
    is ``max_changed_layer_delta`` (largest-|Δ| layer, signed). Legacy top-k
    means are still included when present for backwards-compatible diagnostics.
    """
    if correlation_table.empty:
        return pd.DataFrame()

    out_rows: list[dict] = []
    x_candidates: list[tuple[str, str]] = []
    if "max_changed_layer_delta" in correlation_table.columns:
        x_candidates.extend(
            [
                ("max_changed_layer_delta", "max_changed_layer_delta"),
                ("max_changed_layer_abs_delta", "max_changed_layer_abs_delta"),
            ]
        )
    if "top2_mean_delta" in correlation_table.columns:
        x_candidates.extend(
            [
                ("top2_mean_delta", "top2_mean_delta"),
                ("top2_mean_abs_delta", "top2_mean_abs_delta"),
            ]
        )
    if "topk_mean_delta" in correlation_table.columns:
        x_candidates.extend(
            [
                ("topk_mean_delta", "topk_mean_delta"),
                ("topk_mean_abs_delta", "topk_mean_abs_delta"),
            ]
        )
    if "max_abs_delta" in correlation_table.columns:
        x_candidates.append(("max_abs_delta", "max_abs_delta"))

    for ref in correlation_table["reference"].unique():
        sub_ref = correlation_table[correlation_table["reference"] == ref]
        for metric in sub_ref["similarity_metric"].unique():
            block = sub_ref[sub_ref["similarity_metric"] == metric]
            y = block[y_col].to_numpy(dtype=np.float64)
            for x_key, x_label in x_candidates:
                if x_key not in block.columns:
                    continue
                x = block[x_key].to_numpy(dtype=np.float64)
                mask = np.isfinite(x) & np.isfinite(y)
                if int(mask.sum()) < 3:
                    continue
                x2, y2 = x[mask], y[mask]
                pr, pp = stats.pearsonr(x2, y2)
                sr, sp = stats.spearmanr(x2, y2)
                for corr_type, rval, pval in (
                    ("pearson", float(pr), float(pp)),
                    ("spearman", float(sr), float(sp)),
                ):
                    out_rows.append(
                        {
                            "scope": "pooled_all_algorithms",
                            "reference": ref,
                            "similarity_metric": metric,
                            "x_feature": x_label,
                            "y_feature": y_col,
                            "correlation_type": corr_type,
                            "r": rval,
                            "p_value": pval,
                            "n": int(mask.sum()),
                        }
                    )
    return pd.DataFrame(out_rows)


def scatter_similarity_vs_mia(
    table: pd.DataFrame,
    *,
    out_path: Path,
    x_col: str,
    y_col: str = "mia_reduction",
    title: str | None = None,
) -> Path:
    """Scatter with algorithm colour and a simple least-squares line."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if table.empty or x_col not in table.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return out_path

    fig, ax = plt.subplots(figsize=(7, 5))
    algos = sorted(table["algorithm"].unique())
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4", "#ff7f0e", "#2ca02c"])
    for i, algo in enumerate(algos):
        sub = table[table["algorithm"] == algo]
        ax.scatter(
            sub[x_col],
            sub[y_col],
            label=algo,
            color=palette[i % len(palette)],
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
        )

    x = table[x_col].to_numpy(dtype=np.float64)
    y = table[y_col].to_numpy(dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) >= 2:
        coef = np.polyfit(x[mask], y[mask], 1)
        xs = np.linspace(float(np.min(x[mask])), float(np.max(x[mask])), 50)
        ax.plot(xs, np.poly1d(coef)(xs), color="black", linewidth=1.2, linestyle="--", label="LS line")

    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    ax.axvline(0.0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title or f"{y_col} vs {x_col}")
    ax.legend(loc="best", fontsize="small", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def bar_correlation_summary(
    corr_df: pd.DataFrame,
    *,
    out_path: Path,
    reference: str,
    x_feature: str,
    correlation_type: str = "spearman",
) -> Path:
    """Bar chart: similarity metric vs correlation coefficient for one x_feature."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub = corr_df[
        (corr_df["reference"] == reference)
        & (corr_df["x_feature"] == x_feature)
        & (corr_df["correlation_type"] == correlation_type)
    ]
    if sub.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return out_path

    order = sorted(sub["similarity_metric"].unique())
    rvals = [float(sub.loc[sub["similarity_metric"] == m, "r"].iloc[0]) for m in order]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    xpos = np.arange(len(order))
    ax.bar(xpos, rvals, color="#4c72b0")
    ax.set_xticks(xpos)
    ax.set_xticklabels(order, rotation=25, ha="right")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel(f"{correlation_type} r")
    ax.set_title(f"{reference}: {x_feature} vs mia_reduction")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _load_classwise_epoch_accuracy(path: Path, forget_label: int) -> pd.DataFrame:
    """Per-epoch forget-class accuracy and mean retained-class accuracy (0–1 scale)."""
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "epoch" not in df.columns:
        return pd.DataFrame()
    forget_name = CIFAR10_CLASSES[int(forget_label)]
    if forget_name not in df.columns:
        return pd.DataFrame()
    class_cols = [c for c in df.columns if c != "epoch"]
    retain_cols = [c for c in class_cols if c != forget_name]
    if not retain_cols:
        return pd.DataFrame()
    g = df.sort_values("epoch")
    out = pd.DataFrame(
        {
            "epoch": g["epoch"].astype(int),
            "forget_accuracy": g[forget_name].astype(float),
            "retain_mean_accuracy": g[retain_cols].mean(axis=1).astype(float),
        }
    )
    return out


def _similarity_mean_over_layers_per_epoch(sim_df: pd.DataFrame, metric_names: Sequence[str]) -> pd.DataFrame:
    """Pool layers: one value per metric per unlearning step (matches summary-plot intuition)."""
    present = [m for m in metric_names if m in sim_df.columns]
    if not present:
        return pd.DataFrame()
    g = sim_df.groupby("epoch", as_index=False)[present].mean()
    rename = {m: f"sim_mean_{m}" for m in present}
    return g.rename(columns=rename)


def build_epoch_level_multitarget_long_table(
    base_dir: Path,
    *,
    algorithms: Optional[Sequence[str]] = None,
    forget_labels: Optional[Sequence[int]] = None,
    references: tuple[str, ...] = ("retrained", "original"),
    similarity_metrics: Optional[list[str]] = None,
    mia_value_col: Optional[str] = None,
    lower_better_metrics: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Stack per-target trajectories: each row is one (algorithm, forget class, reference, step).

    Similarity columns are ``sim_mean_<metric>`` (mean over layers, **oriented** scores).
    Outcomes: ``mia_value`` (scalar MIA column), ``forget_accuracy``, ``retain_mean_accuracy``.
    """
    algos = list(algorithms) if algorithms is not None else list(ALGORITHM_ORDER)
    metrics = list(similarity_metrics) if similarity_metrics is not None else list(DEFAULT_SIMILARITY_METRICS)
    lb = list(lower_better_metrics) if lower_better_metrics is not None else list(LOWER_BETTER_METRICS)

    discovered = _discover_multitarget_labels(base_dir)
    if forget_labels is not None:
        targets = [int(x) for x in forget_labels]
    elif discovered:
        targets = discovered
    else:
        targets = [0]
        warnings.warn(
            f"No target_* under {base_dir}; using flat layout forget_label=0.",
            UserWarning,
            stacklevel=2,
        )

    parts: list[pd.DataFrame] = []
    for tgt in targets:
        if discovered and tgt not in discovered:
            continue
        target_dir = base_dir / f"target_{tgt}" if discovered else base_dir
        for algo in algos:
            mia_path = target_dir / f"mia_vs_unlearning_epoch_{algo}.csv"
            classwise_path = target_dir / f"classwise_accuracy_{algo}.csv"
            if not mia_path.is_file():
                continue
            try:
                mia_df, resolved_mia = load_mia_csv(mia_path, mia_value_col=mia_value_col)
            except (FileNotFoundError, ValueError, KeyError):
                continue
            mia_sub = mia_df[["epoch", resolved_mia]].copy()
            mia_sub = mia_sub.rename(columns={resolved_mia: "mia_value"})
            mia_sub["epoch"] = mia_sub["epoch"].astype(int)

            acc_sub = _load_classwise_epoch_accuracy(classwise_path, tgt)
            if acc_sub.empty:
                acc_sub = pd.DataFrame({"epoch": mia_sub["epoch"], "forget_accuracy": np.nan, "retain_mean_accuracy": np.nan})

            for ref in references:
                sim_path = target_dir / f"similarity_vs_unlearning_epoch_{algo}_vs_{ref}.csv"
                if not sim_path.is_file():
                    continue
                try:
                    sim_df = load_similarity_csv(sim_path, apply_orientation=True, lower_better_metrics=lb)
                except (FileNotFoundError, ValueError):
                    continue
                mean_sim = _similarity_mean_over_layers_per_epoch(sim_df, metrics)
                if mean_sim.empty:
                    continue
                mean_sim["epoch"] = mean_sim["epoch"].astype(int)
                merged = mean_sim.merge(mia_sub, on="epoch", how="inner").merge(acc_sub, on="epoch", how="left")
                merged["algorithm"] = algo
                merged["forget_label"] = int(tgt)
                merged["reference"] = ref
                merged["mia_value_col"] = resolved_mia
                parts.append(merged)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def compute_epochwise_cross_run_correlations(
    long_df: pd.DataFrame,
    *,
    min_n: int = 15,
) -> pd.DataFrame:
    """At each unlearning step, correlate similarity means with MIA / accuracy across runs.

    Each correlation uses all (algorithm × forget class) rows available at that ``epoch``
    (typically 50 when five algorithms and ten forget labels are present).
    """
    if long_df.empty:
        return pd.DataFrame()

    sim_cols = [c for c in long_df.columns if c.startswith("sim_mean_")]
    outcome_cols = ["mia_value", "forget_accuracy", "retain_mean_accuracy"]
    rows: list[dict] = []

    for ref in long_df["reference"].unique():
        for epoch in sorted(long_df["epoch"].unique()):
            block = long_df[(long_df["reference"] == ref) & (long_df["epoch"] == int(epoch))]
            for sim_col in sim_cols:
                metric_name = sim_col.replace("sim_mean_", "", 1)
                for out_col in outcome_cols:
                    if out_col not in block.columns:
                        continue
                    x = block[sim_col].to_numpy(dtype=np.float64)
                    y = block[out_col].to_numpy(dtype=np.float64)
                    mask = np.isfinite(x) & np.isfinite(y)
                    if int(mask.sum()) < int(min_n):
                        continue
                    x2, y2 = x[mask], y[mask]
                    if float(np.nanstd(x2)) < 1e-12 or float(np.nanstd(y2)) < 1e-12:
                        continue
                    pr, pp = stats.pearsonr(x2, y2)
                    sr, sp = stats.spearmanr(x2, y2)
                    rows.append(
                        {
                            "reference": ref,
                            "epoch": int(epoch),
                            "similarity_metric": metric_name,
                            "outcome": out_col,
                            "pearson_r": float(pr),
                            "pearson_p": float(pp),
                            "spearman_r": float(sr),
                            "spearman_p": float(sp),
                            "n": int(mask.sum()),
                        }
                    )
    return pd.DataFrame(rows)


def plot_epochwise_correlation_lines(
    epoch_corr: pd.DataFrame,
    *,
    reference: str,
    outcome: str,
    out_path: Path,
    correlation: str = "spearman",
) -> Path:
    """One line per similarity metric: correlation vs unlearning step for a fixed outcome."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rcol = f"{correlation}_r"
    sub = epoch_corr[(epoch_corr["reference"] == reference) & (epoch_corr["outcome"] == outcome)]
    if sub.empty or rcol not in sub.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return out_path

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for metric in sorted(sub["similarity_metric"].unique()):
        msub = sub[sub["similarity_metric"] == metric].sort_values("epoch")
        ax.plot(msub["epoch"], msub[rcol], marker="o", linewidth=1.8, label=metric)
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Unlearning step")
    ax.set_ylabel(f"{correlation} r (across runs)")
    ax.set_title(f"{reference}: similarity vs {outcome}")
    ax.legend(loc="best", fontsize="small", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def run_epochwise_similarity_outcome_correlation(
    out_dir: Path,
    *,
    algorithms: Optional[list[str]] = None,
    forget_labels: Optional[list[int]] = None,
    references: tuple[str, ...] = ("retrained", "original"),
    similarity_metrics: Optional[list[str]] = None,
    mia_value_col: Optional[str] = None,
    lower_better_metrics: Optional[Sequence[str]] = None,
    min_n: int = 15,
) -> dict[str, Path]:
    """Cross-run correlations at each step; saves CSV and line plots under ``out_dir/correlation/epochwise/``."""
    out_dir = Path(out_dir)
    root = out_dir / "correlation" / "epochwise"
    root.mkdir(parents=True, exist_ok=True)

    long_t = build_epoch_level_multitarget_long_table(
        out_dir,
        algorithms=algorithms,
        forget_labels=forget_labels,
        references=references,
        similarity_metrics=similarity_metrics,
        mia_value_col=mia_value_col,
        lower_better_metrics=lower_better_metrics,
    )
    long_csv = root / "epoch_level_multitarget_long.csv"
    long_t.to_csv(long_csv, index=False)

    epoch_corr = compute_epochwise_cross_run_correlations(long_t, min_n=min_n)
    corr_csv = root / "epochwise_similarity_outcome_correlations.csv"
    epoch_corr.to_csv(corr_csv, index=False)

    paths: dict[str, Path] = {"epoch_level_long": long_csv, "epochwise_correlations": corr_csv}

    outcomes = ["mia_value", "forget_accuracy", "retain_mean_accuracy"]
    for ref in references:
        for oc in outcomes:
            safe = oc.replace("/", "_")
            pth = root / f"epochwise_spearman_r_{ref}_{safe}.png"
            plot_epochwise_correlation_lines(
                epoch_corr,
                reference=ref,
                outcome=oc,
                out_path=pth,
                correlation="spearman",
            )
            paths[f"lines_{ref}_{oc}_spearman"] = pth

    return paths


def run_similarity_mia_correlation_analysis(
    out_dir: Path,
    *,
    algorithms: Optional[list[str]] = None,
    forget_labels: Optional[list[int]] = None,
    references: tuple[str, ...] = ("retrained",),
    similarity_metrics: Optional[list[str]] = None,
    top_k_layers: int = 2,
    mia_value_col: Optional[str] = None,
    lower_better_metrics: Optional[Sequence[str]] = None,
) -> dict[str, Path]:
    """Build correlation tables, summary stats, and plots under ``out_dir / 'correlation'``."""
    out_dir = Path(out_dir)
    corr_root = out_dir / "correlation"
    corr_root.mkdir(parents=True, exist_ok=True)

    table = build_similarity_mia_correlation_table(
        out_dir,
        algorithms=algorithms,
        forget_labels=forget_labels,
        references=references,
        similarity_metrics=similarity_metrics,
        top_k_layers=top_k_layers,
        mia_value_col=mia_value_col,
        lower_better_metrics=lower_better_metrics,
    )
    table_csv = corr_root / "similarity_mia_correlation_table.csv"
    table.to_csv(table_csv, index=False)

    corr = compute_correlations(table)
    summary_csv = corr_root / "similarity_mia_correlation_summary.csv"
    corr.to_csv(summary_csv, index=False)

    paths: dict[str, Path] = {"table": table_csv, "summary": summary_csv}

    x_scatter_cols: list[str] = []
    if "max_changed_layer_delta" in table.columns:
        x_scatter_cols.extend(["max_changed_layer_delta", "max_changed_layer_abs_delta"])
    elif top_k_layers == 2 and "top2_mean_delta" in table.columns:
        x_scatter_cols.extend(["top2_mean_delta", "top2_mean_abs_delta"])
    else:
        x_scatter_cols.extend(["topk_mean_delta", "topk_mean_abs_delta"])

    for ref in references:
        sub_ref = table[table["reference"] == ref]
        for metric in sub_ref["similarity_metric"].unique():
            block = sub_ref[sub_ref["similarity_metric"] == metric]
            for xc in x_scatter_cols:
                if xc not in block.columns:
                    continue
                fname = f"scatter_{ref}_{metric}_{xc}_vs_mia_reduction.png"
                pth = corr_root / fname
                scatter_similarity_vs_mia(
                    block,
                    out_path=pth,
                    x_col=xc,
                    y_col="mia_reduction",
                    title=f"{ref} | {metric}: {xc} vs mia_reduction",
                )
                paths[f"scatter_{ref}_{metric}_{xc}"] = pth

        bar_features = list(dict.fromkeys(x_scatter_cols))
        for use_x in bar_features:
            if use_x not in table.columns:
                continue
            pth = corr_root / f"correlation_summary_{ref}_{use_x}.png"
            bar_correlation_summary(corr, out_path=pth, reference=ref, x_feature=use_x)
            paths[f"bar_summary_{ref}_{use_x}"] = pth

    return paths
