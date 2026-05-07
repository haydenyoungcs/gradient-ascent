from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import pandas as pd

from gradient_ascent.correlation import (
    build_similarity_mia_correlation_table,
    compute_correlations,
    compute_epochwise_cross_run_correlations,
    compute_mia_delta,
    compute_topk_layer_delta,
    load_mia_csv,
    load_similarity_csv,
)


class TopkLayerDeltaTest(unittest.TestCase):
    def test_topk_picks_largest_abs_delta(self) -> None:
        # Three layers, two epochs, cosine values already oriented.
        df = pd.DataFrame(
            [
                {"epoch": 0, "layer": "a", "n_samples": 1, "n_features": 1, "cosine": 0.5},
                {"epoch": 1, "layer": "a", "n_samples": 1, "n_features": 1, "cosine": 0.6},
                {"epoch": 0, "layer": "b", "n_samples": 1, "n_features": 1, "cosine": 0.2},
                {"epoch": 1, "layer": "b", "n_samples": 1, "n_features": 1, "cosine": 0.9},
                {"epoch": 0, "layer": "c", "n_samples": 1, "n_features": 1, "cosine": 0.4},
                {"epoch": 1, "layer": "c", "n_samples": 1, "n_features": 1, "cosine": 0.35},
            ]
        )
        out = compute_topk_layer_delta(df, "cosine", k=2)
        # |delta|: b=0.7, a=0.1, c=0.05 — ties broken by layer name.
        self.assertEqual(out["selected_layers"].iloc[0], "b|a")
        # b and a together give mean delta=0.4 and mean |delta|=0.4.
        self.assertAlmostEqual(float(out["topk_mean_delta"].iloc[0]), 0.4)
        self.assertAlmostEqual(float(out["topk_mean_abs_delta"].iloc[0]), 0.4)
        self.assertAlmostEqual(float(out["max_abs_delta"].iloc[0]), 0.7)

    def test_signed_delta_direction(self) -> None:
        df = pd.DataFrame(
            [
                {"epoch": 0, "layer": "x", "n_samples": 1, "n_features": 1, "cosine": 1.0},
                {"epoch": 2, "layer": "x", "n_samples": 1, "n_features": 1, "cosine": 0.0},
            ]
        )
        out = compute_topk_layer_delta(df, "cosine", k=1)
        self.assertAlmostEqual(float(out["topk_mean_delta"].iloc[0]), -1.0)


class MiaReductionTest(unittest.TestCase):
    def test_mia_reduction_positive_when_attack_drops(self) -> None:
        mia_df = pd.DataFrame(
            {
                "epoch": [0, 1],
                "forget_logreg_mean_member_prob": [0.8, 0.3],
            }
        )
        d = compute_mia_delta(mia_df, "forget_logreg_mean_member_prob")
        self.assertAlmostEqual(d["mia_reduction"], 0.5)
        self.assertAlmostEqual(d["mia_delta"], -0.5)


class JoinTableTest(unittest.TestCase):
    def test_joins_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            tdir = base / "target_0"
            tdir.mkdir(parents=True)
            sim = pd.DataFrame(
                [
                    {"epoch": 0, "layer": "L1", "n_samples": 8, "n_features": 4, "cosine": 0.1, "cka_linear": 0.1},
                    {"epoch": 1, "layer": "L1", "n_samples": 8, "n_features": 4, "cosine": 0.9, "cka_linear": 0.2},
                ]
            )
            sim.to_csv(tdir / "similarity_vs_unlearning_epoch_ga_vs_retrained.csv", index=False)
            mia = pd.DataFrame(
                {
                    "epoch": [0, 1],
                    "forget_logreg_mean_member_prob": [0.7, 0.4],
                }
            )
            mia.to_csv(tdir / "mia_vs_unlearning_epoch_ga.csv", index=False)
            cw = pd.DataFrame(
                {
                    "epoch": [0, 1],
                    **{name: [0.5, 0.4] for name in ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")},
                }
            )
            cw.to_csv(tdir / "classwise_accuracy_ga.csv", index=False)

            table = build_similarity_mia_correlation_table(
                base,
                algorithms=["ga"],
                forget_labels=[0],
                references=("retrained",),
                similarity_metrics=["cosine"],
                top_k_layers=1,
            )
            self.assertEqual(len(table), 1)
            self.assertEqual(table.iloc[0]["algorithm"], "ga")
            self.assertEqual(int(table.iloc[0]["forget_label"]), 0)
            self.assertIn("mia_reduction", table.columns)


class EpochwiseCorrelationTest(unittest.TestCase):
    def test_epochwise_perfect_positive(self) -> None:
        rows = []
        for epoch in (0, 1):
            for i in range(5):
                rows.append(
                    {
                        "reference": "retrained",
                        "epoch": epoch,
                        "sim_mean_cosine": float(i),
                        "mia_value": float(2 * i),
                        "forget_accuracy": 0.5,
                        "retain_mean_accuracy": 0.5,
                    }
                )
        df = pd.DataFrame(rows)
        out = compute_epochwise_cross_run_correlations(df, min_n=3)
        self.assertFalse(out.empty)
        c0 = out[(out["epoch"] == 0) & (out["similarity_metric"] == "cosine") & (out["outcome"] == "mia_value")]
        self.assertAlmostEqual(float(c0.iloc[0]["spearman_r"]), 1.0)


class CorrelationValuesTest(unittest.TestCase):
    def test_pearson_on_perfect_line(self) -> None:
        df = pd.DataFrame(
            {
                "reference": ["retrained"] * 5,
                "similarity_metric": ["cosine"] * 5,
                "top2_mean_delta": [0.0, 1.0, 2.0, 3.0, 4.0],
                "mia_reduction": [0.0, 2.0, 4.0, 6.0, 8.0],
            }
        )
        corr = compute_correlations(df)
        sub = corr[(corr["correlation_type"] == "pearson") & (corr["x_feature"] == "top2_mean_delta")]
        self.assertEqual(len(sub), 1)
        self.assertAlmostEqual(float(sub.iloc[0]["r"]), 1.0)
        self.assertLess(float(sub.iloc[0]["p_value"]), 0.05)

    def test_spearman_on_monotone(self) -> None:
        df = pd.DataFrame(
            {
                "reference": ["retrained"] * 4,
                "similarity_metric": ["cosine"] * 4,
                "top2_mean_delta": [1.0, 2.0, 3.0, 100.0],
                "mia_reduction": [0.5, 1.0, 1.5, 2.0],
            }
        )
        corr = compute_correlations(df)
        sub = corr[(corr["correlation_type"] == "spearman") & (corr["x_feature"] == "top2_mean_delta")]
        self.assertEqual(len(sub), 1)
        self.assertAlmostEqual(float(sub.iloc[0]["r"]), 1.0)

    def test_forget_accuracy_reduction_y_col(self) -> None:
        df = pd.DataFrame(
            {
                "reference": ["retrained"] * 5,
                "similarity_metric": ["cosine"] * 5,
                "top2_mean_delta": [0.0, 1.0, 2.0, 3.0, 4.0],
                "forget_accuracy_reduction": [0.0, 0.1, 0.2, 0.3, 0.4],
            }
        )
        corr = compute_correlations(df, y_col="forget_accuracy_reduction")
        sub = corr[(corr["correlation_type"] == "pearson") & (corr["x_feature"] == "top2_mean_delta")]
        self.assertEqual(len(sub), 1)
        self.assertEqual(sub.iloc[0]["y_feature"], "forget_accuracy_reduction")
        self.assertAlmostEqual(float(sub.iloc[0]["r"]), 1.0)


class MissingHandlingTest(unittest.TestCase):
    def test_missing_similarity_warns_and_skips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            tdir = base / "target_1"
            tdir.mkdir(parents=True)
            mia = pd.DataFrame({"epoch": [0, 1], "forget_logreg_mean_member_prob": [0.5, 0.5]})
            mia.to_csv(tdir / "mia_vs_unlearning_epoch_ga.csv", index=False)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                table = build_similarity_mia_correlation_table(
                    base,
                    algorithms=["ga"],
                    forget_labels=[1],
                    references=("retrained",),
                    similarity_metrics=["cosine"],
                )
                self.assertTrue(any("Missing similarity" in str(x.message) for x in w))
            self.assertTrue(table.empty)


class LoadSimilarityCsvTest(unittest.TestCase):
    def test_load_orients_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "s.csv"
            # Raw Euclidean: smaller = more similar.
            raw = pd.DataFrame(
                [
                    {"epoch": 0, "layer": "a", "n_samples": 1, "n_features": 2, "euclidean": 2.0},
                    {"epoch": 1, "layer": "a", "n_samples": 1, "n_features": 2, "euclidean": 1.0},
                    {"epoch": 0, "layer": "b", "n_samples": 1, "n_features": 2, "euclidean": 2.0},
                    {"epoch": 1, "layer": "b", "n_samples": 1, "n_features": 2, "euclidean": 1.0},
                ]
            )
            raw.to_csv(p, index=False)
            df = load_similarity_csv(p, apply_orientation=True)
            self.assertTrue((df["euclidean"] >= 0.0).all() and (df["euclidean"] <= 1.0).all())


class LoadMiaCsvTest(unittest.TestCase):
    def test_default_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.csv"
            pd.DataFrame({"epoch": [0], "forget_logreg_mean_member_prob": [0.42]}).to_csv(p, index=False)
            df, col = load_mia_csv(p)
            self.assertEqual(col, "forget_logreg_mean_member_prob")
            self.assertAlmostEqual(float(df.iloc[0][col]), 0.42)


if __name__ == "__main__":
    unittest.main()
