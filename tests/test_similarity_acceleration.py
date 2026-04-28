from __future__ import annotations

import unittest

import numpy as np

from gradient_ascent.similarity import CCA, CKA, evaluate_pair_rows, linear_cka_doubly_centered_gram


def _legacy_linear_cka_via_centered_gram(x: np.ndarray, y: np.ndarray) -> float:
    """Reference implementation matching the pre-optimisation CKA path."""
    means_k = (x @ x.T).mean(axis=0, keepdims=True)
    k = x @ x.T - means_k - means_k.T + (x @ x.T).mean()
    means_l = (y @ y.T).mean(axis=0, keepdims=True)
    l = y @ y.T - means_l - means_l.T + (y @ y.T).mean()
    hsic = float(np.sum(k * l))
    normalization = float(np.sqrt(np.sum(k * k) * np.sum(l * l)))
    return hsic / normalization if normalization > 0 else 0.0


class LinearCkaEquivalenceTest(unittest.TestCase):
    def test_matches_legacy_random_matrices(self) -> None:
        rng = np.random.RandomState(1)
        for n, dx, dy in [(40, 12, 16), (80, 64, 64)]:
            x = rng.randn(n, dx).astype(np.float32)
            y = rng.randn(n, dy).astype(np.float32)
            want = _legacy_linear_cka_via_centered_gram(x, y)
            got = linear_cka_doubly_centered_gram(x, y)
            self.assertAlmostEqual(got, want, places=5)
            self.assertAlmostEqual(CKA(kernel="linear").compute_similarity(x, y), want, places=5)


class CcaSmokeTest(unittest.TestCase):
    def test_cca_runs_with_column_cap(self) -> None:
        rng = np.random.RandomState(3)
        n, d = 64, 400
        x = rng.randn(n, d).astype(np.float32)
        y = rng.randn(n, d).astype(np.float32)
        s = CCA(max_columns=80, column_subsample_seed=0).compute_similarity(x, y)
        self.assertGreaterEqual(float(s), 0.0)
        self.assertLessEqual(float(s), 1.0)


class EvaluatePairRowsSubsampleTest(unittest.TestCase):
    def test_subsample_reduces_sample_count(self) -> None:
        rng = np.random.RandomState(2)
        n, d = 200, 8
        acts_a = {"L": rng.randn(n, d).astype(np.float32)}
        acts_b = {"L": rng.randn(n, d).astype(np.float32)}
        rows = evaluate_pair_rows(acts_a, acts_b, layers=["L"], max_activation_samples=50, subsample_seed=0)
        self.assertEqual(rows[0]["n_samples"], 50)


if __name__ == "__main__":
    unittest.main()
