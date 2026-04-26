from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import svd

from .models import DEFAULT_LAYER_NAMES


class CCA:
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    def _center_data(self, x: np.ndarray) -> np.ndarray:
        return x - x.mean(axis=0, keepdims=True)

    def compute_similarity(
        self,
        x: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        return_correlations: bool = False,
    ) -> Union[float, tuple]:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        if y.ndim > 2:
            y = y.reshape(y.shape[0], -1)

        x = self._center_data(x)
        y = self._center_data(y)
        qx, _ = np.linalg.qr(x)
        qy, _ = np.linalg.qr(y)

        try:
            _, correlations, _ = svd(qx.T @ qy)
            correlations = np.clip(correlations, 0, 1)
            correlations = correlations[: min(x.shape[1], y.shape[1])]
            mean_correlation = float(np.mean(correlations))
        except np.linalg.LinAlgError:
            correlations = np.zeros(min(x.shape[1], y.shape[1]))
            mean_correlation = 0.0

        if return_correlations:
            return mean_correlation, correlations
        return mean_correlation


class CKA:
    def __init__(self, kernel: str = "linear"):
        self.kernel = kernel

    def _center_gram(self, gram: np.ndarray) -> np.ndarray:
        means = gram.mean(axis=0, keepdims=True)
        return gram - means - means.T + means.mean()

    def _linear_kernel(self, x: np.ndarray) -> np.ndarray:
        return x @ x.T

    def _rbf_kernel(self, x: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
        n_samples = x.shape[0]
        if sigma is None:
            max_subset = min(1000, n_samples)
            rng = np.random.RandomState(42)
            subset_idx = rng.choice(n_samples, size=max_subset, replace=False)
            x_subset = x[subset_idx]
            x_sq = np.sum(x_subset**2, axis=1, keepdims=True)
            sq_dists = x_sq + x_sq.T - 2 * x_subset @ x_subset.T
            sq_dists = np.maximum(sq_dists, 0)
            sigma = np.median(sq_dists[sq_dists > 0])
            if sigma == 0:
                sigma = 1.0

        x_sq = np.sum(x**2, axis=1, keepdims=True)
        sq_dists = x_sq + x_sq.T - 2 * x @ x.T
        sq_dists = np.maximum(sq_dists, 0)
        return np.exp(-sq_dists / (2 * sigma))

    def compute_similarity(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> float:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        if y.ndim > 2:
            y = y.reshape(y.shape[0], -1)

        if self.kernel == "linear":
            k = self._linear_kernel(x)
            l = self._linear_kernel(y)
        else:
            k = self._rbf_kernel(x)
            l = self._rbf_kernel(y)

        k = self._center_gram(k)
        l = self._center_gram(l)
        hsic = np.sum(k * l)
        normalization = np.sqrt(np.sum(k * k) * np.sum(l * l))
        return float(hsic / normalization) if normalization > 0 else 0.0


class EuclideanDistance:
    def compute_similarity(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> float:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        if y.ndim > 2:
            y = y.reshape(y.shape[0], -1)
        if x.shape[1] != y.shape[1]:
            raise ValueError("x and y must have the same number of features")

        x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        y_norm = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-8)
        return float(np.mean(np.linalg.norm(x_norm - y_norm, axis=1)))


class CosineSimilarity:
    def compute_similarity(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> float:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        if y.ndim > 2:
            y = y.reshape(y.shape[0], -1)
        if x.shape[1] != y.shape[1]:
            raise ValueError("x and y must have the same number of features")

        x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        y_norm = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-8)
        return float(np.mean(np.sum(x_norm * y_norm, axis=1)))


class KLDivergence:
    def __init__(self, epsilon: float = 1e-8, symmetric: bool = True):
        self.epsilon = epsilon
        self.symmetric = symmetric

    def _to_probability_distribution(self, x: np.ndarray) -> np.ndarray:
        x_abs = np.abs(x)
        x_stable = x_abs + self.epsilon
        return x_stable / np.sum(x_stable, axis=1, keepdims=True)

    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        q_stable = q + self.epsilon
        return float(np.mean(np.sum(p * np.log(p / q_stable), axis=1)))

    def compute_similarity(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> float:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        if y.ndim > 2:
            y = y.reshape(y.shape[0], -1)
        if x.shape[1] != y.shape[1]:
            raise ValueError("x and y must have the same number of features")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples")

        p = self._to_probability_distribution(x)
        q = self._to_probability_distribution(y)
        if self.symmetric:
            return 0.5 * (self._kl_divergence(p, q) + self._kl_divergence(q, p))
        return self._kl_divergence(p, q)


def build_default_metrics() -> Dict[str, object]:
    return {
        "cka_linear": CKA(kernel="linear"),
        "cca": CCA(),
        "cosine": CosineSimilarity(),
        "euclidean": EuclideanDistance(),
        "kl_sym": KLDivergence(symmetric=True),
    }


HIGHER_BETTER_METRICS = ("cka_linear", "cca", "cosine")
LOWER_BETTER_METRICS = ("euclidean", "kl_sym")


def _amp_dtype_for_device(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float16
    return torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16


def collect_model_activations(
    model: torch.nn.Module,
    loader,
    layers: Iterable[str] = DEFAULT_LAYER_NAMES,
    device: torch.device = torch.device("cpu"),
    max_batches: Optional[int] = 10,
) -> Dict[str, np.ndarray]:
    modules = dict(model.named_modules())
    missing = [layer_name for layer_name in layers if layer_name not in modules]
    if missing:
        raise ValueError(f"Missing layer names in model: {missing}")

    acts: Dict[str, List[torch.Tensor]] = defaultdict(list)
    hooks = []

    def hook_factory(layer_name: str):
        def hook(_module, _inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            out = output.detach().cpu()
            if out.ndim > 2:
                out = out.mean(dim=tuple(range(2, out.ndim)))
            acts[layer_name].append(out)

        return hook

    for layer_name in layers:
        hooks.append(modules[layer_name].register_forward_hook(hook_factory(layer_name)))

    amp_enabled = device.type == "cuda"
    amp_dtype = _amp_dtype_for_device(device)
    with torch.no_grad():
        for batch_idx, (inputs, _labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs = inputs.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                _ = model(inputs)

    for hook in hooks:
        hook.remove()

    result = {}
    for layer_name in layers:
        result[layer_name] = torch.cat(acts[layer_name], dim=0).to(torch.float32).numpy()
    return result


def evaluate_pair_rows(
    acts_a: Dict[str, np.ndarray],
    acts_b: Dict[str, np.ndarray],
    layers: Iterable[str] = DEFAULT_LAYER_NAMES,
    metrics: Optional[Dict[str, object]] = None,
) -> List[dict]:
    metrics = metrics or build_default_metrics()
    rows = []
    for layer in layers:
        x = acts_a[layer]
        y = acts_b[layer]
        row = {"layer": layer, "n_samples": int(x.shape[0]), "n_features": int(x.shape[1])}
        for metric_name, metric in metrics.items():
            row[metric_name] = float(metric.compute_similarity(x, y))
        rows.append(row)
    return rows


def transform_rows_for_plot(rows: List[dict], lower_better_metrics: Iterable[str] = LOWER_BETTER_METRICS) -> List[dict]:
    transformed = [{k: v for k, v in row.items()} for row in rows]
    for metric_name in lower_better_metrics:
        vals = np.array([row[metric_name] for row in rows], dtype=np.float64)
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        if vmax > vmin:
            sim_vals = 1.0 - ((vals - vmin) / (vmax - vmin))
        else:
            sim_vals = np.full_like(vals, 0.5)
        for idx in range(len(transformed)):
            transformed[idx][metric_name] = float(sim_vals[idx])
    return transformed
