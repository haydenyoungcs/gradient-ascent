from __future__ import annotations

from collections import defaultdict
import time
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import torch

from .models import DEFAULT_LAYER_NAMES
from .reporting import min_max_to_similarity_01


def _cca_subsample_columns(
    x: np.ndarray,
    y: np.ndarray,
    max_columns: Optional[int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sub-sample feature columns before QR (SVCCA-style speed-up for wide layers)."""
    if max_columns is None:
        return x, y
    # When widths match, use the same column indices so activations stay aligned.
    if x.shape[1] == y.shape[1] and x.shape[1] > max_columns:
        rng = np.random.RandomState(seed)
        idx = rng.choice(x.shape[1], size=max_columns, replace=False)
        return x[:, idx], y[:, idx]

    rng_x = np.random.RandomState(seed)
    rng_y = np.random.RandomState(seed + 7919)
    if x.shape[1] > max_columns:
        idx_x = rng_x.choice(x.shape[1], size=max_columns, replace=False)
        x = x[:, idx_x]
    if y.shape[1] > max_columns:
        idx_y = rng_y.choice(y.shape[1], size=max_columns, replace=False)
        y = y[:, idx_y]
    return x, y


def _as_2d_float32(array: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    return np.ascontiguousarray(array, dtype=np.float32)


def _row_normalize(array: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    return array / (np.linalg.norm(array, axis=1, keepdims=True) + epsilon)


def _to_probability_rows(array: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    stable = np.abs(array) + epsilon
    return stable / np.sum(stable, axis=1, keepdims=True)


def _mean_symmetric_kl(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-8) -> float:
    q_stable = q + epsilon
    p_stable = p + epsilon
    kl_pq = np.mean(np.sum(p * np.log(p / q_stable), axis=1))
    kl_qp = np.mean(np.sum(q * np.log(q / p_stable), axis=1))
    return float(0.5 * (kl_pq + kl_qp))


def _needs_row_normalized(metrics: Dict[str, object]) -> bool:
    return any(name in metrics for name in ("cosine", "euclidean"))


def _needs_probability_rows(metrics: Dict[str, object]) -> bool:
    return "kl_sym" in metrics


def _build_activation_index(
    acts: Dict[str, np.ndarray],
    layers: Iterable[str],
    max_activation_samples: Optional[int],
    subsample_seed: int,
) -> Optional[np.ndarray]:
    layer_list = list(layers)
    if max_activation_samples is None or not layer_list:
        return None
    n = int(acts[layer_list[0]].shape[0])
    if n <= max_activation_samples:
        return None
    rng = np.random.RandomState(subsample_seed)
    return rng.choice(n, size=max_activation_samples, replace=False)


def _prepare_layer_array(
    array: np.ndarray,
    idx: Optional[np.ndarray],
    need_norm: bool,
    need_prob: bool,
) -> Dict[str, np.ndarray]:
    x = _as_2d_float32(array)
    if idx is not None:
        x = x[idx]
    prepared: Dict[str, np.ndarray] = {"raw": x}
    if need_norm:
        prepared["row_norm"] = _row_normalize(x)
    if need_prob:
        prepared["prob"] = _to_probability_rows(x)
    return prepared


def _cca_subsample_columns_with_optional_reference_idx(
    x: np.ndarray,
    max_columns: Optional[int],
    seed: int,
    reference_idx: Optional[np.ndarray],
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if max_columns is None or x.shape[1] <= max_columns:
        return x, None
    if reference_idx is not None and int(reference_idx.shape[0]) == int(max_columns):
        return x[:, reference_idx], reference_idx
    rng_x = np.random.RandomState(seed)
    idx = rng_x.choice(x.shape[1], size=max_columns, replace=False)
    return x[:, idx], idx


def prepare_activations_for_evaluation(
    acts: Dict[str, np.ndarray],
    *,
    layers: Iterable[str] = DEFAULT_LAYER_NAMES,
    metrics: Optional[Dict[str, object]] = None,
    max_activation_samples: Optional[int] = None,
    subsample_seed: int = 42,
    precompute_metric_reference_cache: bool = False,
) -> Dict[str, Dict[str, object]]:
    """Precompute per-layer transforms once so repeated similarity calls are cheap."""
    metrics = metrics or build_default_metrics()
    layer_list = list(layers)
    idx = _build_activation_index(acts, layer_list, max_activation_samples, subsample_seed)
    need_norm = _needs_row_normalized(metrics)
    need_prob = _needs_probability_rows(metrics)
    prepared_map = {}
    for layer in layer_list:
        prepared = _prepare_layer_array(acts[layer], idx=idx, need_norm=need_norm, need_prob=need_prob)
        if precompute_metric_reference_cache:
            metric_cache: Dict[str, object] = {}
            for metric_name, metric in metrics.items():
                if metric_name == "cca" and isinstance(metric, CCA):
                    metric_cache["cca"] = metric.prepare_reference(prepared["raw"])
                elif metric_name == "cka_linear" and isinstance(metric, CKA) and metric.kernel == "linear":
                    metric_cache["cka_linear"] = metric.prepare_linear_reference(prepared["raw"])
            if metric_cache:
                prepared["metric_reference_cache"] = metric_cache
        prepared_map[layer] = prepared
    return prepared_map


class CCA:
    def __init__(
        self,
        epsilon: float = 1e-8,
        max_columns: Optional[int] = None,
        column_subsample_seed: int = 43,
    ):
        self.epsilon = epsilon
        self.max_columns = max_columns
        self.column_subsample_seed = column_subsample_seed

    def _center_data(self, x: np.ndarray) -> np.ndarray:
        return x - x.mean(axis=0, keepdims=True)

    def compute_similarity(
        self,
        x: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        return_correlations: bool = False,
    ) -> Union[float, tuple]:
        x = _as_2d_float32(x)
        y = _as_2d_float32(y)
        x, y = _cca_subsample_columns(x, y, self.max_columns, self.column_subsample_seed)
        slice_cap = min(x.shape[1], y.shape[1])

        x = self._center_data(x)
        y = self._center_data(y)
        qx, _ = np.linalg.qr(x, mode="reduced")
        qy, _ = np.linalg.qr(y, mode="reduced")

        try:
            # Singular values only — no need to build the U/V factors.
            correlations = np.linalg.svd(qx.T @ qy, compute_uv=False)
            correlations = np.clip(correlations, 0, 1)
            correlations = correlations[:slice_cap]
            mean_correlation = float(np.mean(correlations)) if correlations.size > 0 else 0.0
        except np.linalg.LinAlgError:
            correlations = np.zeros(slice_cap)
            mean_correlation = 0.0

        if return_correlations:
            return mean_correlation, correlations
        return mean_correlation

    def prepare_reference(self, y: Union[np.ndarray, torch.Tensor]) -> Dict[str, np.ndarray | int]:
        y_arr = _as_2d_float32(y)
        y_idx = None
        if self.max_columns is not None and y_arr.shape[1] > self.max_columns:
            rng_y = np.random.RandomState(self.column_subsample_seed + 7919)
            y_idx = rng_y.choice(y_arr.shape[1], size=self.max_columns, replace=False)
            y_arr = y_arr[:, y_idx]
        y_centered = self._center_data(y_arr)
        qy, _ = np.linalg.qr(y_centered, mode="reduced")
        return {"qy": qy, "n_cols": int(y_arr.shape[1]), "col_idx": y_idx}

    def compute_similarity_to_prepared_reference(
        self,
        x: Union[np.ndarray, torch.Tensor],
        prepared_reference: Dict[str, np.ndarray | int],
        return_correlations: bool = False,
    ) -> Union[float, tuple]:
        x_arr = _as_2d_float32(x)
        reference_idx = prepared_reference.get("col_idx")
        if isinstance(reference_idx, np.ndarray):
            x_arr, _ = _cca_subsample_columns_with_optional_reference_idx(
                x_arr,
                self.max_columns,
                self.column_subsample_seed,
                reference_idx,
            )
        else:
            x_arr, _ = _cca_subsample_columns_with_optional_reference_idx(
                x_arr,
                self.max_columns,
                self.column_subsample_seed,
                None,
            )
        slice_cap = min(int(x_arr.shape[1]), int(prepared_reference["n_cols"]))

        x_arr = self._center_data(x_arr)
        qx, _ = np.linalg.qr(x_arr, mode="reduced")
        qy = prepared_reference["qy"]

        try:
            correlations = np.linalg.svd(qx.T @ qy, compute_uv=False)
            correlations = np.clip(correlations, 0, 1)
            correlations = correlations[:slice_cap]
            mean_correlation = float(np.mean(correlations)) if correlations.size > 0 else 0.0
        except np.linalg.LinAlgError:
            correlations = np.zeros(slice_cap)
            mean_correlation = 0.0

        if return_correlations:
            return mean_correlation, correlations
        return mean_correlation


def linear_cka_doubly_centered_gram(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA via the Frobenius form from Kornblith et al. (2019).

    Avoids building the full n×n Gram matrices, so it's cheaper when n is large.
    Rows of ``x`` and ``y`` must come from the same batch order.
    """
    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean(axis=0, keepdims=True)
    cross = xc.T @ yc
    num = float(np.sum(cross * cross))
    x_cov = xc.T @ xc
    y_cov = yc.T @ yc
    den = float(np.linalg.norm(x_cov, ord="fro") * np.linalg.norm(y_cov, ord="fro"))
    return num / den if den > 0.0 else 0.0


class CKA:
    def __init__(self, kernel: str = "linear"):
        self.kernel = kernel

    def _center_gram(self, gram: np.ndarray) -> np.ndarray:
        means = gram.mean(axis=0, keepdims=True)
        return gram - means - means.T + means.mean()

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
        x = _as_2d_float32(x)
        y = _as_2d_float32(y)

        if self.kernel == "linear":
            return linear_cka_doubly_centered_gram(x, y)

        k = self._rbf_kernel(x)
        l = self._rbf_kernel(y)
        k = self._center_gram(k)
        l = self._center_gram(l)
        hsic = np.sum(k * l)
        normalization = np.sqrt(np.sum(k * k) * np.sum(l * l))
        return float(hsic / normalization) if normalization > 0 else 0.0

    def prepare_linear_reference(self, y: Union[np.ndarray, torch.Tensor]) -> Dict[str, np.ndarray | float]:
        y_arr = _as_2d_float32(y)
        yc = y_arr - y_arr.mean(axis=0, keepdims=True)
        y_cov = yc.T @ yc
        y_cov_fro = float(np.linalg.norm(y_cov, ord="fro"))
        return {"yc": yc, "y_cov_fro": y_cov_fro}

    def compute_similarity_linear_to_prepared_reference(
        self,
        x: Union[np.ndarray, torch.Tensor],
        prepared_reference: Dict[str, np.ndarray | float],
    ) -> float:
        x_arr = _as_2d_float32(x)
        xc = x_arr - x_arr.mean(axis=0, keepdims=True)
        yc = prepared_reference["yc"]
        cross = xc.T @ yc
        num = float(np.sum(cross * cross))
        x_cov = xc.T @ xc
        den = float(np.linalg.norm(x_cov, ord="fro") * float(prepared_reference["y_cov_fro"]))
        return num / den if den > 0.0 else 0.0


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


def build_default_metrics(
    *,
    cca_max_columns: Optional[int] = 512,
    cca_column_subsample_seed: int = 43,
) -> Dict[str, object]:
    return {
        "cka_linear": CKA(kernel="linear"),
        "cca": CCA(max_columns=cca_max_columns, column_subsample_seed=cca_column_subsample_seed),
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
    max_activation_samples: Optional[int] = None,
    activation_subsample_seed: int = 42,
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
            # Pool spatial dims on-device to keep CPU transfers small.
            out = output.detach()
            if out.ndim > 2:
                out = out.mean(dim=tuple(range(2, out.ndim)))
            out = out.cpu()
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

    layer_list = list(layers)
    result: Dict[str, np.ndarray] = {}
    for layer_name in layer_list:
        result[layer_name] = torch.cat(acts[layer_name], dim=0).to(torch.float32).numpy()

    if max_activation_samples is not None and layer_list:
        n = int(result[layer_list[0]].shape[0])
        if n > max_activation_samples:
            rng = np.random.RandomState(activation_subsample_seed)
            idx = rng.choice(n, size=max_activation_samples, replace=False)
            for layer_name in layer_list:
                result[layer_name] = result[layer_name][idx]

    return result


def evaluate_pair_rows(
    acts_a: Dict[str, np.ndarray],
    acts_b: Dict[str, np.ndarray],
    layers: Iterable[str] = DEFAULT_LAYER_NAMES,
    metrics: Optional[Dict[str, object]] = None,
    max_activation_samples: Optional[int] = None,
    subsample_seed: int = 42,
    metric_timing_seconds: Optional[Dict[str, float]] = None,
) -> List[dict]:
    metrics = metrics or build_default_metrics()
    layer_list = list(layers)
    need_norm = _needs_row_normalized(metrics)
    need_prob = _needs_probability_rows(metrics)

    idx_a = _build_activation_index(acts_a, layer_list, max_activation_samples, subsample_seed)
    idx_b = _build_activation_index(acts_b, layer_list, max_activation_samples, subsample_seed)

    rows = []
    for layer in layer_list:
        x_prepared = _prepare_layer_array(acts_a[layer], idx=idx_a, need_norm=need_norm, need_prob=need_prob)
        y_prepared = _prepare_layer_array(acts_b[layer], idx=idx_b, need_norm=need_norm, need_prob=need_prob)
        x = x_prepared["raw"]
        y = y_prepared["raw"]
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Layer {layer}: mismatched sample counts {x.shape[0]} vs {y.shape[0]}")
        row = {"layer": layer, "n_samples": int(x.shape[0]), "n_features": int(x.shape[1])}
        for metric_name, metric in metrics.items():
            metric_t0 = time.perf_counter()
            if metric_name == "cosine":
                row[metric_name] = float(
                    np.mean(np.sum(x_prepared["row_norm"] * y_prepared["row_norm"], axis=1))
                )
            elif metric_name == "euclidean":
                row[metric_name] = float(
                    np.mean(np.linalg.norm(x_prepared["row_norm"] - y_prepared["row_norm"], axis=1))
                )
            elif metric_name == "kl_sym":
                row[metric_name] = _mean_symmetric_kl(x_prepared["prob"], y_prepared["prob"])
            else:
                row[metric_name] = float(metric.compute_similarity(x, y))
            if metric_timing_seconds is not None:
                metric_timing_seconds[metric_name] = (
                    float(metric_timing_seconds.get(metric_name, 0.0)) + (time.perf_counter() - metric_t0)
                )
        rows.append(row)
    return rows


def evaluate_pair_rows_prepared(
    prepared_acts_a: Dict[str, Dict[str, object]],
    prepared_acts_b: Dict[str, Dict[str, object]],
    layers: Iterable[str] = DEFAULT_LAYER_NAMES,
    metrics: Optional[Dict[str, object]] = None,
    metric_timing_seconds: Optional[Dict[str, float]] = None,
) -> List[dict]:
    """Score similarity using precomputed activation transforms for both models."""
    metrics = metrics or build_default_metrics()
    layer_list = list(layers)
    rows = []
    for layer in layer_list:
        x_prepared = prepared_acts_a[layer]
        y_prepared = prepared_acts_b[layer]
        x = x_prepared["raw"]
        y = y_prepared["raw"]
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Layer {layer}: mismatched sample counts {x.shape[0]} vs {y.shape[0]}")
        row = {"layer": layer, "n_samples": int(x.shape[0]), "n_features": int(x.shape[1])}
        for metric_name, metric in metrics.items():
            metric_t0 = time.perf_counter()
            if metric_name == "cosine":
                row[metric_name] = float(np.mean(np.sum(x_prepared["row_norm"] * y_prepared["row_norm"], axis=1)))
            elif metric_name == "euclidean":
                row[metric_name] = float(np.mean(np.linalg.norm(x_prepared["row_norm"] - y_prepared["row_norm"], axis=1)))
            elif metric_name == "kl_sym":
                row[metric_name] = _mean_symmetric_kl(x_prepared["prob"], y_prepared["prob"])
            elif (
                metric_name == "cca"
                and isinstance(metric, CCA)
                and "metric_reference_cache" in y_prepared
                and "cca" in y_prepared["metric_reference_cache"]
            ):
                row[metric_name] = float(
                    metric.compute_similarity_to_prepared_reference(
                        x,
                        y_prepared["metric_reference_cache"]["cca"],
                    )
                )
            elif (
                metric_name == "cka_linear"
                and isinstance(metric, CKA)
                and metric.kernel == "linear"
                and "metric_reference_cache" in y_prepared
                and "cka_linear" in y_prepared["metric_reference_cache"]
            ):
                row[metric_name] = float(
                    metric.compute_similarity_linear_to_prepared_reference(
                        x,
                        y_prepared["metric_reference_cache"]["cka_linear"],
                    )
                )
            else:
                row[metric_name] = float(metric.compute_similarity(x, y))
            if metric_timing_seconds is not None:
                metric_timing_seconds[metric_name] = (
                    float(metric_timing_seconds.get(metric_name, 0.0)) + (time.perf_counter() - metric_t0)
                )
        rows.append(row)
    return rows


_ROW_METADATA_KEYS = frozenset({"layer", "n_samples", "n_features", "epoch"})


def transform_rows_for_plot(rows: List[dict], lower_better_metrics: Iterable[str] = LOWER_BETTER_METRICS) -> List[dict]:
    """Min–max each metric across ``rows`` to [0, 1] so 1 = most similar."""
    lower = set(lower_better_metrics)
    transformed = [{k: v for k, v in row.items()} for row in rows]
    if not rows:
        return transformed
    metric_keys = [key for key in rows[0].keys() if key not in _ROW_METADATA_KEYS]
    for metric_name in metric_keys:
        vals = np.array([row[metric_name] for row in rows], dtype=np.float64)
        scaled = min_max_to_similarity_01(vals, lower_is_more_similar=(metric_name in lower))
        for idx in range(len(transformed)):
            transformed[idx][metric_name] = float(scaled[idx])
    return transformed
