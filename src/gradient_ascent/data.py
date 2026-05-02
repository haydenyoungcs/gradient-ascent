from __future__ import annotations

import copy
import os
import time
from typing import Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError

import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset


# Same file as torchvision.datasets.CIFAR10 (see torchvision/datasets/cifar.py); mirrors must match MD5 tgz_md5.
DEFAULT_CIFAR10_ARCHIVE_URLS: Tuple[str, ...] = (
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz",
)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def cifar10_transform(train: bool = False) -> transforms.Compose:
    augmentations = []
    if train:
        augmentations = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    return transforms.Compose(
        augmentations
        + [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def _resolve_cifar10_download_urls(
    explicit: Optional[Sequence[str]],
) -> Tuple[str, ...]:
    """Pick download URLs: explicit argument, then env, then built-in defaults."""
    if explicit is not None:
        return tuple(explicit)
    env = os.environ.get("GRADIENT_ASCENT_CIFAR10_URLS", "").strip()
    if env:
        return tuple(u.strip() for u in env.split(",") if u.strip())
    return DEFAULT_CIFAR10_ARCHIVE_URLS


def load_cifar10_datasets(
    root: str = "./data",
    *,
    download: bool = True,
    download_retries: int = 6,
    download_retry_initial_delay_sec: float = 3.0,
    download_retry_max_delay_sec: float = 120.0,
    download_urls: Optional[Sequence[str]] = None,
) -> Tuple[Dataset, Dataset]:
    """Load CIFAR-10 train/test sets.

    When ``download=True``, the first run may fetch archives from the network.
    Transient failures (HTTP 503, timeouts, etc.) are retried with exponential backoff.

    If the default Toronto host is down, this function tries further URLs (HTTP variant
    and a listed mirror) that serve the same ``cifar-10-python.tar.gz`` as torchvision.
    Override order via ``download_urls=`` or env ``GRADIENT_ASCENT_CIFAR10_URLS`` (comma-separated).
    """
    retryable = (HTTPError, URLError, TimeoutError, ConnectionError)
    urls = _resolve_cifar10_download_urls(download_urls)

    cifar_cls = torchvision.datasets.CIFAR10
    saved_url = cifar_cls.url
    try:
        last_exc: BaseException | None = None
        for mirror_idx, mirror_url in enumerate(urls):
            cifar_cls.url = mirror_url
            if download and len(urls) > 1:
                print(
                    f"[gradient_ascent] CIFAR-10 download source {mirror_idx + 1}/{len(urls)}: {mirror_url}",
                    flush=True,
                )
            for attempt in range(max(1, download_retries)):
                try:
                    trainset = cifar_cls(
                        root=root,
                        train=True,
                        download=download,
                        transform=cifar10_transform(train=True),
                    )
                    testset = cifar_cls(
                        root=root,
                        train=False,
                        download=download,
                        transform=cifar10_transform(train=False),
                    )
                    return trainset, testset
                except retryable as exc:
                    last_exc = exc
                    if attempt >= download_retries - 1 or not download:
                        break
                    delay = min(
                        download_retry_max_delay_sec,
                        download_retry_initial_delay_sec * (2**attempt),
                    )
                    print(
                        "[gradient_ascent] CIFAR-10 download/load failed "
                        f"({type(exc).__name__}: {exc}); retrying in {delay:.0f}s "
                        f"(attempt {attempt + 1}/{download_retries})...",
                        flush=True,
                    )
                    time.sleep(delay)

            # This mirror exhausted retries; try next URL if any.
            if mirror_idx < len(urls) - 1 and download:
                print(
                    "[gradient_ascent] Switching to next CIFAR-10 mirror after repeated failures.",
                    flush=True,
                )

        assert last_exc is not None
        hint = (
            " The CIFAR-10 mirror sometimes returns HTTP 503; wait and retry, place a local "
            f"copy under {root!r} (folder cifar-10-batches-py/) and call with download=False, "
            "or set GRADIENT_ASCENT_CIFAR10_URLS to a comma-separated list of tarball URLs."
        )
        raise RuntimeError(
            f"Could not download or load CIFAR-10 after {download_retries} attempt(s) "
            f"per mirror ({len(urls)} source(s) tried).{hint}"
        ) from last_exc
    finally:
        cifar_cls.url = saved_url


def clone_dataset_with_eval_transform(dataset: Dataset) -> Dataset:
    """Return a shallow dataset clone that uses deterministic eval transforms.

    This is used by MIA probes so member examples are evaluated without random
    train-time augmentation (crop/flip), making member/non-member comparisons
    more stable and easier to interpret.
    """
    if isinstance(dataset, Subset):
        cloned_parent = clone_dataset_with_eval_transform(dataset.dataset)
        return Subset(cloned_parent, list(dataset.indices))

    if not hasattr(dataset, "transform"):
        raise TypeError("Dataset must expose a 'transform' attribute to clone eval view.")

    cloned = copy.copy(dataset)
    cloned.transform = cifar10_transform(train=False)
    return cloned


def default_num_workers(use_cuda: bool) -> int:
    return min(12, os.cpu_count() or 2) if use_cuda else 2


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    use_cuda: bool,
) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": use_cuda,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **kwargs)


def subset_for_class(dataset: Dataset, class_label: int, include: bool = True) -> Subset:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise AttributeError("Dataset must expose a 'targets' attribute.")

    if include:
        indices = [idx for idx, lbl in enumerate(targets) if int(lbl) == int(class_label)]
    else:
        indices = [idx for idx, lbl in enumerate(targets) if int(lbl) != int(class_label)]
    return Subset(dataset, indices)


def make_forget_retain_subsets(dataset: Dataset, target_label: int) -> Tuple[Subset, Subset]:
    forget_subset = subset_for_class(dataset, target_label, include=True)
    retain_subset = subset_for_class(dataset, target_label, include=False)
    return forget_subset, retain_subset


def sample_class_subsets(
    train_dataset: Dataset,
    test_dataset: Dataset,
    class_label: int,
    seed: int,
    balance_classes: bool = True,
) -> Tuple[Subset, Subset, int, int]:
    train_targets = getattr(train_dataset, "targets", None)
    test_targets = getattr(test_dataset, "targets", None)
    if train_targets is None or test_targets is None:
        raise AttributeError("Datasets must expose a 'targets' attribute.")

    train_idx = [i for i, lbl in enumerate(train_targets) if int(lbl) == int(class_label)]
    test_idx = [i for i, lbl in enumerate(test_targets) if int(lbl) == int(class_label)]
    if len(train_idx) < 2 or len(test_idx) < 2:
        raise RuntimeError(
            f"Not enough samples for class {class_label}: train={len(train_idx)}, test={len(test_idx)}"
        )

    rng = np.random.default_rng(int(seed) + int(class_label))
    if balance_classes:
        n = min(len(train_idx), len(test_idx))
        train_sel = sorted(rng.choice(train_idx, size=n, replace=False).tolist())
        test_sel = sorted(rng.choice(test_idx, size=n, replace=False).tolist())
    else:
        train_sel = sorted(rng.permutation(train_idx).tolist())
        test_sel = sorted(rng.permutation(test_idx).tolist())
    return (
        Subset(train_dataset, train_sel),
        Subset(test_dataset, test_sel),
        len(train_sel),
        len(test_sel),
    )


def sample_balanced_class_subsets(
    train_dataset: Dataset,
    test_dataset: Dataset,
    class_label: int,
    seed: int,
) -> Tuple[Subset, Subset, int]:
    member_subset, nonmember_subset, n_member, n_nonmember = sample_class_subsets(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        class_label=class_label,
        seed=seed,
        balance_classes=True,
    )
    if n_member != n_nonmember:
        raise RuntimeError("Balanced subset sampling produced mismatched sizes.")
    return member_subset, nonmember_subset, n_member
