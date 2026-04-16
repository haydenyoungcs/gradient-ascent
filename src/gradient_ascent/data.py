from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset


CIFAR10_MEAN = (0.5, 0.5, 0.5)
CIFAR10_STD = (0.5, 0.5, 0.5)
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


def cifar10_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def load_cifar10_datasets(root: str = "./data") -> Tuple[Dataset, Dataset]:
    transform = cifar10_transform()
    trainset = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )
    testset = torchvision.datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=transform,
    )
    return trainset, testset


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


def sample_balanced_class_subsets(
    train_dataset: Dataset,
    test_dataset: Dataset,
    class_label: int,
    seed: int,
) -> Tuple[Subset, Subset, int]:
    train_targets = getattr(train_dataset, "targets", None)
    test_targets = getattr(test_dataset, "targets", None)
    if train_targets is None or test_targets is None:
        raise AttributeError("Datasets must expose a 'targets' attribute.")

    train_idx = [i for i, lbl in enumerate(train_targets) if int(lbl) == int(class_label)]
    test_idx = [i for i, lbl in enumerate(test_targets) if int(lbl) == int(class_label)]
    n = min(len(train_idx), len(test_idx))
    if n < 2:
        raise RuntimeError(
            f"Not enough samples for class {class_label}: train={len(train_idx)}, test={len(test_idx)}"
        )

    rng = np.random.default_rng(int(seed) + int(class_label))
    train_sel = sorted(rng.choice(train_idx, size=n, replace=False).tolist())
    test_sel = sorted(rng.choice(test_idx, size=n, replace=False).tolist())
    return Subset(train_dataset, train_sel), Subset(test_dataset, test_sel), n
