from __future__ import annotations

import csv
import os
import time
from typing import Callable, Dict, Optional

import numpy as np
import torch

from ..data import (
    CIFAR10_CLASSES,
    make_forget_retain_subsets,
    make_loader,
    subset_for_class,
)
from ..reporting import (
    save_accuracy_history_plot,
    save_classwise_accuracy_bar_plot,
    save_classwise_absolute_accuracy_plot,
    save_classwise_history_csv,
    save_classwise_percent_change_plot,
    save_classwise_percent_difference_bar_plot,
    save_unlearning_runtime_bar_plot,
)
from ..training import evaluate, train_model
from ..unlearning import (
    run_certified_unlearning,
    run_ga_unlearning,
    run_salun_unlearning,
    run_scrub_unlearning,
    run_ssd_unlearning,
)

from .types import AlgorithmArtifacts, CoreExperimentArtifacts, CoreExperimentConfig
from .wandb_utils import log_media


def _save_classwise_artifacts(
    algorithm_key: str,
    algorithm_label: str,
    history,
    config: CoreExperimentConfig,
) -> AlgorithmArtifacts:
    history_tensor = np.asarray(history, dtype=np.float32)
    epochs = list(range(history_tensor.shape[0]))
    csv_path = f"{config.out_dir}/classwise_accuracy_{algorithm_key}.csv"
    percent_plot_path = f"{config.out_dir}/classwise_percent_change_{algorithm_key}.png"
    absolute_plot_path = f"{config.out_dir}/classwise_absolute_accuracy_{algorithm_key}.png"

    save_classwise_history_csv(epochs, history_tensor, CIFAR10_CLASSES, csv_path)
    save_classwise_percent_change_plot(
        history_tensor,
        CIFAR10_CLASSES,
        percent_plot_path,
        f"Relative Change in Class Accuracy During Unlearning ({algorithm_label}, ResNet-{config.model_depth})",
    )
    secondary_title = (
        f"Relative Change in Class Accuracy During Unlearning ({algorithm_label}, ResNet-{config.model_depth})"
        if len(epochs) > 3
        else f"Absolute Class Accuracy During Unlearning ({algorithm_label}, ResNet-{config.model_depth})"
    )
    save_classwise_absolute_accuracy_plot(
        epochs,
        history_tensor,
        CIFAR10_CLASSES,
        absolute_plot_path,
        secondary_title,
    )

    snapshot_dir = f"{config.out_dir}/unlearning_snapshots_{algorithm_key}"
    final_checkpoint_path = f"{config.out_dir}/unlearned_net_{algorithm_key}.pt"
    return AlgorithmArtifacts(
        snapshot_dir=snapshot_dir,
        final_checkpoint_path=final_checkpoint_path,
        classwise_history_csv_path=csv_path,
        classwise_percent_plot_path=percent_plot_path,
        classwise_absolute_plot_path=absolute_plot_path,
    )


def run_core_checkpoints(
    model_factory: Callable[[], torch.nn.Module],
    trainset,
    testset,
    device: torch.device,
    use_cuda: bool,
    num_workers: int,
    config: Optional[CoreExperimentConfig] = None,
    reuse_existing_checkpoints: bool = False,
    reuse_original_checkpoint: Optional[bool] = None,
    reuse_retrained_checkpoint: Optional[bool] = None,
    reuse_unlearned_checkpoints: Optional[bool] = None,
    wandb_run=None,
    wandb_module=None,
) -> CoreExperimentArtifacts:
    config = config or CoreExperimentConfig()
    os.makedirs(config.out_dir, exist_ok=True)

    trainloader = make_loader(trainset, config.batch_size, True, num_workers, use_cuda)
    testloader = make_loader(testset, config.batch_size, False, num_workers, use_cuda)

    original_path = f"{config.out_dir}/original_net.pt"
    retrained_path = f"{config.out_dir}/retrained_from_scratch_net.pt"
    original_vs_retrain_plot_path = f"{config.out_dir}/original_vs_retrain_acc.png"

    keep_subset = subset_for_class(trainset, config.target_label, include=False)
    keep_loader = make_loader(keep_subset, config.batch_size, True, num_workers, use_cuda)
    reuse_original = reuse_existing_checkpoints if reuse_original_checkpoint is None else reuse_original_checkpoint
    reuse_retrained = (
        reuse_existing_checkpoints if reuse_retrained_checkpoint is None else reuse_retrained_checkpoint
    )
    reuse_unlearned = (
        reuse_existing_checkpoints if reuse_unlearned_checkpoints is None else reuse_unlearned_checkpoints
    )

    original_model = model_factory()
    loaded_original = reuse_original and os.path.exists(original_path)
    if loaded_original:
        original_model.load_state_dict(torch.load(original_path, map_location=device))
        original_acc, _ = evaluate(original_model, testloader, num_classes=config.num_classes, device=device)
        original_acc_history = [original_acc]
        print(f"Loaded existing original checkpoint from {original_path}")
    else:
        original_acc_history, _ = train_model(
            original_model,
            trainloader,
            testloader,
            num_epochs=config.num_epochs,
            device=device,
            lr=config.training_lr,
            momentum=config.training_momentum,
            weight_decay=config.weight_decay,
            save_path=original_path,
            log_prefix="original ",
            num_classes=config.num_classes,
            wandb_run=wandb_run,
            wandb_prefix="original",
            wandb_step_offset=0,
        )

    retrained_model = model_factory()
    loaded_retrained = reuse_retrained and os.path.exists(retrained_path)
    if loaded_retrained:
        retrained_model.load_state_dict(torch.load(retrained_path, map_location=device))
        retrained_acc, _ = evaluate(retrained_model, testloader, num_classes=config.num_classes, device=device)
        retrained_acc_history = [retrained_acc]
        print(f"Loaded existing retrained checkpoint from {retrained_path}")
    else:
        retrained_acc_history, _ = train_model(
            retrained_model,
            keep_loader,
            testloader,
            num_epochs=config.num_epochs,
            device=device,
            lr=config.training_lr,
            momentum=config.training_momentum,
            weight_decay=config.weight_decay,
            save_path=retrained_path,
            log_prefix="retrain ",
            num_classes=config.num_classes,
            wandb_run=wandb_run,
            wandb_prefix="retrain",
            wandb_step_offset=config.num_epochs,
        )

    original_vs_retrain_plot_path = save_accuracy_history_plot(
        {
            "Original (all data)": original_acc_history,
            "Retrained from scratch": retrained_acc_history,
        },
        original_vs_retrain_plot_path,
        f"Original and retrained-from-scratch test accuracy vs epoch (ResNet-{config.model_depth})",
    )
    log_media(wandb_run, wandb_module, "plots/original_vs_retrain", original_vs_retrain_plot_path)

    forget_subset, retain_subset = make_forget_retain_subsets(trainset, config.target_label)
    unlearn_batch_size = config.unlearn_batch_size or 256
    forget_loader = make_loader(forget_subset, unlearn_batch_size, True, num_workers, use_cuda)
    retain_loader = make_loader(retain_subset, unlearn_batch_size, True, num_workers, use_cuda)
    scrub_batch_size = config.scrub_config.batch_size or unlearn_batch_size
    scrub_forget_loader = (
        forget_loader
        if scrub_batch_size == unlearn_batch_size
        else make_loader(forget_subset, scrub_batch_size, True, num_workers, use_cuda)
    )
    scrub_retain_loader = (
        retain_loader
        if scrub_batch_size == unlearn_batch_size
        else make_loader(retain_subset, scrub_batch_size, True, num_workers, use_cuda)
    )

    unlearning_runtime_seconds: Dict[str, float] = {}
    baseline_results: Dict[str, dict] = {}
    algorithm_artifacts: Dict[str, AlgorithmArtifacts] = {}

    def _algorithm_artifact_paths(algorithm_key: str) -> AlgorithmArtifacts:
        return AlgorithmArtifacts(
            snapshot_dir=f"{config.out_dir}/unlearning_snapshots_{algorithm_key}",
            final_checkpoint_path=f"{config.out_dir}/unlearned_net_{algorithm_key}.pt",
            classwise_history_csv_path=f"{config.out_dir}/classwise_accuracy_{algorithm_key}.csv",
            classwise_percent_plot_path=f"{config.out_dir}/classwise_percent_change_{algorithm_key}.png",
            classwise_absolute_plot_path=f"{config.out_dir}/classwise_absolute_accuracy_{algorithm_key}.png",
        )

    def _can_reuse_algorithm_outputs(artifact: AlgorithmArtifacts) -> bool:
        if not os.path.exists(artifact.final_checkpoint_path):
            return False
        if not os.path.isdir(artifact.snapshot_dir):
            return False
        if not any(name.startswith("epoch_") and name.endswith(".pt") for name in os.listdir(artifact.snapshot_dir)):
            return False
        required_files = [
            artifact.classwise_history_csv_path,
            artifact.classwise_percent_plot_path,
            artifact.classwise_absolute_plot_path,
        ]
        return all(os.path.exists(path) for path in required_files)

    def _load_reused_result(algorithm_key: str, artifact: AlgorithmArtifacts) -> dict:
        model = model_factory()
        model.load_state_dict(torch.load(artifact.final_checkpoint_path, map_location=device))
        print(f"[Core] Reused cached {algorithm_key.upper()} outputs from {artifact.final_checkpoint_path}")
        return {"model": model}

    for algorithm_key in ["ga", "ssd", "salun", "certified", "scrub"]:
        artifact = _algorithm_artifact_paths(algorithm_key)
        if reuse_unlearned and _can_reuse_algorithm_outputs(artifact):
            baseline_results[algorithm_key] = _load_reused_result(algorithm_key, artifact)
            algorithm_artifacts[algorithm_key] = artifact
            unlearning_runtime_seconds[algorithm_key] = 0.0
            continue

        t0 = time.perf_counter()
        if algorithm_key == "ga":
            ga_model = model_factory()
            ga_model.load_state_dict(torch.load(original_path, map_location=device))
            print("[Core] Running GA unlearning...")
            result = run_ga_unlearning(
                ga_model,
                forget_loader,
                testloader,
                device,
                config=config.ga_config,
                num_classes=config.num_classes,
                snapshot_dir=artifact.snapshot_dir,
            )
            torch.save(result["model"].state_dict(), f"{config.out_dir}/unlearned_net.pt")
            print("[Core] GA complete")
            label = "GA"
        elif algorithm_key == "ssd":
            ssd_model = model_factory()
            ssd_model.load_state_dict(torch.load(original_path, map_location=device))
            print("[Core] Running SSD unlearning...")
            result = run_ssd_unlearning(
                ssd_model,
                forget_loader,
                retain_loader,
                testloader,
                device,
                config=config.ssd_config,
                num_classes=config.num_classes,
                snapshot_dir=artifact.snapshot_dir,
            )
            print("[Core] SSD complete")
            label = "SSD"
        elif algorithm_key == "salun":
            salun_model = model_factory()
            salun_model.load_state_dict(torch.load(original_path, map_location=device))
            print("[Core] Running SalUn unlearning...")
            result = run_salun_unlearning(
                salun_model,
                forget_loader,
                retain_loader,
                testloader,
                device,
                config=config.salun_config,
                num_classes=config.num_classes,
                snapshot_dir=artifact.snapshot_dir,
            )
            print("[Core] SalUn complete")
            label = "SalUn"
        elif algorithm_key == "certified":
            certified_model = model_factory()
            certified_model.load_state_dict(torch.load(original_path, map_location=device))
            print("[Core] Running Certified Removal unlearning...")
            result = run_certified_unlearning(
                certified_model,
                forget_loader,
                retain_loader,
                testloader,
                device,
                config=config.certified_config,
                num_classes=config.num_classes,
                snapshot_dir=artifact.snapshot_dir,
            )
            print("[Core] Certified Removal complete")
            label = "Certified"
        else:
            scrub_model = model_factory()
            scrub_model.load_state_dict(torch.load(original_path, map_location=device))
            print("[Core] Running SCRUB unlearning...")
            result = run_scrub_unlearning(
                scrub_model,
                scrub_forget_loader,
                scrub_retain_loader,
                testloader,
                device,
                config=config.scrub_config,
                num_classes=config.num_classes,
                snapshot_dir=artifact.snapshot_dir,
            )
            print("[Core] SCRUB complete")
            label = "SCRUB"

        unlearning_runtime_seconds[algorithm_key] = time.perf_counter() - t0
        torch.save(result["model"].state_dict(), artifact.final_checkpoint_path)
        baseline_results[algorithm_key] = result
        algorithm_artifacts[algorithm_key] = _save_classwise_artifacts(
            algorithm_key,
            label,
            result["classwise_history"],
            config,
        )
    for algorithm_key, artifact in algorithm_artifacts.items():
        log_media(wandb_run, wandb_module, f"plots/classwise_percent_change_{algorithm_key}", artifact.classwise_percent_plot_path)
        log_media(
            wandb_run,
            wandb_module,
            f"plots/classwise_absolute_accuracy_{algorithm_key}",
            artifact.classwise_absolute_plot_path,
        )

    unlearning_runtime_csv_path = f"{config.out_dir}/unlearning_runtime_seconds.csv"
    with open(unlearning_runtime_csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["algorithm", "seconds"])
        writer.writeheader()
        for algorithm_key, seconds in unlearning_runtime_seconds.items():
            writer.writerow({"algorithm": algorithm_key, "seconds": float(seconds)})
    unlearning_runtime_plot_path = save_unlearning_runtime_bar_plot(
        {key.upper(): value for key, value in unlearning_runtime_seconds.items()},
        f"{config.out_dir}/unlearning_runtime_seconds.png",
        title=f"Unlearning runtime by algorithm (ResNet-{config.model_depth})",
    )
    log_media(
        wandb_run,
        wandb_module,
        "plots/unlearning_runtime_seconds",
        unlearning_runtime_plot_path,
    )

    original_overall, original_per = evaluate(original_model, testloader, num_classes=config.num_classes, device=device)
    retrained_overall, retrained_per = evaluate(retrained_model, testloader, num_classes=config.num_classes, device=device)
    original_classwise_plot_path = save_classwise_accuracy_bar_plot(
        original_per,
        CIFAR10_CLASSES,
        f"{config.out_dir}/classwise_accuracy_original.png",
        f"Original model classwise accuracy (ResNet-{config.model_depth})",
    )
    retrained_classwise_plot_path = save_classwise_accuracy_bar_plot(
        retrained_per,
        CIFAR10_CLASSES,
        f"{config.out_dir}/classwise_accuracy_retrained.png",
        f"Retrained model classwise accuracy (ResNet-{config.model_depth})",
    )
    retrained_vs_original_percent_diff_plot_path = save_classwise_percent_difference_bar_plot(
        original_per,
        retrained_per,
        CIFAR10_CLASSES,
        f"{config.out_dir}/classwise_percent_diff_retrained_vs_original.png",
        f"Classwise % difference: retrained vs original (ResNet-{config.model_depth})",
    )
    log_media(wandb_run, wandb_module, "plots/classwise_accuracy_original", original_classwise_plot_path)
    log_media(wandb_run, wandb_module, "plots/classwise_accuracy_retrained", retrained_classwise_plot_path)
    log_media(
        wandb_run,
        wandb_module,
        "plots/classwise_percent_diff_retrained_vs_original",
        retrained_vs_original_percent_diff_plot_path,
    )
    summary_metrics = {
        "original_overall_acc": float(original_overall),
        "retrain_overall_acc": float(retrained_overall),
        f"class_{config.target_label}_original_acc": float(original_per[config.target_label]),
        f"class_{config.target_label}_retrain_acc": float(retrained_per[config.target_label]),
    }
    for algorithm_key, result in baseline_results.items():
        overall_acc, per_class_acc = evaluate(
            result["model"], testloader, num_classes=config.num_classes, device=device
        )
        summary_metrics[f"unlearned_{algorithm_key}_overall_acc"] = float(overall_acc)
        summary_metrics[f"class_{config.target_label}_unlearned_{algorithm_key}_acc"] = float(
            per_class_acc[config.target_label]
        )
    if wandb_run is not None:
        wandb_run.log({f"summary/{key}": value for key, value in summary_metrics.items()})

    return CoreExperimentArtifacts(
        original_checkpoint_path=original_path,
        retrained_checkpoint_path=retrained_path,
        original_vs_retrain_plot_path=original_vs_retrain_plot_path,
        original_classwise_plot_path=original_classwise_plot_path,
        retrained_classwise_plot_path=retrained_classwise_plot_path,
        retrained_vs_original_percent_diff_plot_path=retrained_vs_original_percent_diff_plot_path,
        algorithm_artifacts=algorithm_artifacts,
        unlearning_runtime_csv_path=unlearning_runtime_csv_path,
        unlearning_runtime_plot_path=unlearning_runtime_plot_path,
        unlearning_runtime_seconds={k: float(v) for k, v in unlearning_runtime_seconds.items()},
        summary_metrics=summary_metrics,
    )
