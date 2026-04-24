from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import (
    default_num_workers,
    load_cifar10_datasets,
    make_forget_retain_subsets,
    make_loader,
    subset_for_class,
)
from .experiments import (
    CombinedComparisonConfig,
    CoreExperimentArtifacts,
    CoreExperimentConfig,
    TrajectoryExperimentArtifacts,
    TrajectoryExperimentConfig,
    ensure_wandb_run,
    run_core_checkpoints,
    run_trajectory_analysis,
    save_combined_trajectory_comparison,
)
from .models import Net
from .similarity import (
    DEFAULT_LAYER_NAMES,
    HIGHER_BETTER_METRICS,
    LOWER_BETTER_METRICS,
    build_default_metrics,
    collect_model_activations,
    evaluate_pair_rows,
    transform_rows_for_plot,
)
from .training import build_amp_config, configure_runtime
from .unlearning import GAConfig, SCRUBConfig, SSDConfig

ALGORITHM_ORDER = ["ga", "ssd", "salun", "certified", "scrub"]


@dataclass(frozen=True)
class NotebookRuntime:
    num_classes: int
    out_dir: str
    model_depth: int
    device: torch.device
    use_cuda: bool
    num_workers: int
    use_bf16: bool
    trainset: object
    testset: object
    model_factory: Callable[[], nn.Module]
    core_config: CoreExperimentConfig


@dataclass(frozen=True)
class SimilaritySetup:
    layer_names: list[str]
    metrics: dict[str, object]
    higher_better_metrics: list[str]
    lower_better_metrics: list[str]
    plot_metric_names: list[str]


def build_default_core_config(
    num_classes: int = 10,
    model_depth: int = 50,
    out_dir: str = "out",
    use_bf16: bool = False,
) -> CoreExperimentConfig:
    """Return the notebook's default baseline configuration bundle."""
    # GA now uses a sharper but shorter schedule. The method is still plain
    # gradient ascent on the forget set, but instead of a long run over every
    # forget batch each epoch, we take somewhat larger steps for fewer capped
    # updates. In practice this tends to damage the target class more directly
    # while reducing the slow collateral drift on the retained classes.
    ga_config = GAConfig(
        lr=6e-5,
        epochs=14,
        max_batches_per_epoch=12,
        freeze_bn=True,
        grad_clip_norm=7.5,
    )
    # SSD can become effectively a no-op if the selection threshold is a little
    # too strict for the current checkpoint, so the notebook preset keeps the
    # same one-shot rule but uses a slightly more permissive threshold to make
    # sure some forget-dominated parameters are actually selected and damped.
    ssd_config = SSDConfig(
        alpha=6.5,
        lambda_=0.85,
        eps=1e-12,
        fisher_batches=100,
        fisher_samples_per_batch=32,
        selection_basis="retain",
    )
    # The earlier SCRUB preset still hit the model too hard in the initial
    # forget phase, producing unrealistic collapses before recovery. This
    # notebook preset keeps the same teacher-student idea but makes the scrub
    # phase much shorter and smaller, then gives the retain-only recovery phase
    # more room to stabilise the non-forget classes.
    scrub_config = SCRUBConfig(
        lr=5e-5,
        epochs=12,
        forget_phase_epochs=1,
        max_forget_batches_per_epoch=6,
        alpha=4.0,
        beta=0.15,
        gamma=4.0,
        temperature=2.0,
        weight_decay=1e-4,
        grad_clip_norm=0.5,
    )
    return CoreExperimentConfig(
        num_classes=num_classes,
        model_depth=model_depth,
        out_dir=out_dir,
        num_epochs=10,
        unlearn_batch_size=512 if use_bf16 else 256,
        ga_config=ga_config,
        ssd_config=ssd_config,
        scrub_config=scrub_config,
    )


def build_core_wandb_config(config: CoreExperimentConfig) -> dict[str, object]:
    return {
        "target_label": config.target_label,
        "num_epochs": config.num_epochs,
        "batch_size": config.batch_size,
        "model_depth": config.model_depth,
        "ga_lr": config.ga_config.lr,
        "ga_epochs": config.ga_config.epochs,
        "ga_max_batches_per_epoch": config.ga_config.max_batches_per_epoch,
        "ga_grad_clip_norm": config.ga_config.grad_clip_norm,
        "ssd_alpha": config.ssd_config.alpha,
        "ssd_lambda": config.ssd_config.lambda_,
        "ssd_fisher_batches": config.ssd_config.fisher_batches,
        "scrub_lr": config.scrub_config.lr,
        "scrub_epochs": config.scrub_config.epochs,
        "scrub_forget_phase_epochs": config.scrub_config.forget_phase_epochs,
        "scrub_max_forget_batches_per_epoch": config.scrub_config.max_forget_batches_per_epoch,
        "scrub_alpha": config.scrub_config.alpha,
        "scrub_beta": config.scrub_config.beta,
        "scrub_gamma": config.scrub_config.gamma,
        "scrub_temperature": config.scrub_config.temperature,
    }


def prepare_notebook_runtime(
    num_classes: int = 10,
    out_dir: str = "out",
    model_depth: int = 50,
    data_root: str = "./data",
) -> NotebookRuntime:
    configure_runtime()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_config = build_amp_config(device)
    use_bf16 = amp_config.dtype == torch.bfloat16

    trainset, testset = load_cifar10_datasets(root=data_root)
    use_cuda = device.type == "cuda"
    num_workers = default_num_workers(use_cuda)
    model_factory = lambda: Net(num_classes=num_classes, pretrained=False, model_depth=model_depth).to(device)
    core_config = build_default_core_config(
        num_classes=num_classes,
        model_depth=model_depth,
        out_dir=out_dir,
        use_bf16=use_bf16,
    )
    return NotebookRuntime(
        num_classes=num_classes,
        out_dir=out_dir,
        model_depth=model_depth,
        device=device,
        use_cuda=use_cuda,
        num_workers=num_workers,
        use_bf16=use_bf16,
        trainset=trainset,
        testset=testset,
        model_factory=model_factory,
        core_config=core_config,
    )


def run_notebook_core_experiment(
    runtime: NotebookRuntime,
    wandb_module=None,
    wandb_project: str = "gradient-ascent",
    wandb_name: str = "core-checkpoints",
) -> tuple[CoreExperimentArtifacts, object]:
    wandb_run = ensure_wandb_run(
        wandb_module,
        project=wandb_project,
        name=wandb_name,
        config=build_core_wandb_config(runtime.core_config),
    )
    artifacts = run_core_checkpoints(
        model_factory=runtime.model_factory,
        trainset=runtime.trainset,
        testset=runtime.testset,
        device=runtime.device,
        use_cuda=runtime.use_cuda,
        num_workers=runtime.num_workers,
        config=runtime.core_config,
        wandb_run=wandb_run,
        wandb_module=wandb_module,
    )
    return artifacts, wandb_run


def prepare_similarity_setup() -> SimilaritySetup:
    """Return the notebook's default similarity-analysis configuration."""
    layer_names = list(DEFAULT_LAYER_NAMES)
    metrics = build_default_metrics()
    higher_better_metrics = list(HIGHER_BETTER_METRICS)
    lower_better_metrics = list(LOWER_BETTER_METRICS)
    plot_metric_names = higher_better_metrics + lower_better_metrics
    return SimilaritySetup(
        layer_names=layer_names,
        metrics=metrics,
        higher_better_metrics=higher_better_metrics,
        lower_better_metrics=lower_better_metrics,
        plot_metric_names=plot_metric_names,
    )


def build_default_trajectory_config(runtime: NotebookRuntime) -> TrajectoryExperimentConfig:
    return TrajectoryExperimentConfig(
        num_classes=runtime.num_classes,
        model_depth=runtime.model_depth,
        out_dir=runtime.out_dir,
        trajectory_batch_size=512 if runtime.use_bf16 else 256,
    )


def run_notebook_trajectory_experiment(
    runtime: NotebookRuntime,
    core_artifacts: CoreExperimentArtifacts,
    similarity_setup: SimilaritySetup,
    wandb_module=None,
    wandb_project: str = "gradient-ascent",
    wandb_name: str = "unlearning-algorithm-comparison",
) -> tuple[TrajectoryExperimentArtifacts, object]:
    """Run the notebook's full trajectory/MIA/similarity pipeline."""
    trajectory_config = build_default_trajectory_config(runtime)
    wandb_run = ensure_wandb_run(wandb_module, project=wandb_project, name=wandb_name)
    artifacts = run_trajectory_analysis(
        model_factory=runtime.model_factory,
        trainset=runtime.trainset,
        testset=runtime.testset,
        device=runtime.device,
        use_cuda=runtime.use_cuda,
        num_workers=runtime.num_workers,
        config=trajectory_config,
        original_checkpoint_path=f"{runtime.out_dir}/original_net.pt",
        retrained_checkpoint_path=f"{runtime.out_dir}/retrained_from_scratch_net.pt",
        snapshot_dirs={
            algorithm_key: core_artifacts.algorithm_artifacts[algorithm_key].snapshot_dir
            for algorithm_key in ALGORITHM_ORDER
        },
        layer_names=similarity_setup.layer_names,
        metric_names=similarity_setup.plot_metric_names,
        lower_better_metrics=similarity_setup.lower_better_metrics,
        activation_collector=lambda model, loader: collect_model_activations(
            model,
            loader,
            similarity_setup.layer_names,
            runtime.device,
            max_batches=trajectory_config.max_batches_for_similarity,
        ),
        pair_evaluator=lambda acts_a, acts_b: evaluate_pair_rows(
            acts_a,
            acts_b,
            layers=similarity_setup.layer_names,
            metrics=similarity_setup.metrics,
        ),
        transform_rows_for_plot=lambda rows: transform_rows_for_plot(
            rows, lower_better_metrics=similarity_setup.lower_better_metrics
        ),
        wandb_run=wandb_run,
        wandb_module=wandb_module,
    )
    return artifacts, wandb_run


def save_notebook_combined_comparison(
    runtime: NotebookRuntime,
    wandb_module=None,
    wandb_project: str = "gradient-ascent",
    wandb_name: str = "unlearning-algorithm-comparison",
) -> str:
    """Build the combined cross-algorithm similarity + MIA figure."""
    wandb_run = ensure_wandb_run(wandb_module, project=wandb_project, name=wandb_name)
    return save_combined_trajectory_comparison(
        CombinedComparisonConfig(out_dir=runtime.out_dir),
        wandb_run=wandb_run,
        wandb_module=wandb_module,
    )


def _compute_mean_forget_loss_and_grad_norm(model, loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_samples = 0
    grad_norms = []
    params = [param for param in model.parameters() if param.requires_grad]

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        model.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)

        sq_norm = 0.0
        for grad in grads:
            sq_norm += grad.detach().float().pow(2).sum().item()
        grad_norms.append(math.sqrt(sq_norm))

        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return {
        "forget_loss": total_loss / max(total_samples, 1),
        "forget_grad_norm": sum(grad_norms) / max(len(grad_norms), 1),
    }


def save_ga_diagnostics(
    runtime: NotebookRuntime,
    core_artifacts: CoreExperimentArtifacts,
    wandb_run=None,
    wandb_module=None,
) -> tuple[str, str]:
    forget_subset, _ = make_forget_retain_subsets(runtime.trainset, runtime.core_config.target_label)
    diagnostic_loader = make_loader(
        forget_subset,
        batch_size=min(256, runtime.core_config.unlearn_batch_size or 256),
        shuffle=False,
        num_workers=runtime.num_workers,
        use_cuda=runtime.use_cuda,
    )

    snapshot_dir = Path(core_artifacts.algorithm_artifacts["ga"].snapshot_dir)
    snapshot_paths = sorted(snapshot_dir.glob("epoch_*.pt"))
    if not snapshot_paths:
        raise RuntimeError(f"No GA snapshots found under {snapshot_dir}")

    rows = []
    for snapshot_path in snapshot_paths:
        epoch = int(snapshot_path.stem.split("_")[-1])
        model = runtime.model_factory()
        model.load_state_dict(torch.load(snapshot_path, map_location=runtime.device))
        diagnostics = _compute_mean_forget_loss_and_grad_norm(model, diagnostic_loader, runtime.device)
        rows.append({"epoch": epoch, **diagnostics})

    csv_path = Path(runtime.out_dir) / "ga_diagnostics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "forget_loss", "forget_grad_norm"])
        writer.writeheader()
        writer.writerows(rows)

    plot_path = Path(runtime.out_dir) / "ga_diagnostics.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    epochs = [row["epoch"] for row in rows]
    axes[0].plot(epochs, [row["forget_loss"] for row in rows], marker="o", linewidth=2)
    axes[0].set_title("GA forget-set loss")
    axes[0].set_xlabel("Unlearning step")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [row["forget_grad_norm"] for row in rows], marker="o", linewidth=2)
    axes[1].set_title("GA forget-set gradient norm")
    axes[1].set_xlabel("Unlearning step")
    axes[1].set_ylabel("L2 norm")
    axes[1].grid(alpha=0.3)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    if wandb_run is not None and wandb_module is not None:
        wandb_run.log(
            {
                "plots/ga_diagnostics": wandb_module.Image(str(plot_path)),
                "tables/ga_diagnostics": wandb_module.Table(
                    data=[[row["epoch"], row["forget_loss"], row["forget_grad_norm"]] for row in rows],
                    columns=["epoch", "forget_loss", "forget_grad_norm"],
                ),
            }
        )

    return str(csv_path), str(plot_path)


def _temperature_kl(student_logits, teacher_logits, temperature):
    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits.float() / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="sum") * (temperature ** 2)


@torch.inference_mode()
def _compute_scrub_snapshot_diagnostics(student_model, teacher_model, forget_loader, retain_loader, device, temperature):
    student_model.eval()
    teacher_model.eval()

    forget_kl_sum = 0.0
    forget_correct = 0
    forget_total = 0
    for inputs, labels in forget_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        student_logits = student_model(inputs)
        teacher_logits = teacher_model(inputs)
        forget_kl_sum += float(_temperature_kl(student_logits, teacher_logits, temperature).item())
        forget_correct += int((student_logits.argmax(dim=1) == labels).sum().item())
        forget_total += int(labels.numel())

    retain_kl_sum = 0.0
    retain_ce_sum = 0.0
    retain_correct = 0
    retain_total = 0
    for inputs, labels in retain_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        student_logits = student_model(inputs)
        teacher_logits = teacher_model(inputs)
        retain_kl_sum += float(_temperature_kl(student_logits, teacher_logits, temperature).item())
        retain_ce_sum += float(F.cross_entropy(student_logits.float(), labels, reduction="sum").item())
        retain_correct += int((student_logits.argmax(dim=1) == labels).sum().item())
        retain_total += int(labels.numel())

    return {
        "forget_kl": forget_kl_sum / max(forget_total, 1),
        "retain_kl": retain_kl_sum / max(retain_total, 1),
        "retain_ce": retain_ce_sum / max(retain_total, 1),
        "forget_acc": forget_correct / max(forget_total, 1),
        "retain_acc": retain_correct / max(retain_total, 1),
    }


def save_scrub_diagnostics(
    runtime: NotebookRuntime,
    core_artifacts: CoreExperimentArtifacts,
    wandb_run=None,
    wandb_module=None,
) -> tuple[str, str]:
    forget_subset, _ = make_forget_retain_subsets(runtime.trainset, runtime.core_config.target_label)
    forget_loader = make_loader(
        forget_subset,
        batch_size=min(256, runtime.core_config.unlearn_batch_size or 256),
        shuffle=False,
        num_workers=runtime.num_workers,
        use_cuda=runtime.use_cuda,
    )

    retain_subset = subset_for_class(runtime.trainset, runtime.core_config.target_label, include=False)
    retain_loader = make_loader(
        retain_subset,
        batch_size=min(
            256,
            runtime.core_config.scrub_config.batch_size or runtime.core_config.unlearn_batch_size or 256,
        ),
        shuffle=False,
        num_workers=runtime.num_workers,
        use_cuda=runtime.use_cuda,
    )

    teacher_model = runtime.model_factory()
    teacher_model.load_state_dict(torch.load(core_artifacts.original_checkpoint_path, map_location=runtime.device))
    teacher_model.eval()

    snapshot_dir = Path(core_artifacts.algorithm_artifacts["scrub"].snapshot_dir)
    snapshot_paths = sorted(snapshot_dir.glob("epoch_*.pt"))
    if not snapshot_paths:
        raise RuntimeError(f"No SCRUB snapshots found under {snapshot_dir}")

    rows = []
    for snapshot_path in snapshot_paths:
        epoch = int(snapshot_path.stem.split("_")[-1])
        student_model = runtime.model_factory()
        student_model.load_state_dict(torch.load(snapshot_path, map_location=runtime.device))
        diagnostics = _compute_scrub_snapshot_diagnostics(
            student_model,
            teacher_model,
            forget_loader,
            retain_loader,
            runtime.device,
            runtime.core_config.scrub_config.temperature,
        )
        rows.append({"epoch": epoch, **diagnostics})

    csv_path = Path(runtime.out_dir) / "scrub_diagnostics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "forget_kl", "retain_kl", "retain_ce", "forget_acc", "retain_acc"],
        )
        writer.writeheader()
        writer.writerows(rows)

    plot_path = Path(runtime.out_dir) / "scrub_diagnostics.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    epochs = [row["epoch"] for row in rows]

    phase_boundary = runtime.core_config.scrub_config.forget_phase_epochs + 0.5

    axes[0].plot(
        epochs,
        [100.0 * row["retain_acc"] for row in rows],
        marker="o",
        linewidth=2,
        label="retain accuracy",
    )
    axes[0].plot(
        epochs,
        [100.0 * row["forget_acc"] for row in rows],
        marker="s",
        linewidth=2,
        label="forget accuracy",
    )
    axes[0].axvline(phase_boundary, color="black", linestyle="--", linewidth=1, alpha=0.7)
    axes[0].set_title("SCRUB retain vs forget accuracy")
    axes[0].set_xlabel("Unlearning step")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    forget_kl = np.array([row["forget_kl"] for row in rows], dtype=np.float64)
    retain_kl = np.array([row["retain_kl"] for row in rows], dtype=np.float64)
    eps = 1e-8
    axes[1].plot(
        epochs,
        np.maximum(forget_kl, eps),
        marker="o",
        linewidth=2,
        label="forget KL",
    )
    axes[1].plot(
        epochs,
        np.maximum(retain_kl, eps),
        marker="s",
        linewidth=2,
        label="retain KL",
    )
    axes[1].axvline(phase_boundary, color="black", linestyle="--", linewidth=1, alpha=0.7)
    axes[1].set_yscale("log")
    axes[1].set_title("SCRUB divergence from teacher")
    axes[1].set_xlabel("Unlearning step")
    axes[1].set_ylabel("KL (log scale)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    if wandb_run is not None and wandb_module is not None:
        wandb_run.log(
            {
                "plots/scrub_diagnostics": wandb_module.Image(str(plot_path)),
                "tables/scrub_diagnostics": wandb_module.Table(
                    data=[
                        [
                            row["epoch"],
                            row["forget_kl"],
                            row["retain_kl"],
                            row["retain_ce"],
                            row["forget_acc"],
                            row["retain_acc"],
                        ]
                        for row in rows
                    ],
                    columns=["epoch", "forget_kl", "retain_kl", "retain_ce", "forget_acc", "retain_acc"],
                ),
            }
        )

    return str(csv_path), str(plot_path)

