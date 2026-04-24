from __future__ import annotations

import copy
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from gradient_ascent.experiments import CombinedComparisonConfig, CoreExperimentConfig
from gradient_ascent.unlearning import SCRUBConfig, run_scrub_unlearning


class TinyNet(torch.nn.Module):
    def __init__(self, in_dim: int = 4, hidden_dim: int = 8, num_classes: int = 3) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScrubSmokeTest(unittest.TestCase):
    def _make_loader(self, inputs: torch.Tensor, labels: torch.Tensor, batch_size: int = 4) -> DataLoader:
        return DataLoader(TensorDataset(inputs, labels), batch_size=batch_size, shuffle=False)

    def test_scrub_smoke_returns_expected_shape_and_preserves_teacher(self) -> None:
        torch.manual_seed(0)
        device = torch.device("cpu")

        retain_x = torch.randn(12, 4)
        retain_y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=torch.long)
        forget_x = torch.randn(6, 4)
        forget_y = torch.tensor([1, 1, 1, 2, 2, 2], dtype=torch.long)
        test_x = torch.randn(9, 4)
        test_y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=torch.long)

        retain_loader = self._make_loader(retain_x, retain_y)
        forget_loader = self._make_loader(forget_x, forget_y)
        test_loader = self._make_loader(test_x, test_y)

        student = TinyNet().to(device)
        teacher = copy.deepcopy(student).to(device)
        student_before = copy.deepcopy(student.state_dict())
        teacher_before = copy.deepcopy(teacher.state_dict())

        config = SCRUBConfig(
            lr=1e-2,
            epochs=2,
            alpha=1.0,
            beta=1.0,
            gamma=1.0,
            temperature=2.0,
            weight_decay=0.0,
            grad_clip_norm=1.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_scrub_unlearning(
                student,
                forget_loader,
                retain_loader,
                test_loader,
                device,
                config=config,
                num_classes=3,
                snapshot_dir=tmpdir,
                teacher_model=teacher,
            )

        self.assertEqual(set(result.keys()), {"model", "classwise_history", "snapshot_paths", "diagnostics"})
        self.assertEqual(len(result["classwise_history"]), config.epochs + 1)
        self.assertEqual(len(result["snapshot_paths"]), config.epochs + 1)
        self.assertEqual(len(result["diagnostics"]), config.epochs)

        teacher_after = teacher.state_dict()
        for name, before in teacher_before.items():
            self.assertTrue(torch.equal(before, teacher_after[name]), msg=f"Teacher changed at {name}")

        student_after = result["model"].state_dict()
        self.assertTrue(
            any(not torch.equal(student_before[name], student_after[name]) for name in student_before),
            msg="Student parameters did not change during SCRUB.",
        )

    def test_scrub_is_registered_in_default_experiment_configs(self) -> None:
        self.assertIn("scrub", CombinedComparisonConfig().algo_keys)
        self.assertEqual(CombinedComparisonConfig().algo_display["scrub"], "SCRUB")
        self.assertIsInstance(CoreExperimentConfig().scrub_config, SCRUBConfig)


if __name__ == "__main__":
    unittest.main()
