"""Numerical contracts for the output-head backward-pass exercise."""
import importlib.util
from pathlib import Path
import unittest

import torch


class TrainingStepTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).resolve().parents[1] / "examples/training_step.py"
        self.assertTrue(path.exists(), "Output-head training exercise is missing")
        spec = importlib.util.spec_from_file_location("training_step", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_gradients_match_autograd_and_finite_differences(self):
        result = self.module().run_step()
        self.assertEqual(result["logits"].shape, (2, 3))
        torch.testing.assert_close(result["probabilities"].sum(-1),
                                   torch.ones(2, dtype=torch.float64))
        for name in ("x", "weight", "bias"):
            with self.subTest(name=name):
                torch.testing.assert_close(result["manual"][name],
                                           result["autograd"][name], rtol=0, atol=1e-12)
                torch.testing.assert_close(result["manual"][name],
                                           result["finite_difference"][name], rtol=0, atol=1e-8)

    def test_sgd_updates_parameters_and_reduces_fixed_batch_loss(self):
        result = self.module().run_step(lr=0.1)
        self.assertLess(result["loss_after"], result["loss_before"])
        for name in ("weight", "bias"):
            torch.testing.assert_close(result["updated"][name],
                                       result[name] - 0.1 * result["manual"][name])
        self.assertEqual(result["x"].shape, (2, 2))
        self.assertEqual(result["weight"].shape, (2, 3))

    def test_rejects_invalid_learning_rate(self):
        module = self.module()
        for lr in (0, -0.1, float("inf"), float("nan")):
            with self.subTest(lr=lr), self.assertRaises(ValueError):
                module.run_step(lr=lr)


if __name__ == "__main__":
    unittest.main()
