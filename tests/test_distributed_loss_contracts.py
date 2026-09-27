"""Regression examples for global token weighting and gradient noise scale."""
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


class DistributedLossContracts(unittest.TestCase):
    def test_ddp_document_explains_averaging_compensation(self):
        text = (ROOT / "07_distributed_training/7.1_data_parallel.md").read_text()
        self.assertTrue("K * local_loss_sum / global_token_count" in text,
                        "Document DDP's world-size compensation")

    def test_unequal_token_counts_require_world_size_factor(self):
        # Simulate DDP's mean of two local parameter gradients, not a GPU collective.
        values = (torch.tensor([1.0, 3.0], dtype=torch.float64),
                  torch.tensor([5.0], dtype=torch.float64))
        total = sum(x.numel() for x in values)
        gradients = []
        incorrect = []
        for x in values:
            w = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
            local_sum = (w * x).square().sum()
            gradients.append(torch.autograd.grad(2 * local_sum / total, w, retain_graph=True)[0])
            incorrect.append(torch.autograd.grad(local_sum / total, w)[0])
        w = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
        full_gradient, = torch.autograd.grad((w * torch.cat(values)).square().mean(), w)
        torch.testing.assert_close(torch.stack(gradients).mean(), full_gradient)
        torch.testing.assert_close(torch.stack(incorrect).mean(), full_gradient / 2)

    def test_noise_scale_prose_has_correct_direction(self):
        text = (ROOT / "06_training_techniques/6.4_batch_sequence.md").read_text()
        self.assertFalse("信噪比因此升高" in text, "Noise-to-signal and SNR have opposite directions")
        self.assertTrue("信噪比下降" in text, "Explain the decreasing signal-to-noise ratio")

    def test_cross_entropy_does_not_claim_to_fuse_output_projection(self):
        text = (ROOT / "06_training_techniques/6.1_loss_optimizer.md").read_text()
        self.assertTrue("不会自动融合" in text, "Ordinary cross_entropy receives existing logits")


if __name__ == "__main__":
    unittest.main()
