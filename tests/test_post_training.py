"""Numerical contracts for the download-free post-training core experiments."""
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import unittest

import torch
from torch import nn
from torch.nn import functional as F


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/post_training.py"


class PostTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def module(self):
        self.assertTrue(EXAMPLE.exists(), "Post-training core example is missing")
        spec = importlib.util.spec_from_file_location("post_training", EXAMPLE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_sft_shifts_target_mask_and_excludes_padding(self):
        m = self.module()
        tokens = torch.tensor([[1, 2, 3, 4, 0], [1, 2, 4, 3, 4]])
        assistant = torch.tensor([[0, 0, 1, 1, 1], [0, 0, 0, 1, 1]], dtype=torch.bool)
        valid = tokens.ne(0)
        original = tokens.clone()
        x, labels = m.sft_batch(tokens, assistant, valid)
        self.assertTrue(torch.equal(x, tokens[:, :-1]))
        self.assertEqual(labels.tolist(), [[-100, 3, 4, -100], [-100, -100, 3, 4]])
        self.assertTrue(torch.equal(tokens, original))

    def test_sft_token_mean_and_zero_gradient_at_ignored_positions(self):
        m = self.module()
        logits = torch.tensor([[[2., 0.], [0., 1.]], [[1., 0.], [0., 2.]]], requires_grad=True)
        labels = torch.tensor([[0, -100], [1, 1]])
        expected = (-F.log_softmax(logits, -1)[0, 0, 0]
                    - F.log_softmax(logits, -1)[1, 0, 1]
                    - F.log_softmax(logits, -1)[1, 1, 1]) / 3
        loss = m.sft_loss(logits, labels)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(logits.grad[0, 1].abs().sum().item(), 0.)
        self.assertGreater(logits.grad[1].abs().sum().item(), 0.)

    def test_sft_rejects_all_ignored_and_invalid_shapes(self):
        m = self.module()
        with self.assertRaisesRegex(ValueError, "supervised"):
            m.sft_loss(torch.zeros(1, 2, 3), torch.full((1, 2), -100))
        with self.assertRaises(ValueError):
            m.sft_batch(torch.ones(1, 1, dtype=torch.long), torch.ones(1, 1, dtype=torch.bool))
        with self.assertRaises(ValueError):
            m.sft_batch(torch.ones(1, 3, dtype=torch.long), torch.ones(1, 2, dtype=torch.bool))

    def test_sequence_log_probs_sum_response_tokens_not_mean(self):
        m = self.module()
        logits = torch.zeros(2, 3, 4)
        labels = torch.tensor([[-100, 1, 2], [-100, -100, 3]])
        torch.testing.assert_close(m.sequence_log_probs(logits, labels),
                                   torch.tensor([-2 * math.log(4), -math.log(4)]))
        with self.assertRaisesRegex(ValueError, "supervised"):
            m.sequence_log_probs(logits, torch.full((2, 3), -100))

    def test_lora_initial_equivalence_shapes_and_scaling(self):
        m = self.module()
        torch.manual_seed(17)
        base = nn.Linear(4, 3).double()
        layer = m.LoRALinear(base, rank=2, alpha=6.)
        x = torch.arange(8, dtype=torch.float64).reshape(2, 4) / 8
        self.assertEqual(layer.A.shape, (2, 4))
        self.assertEqual(layer.B.shape, (3, 2))
        self.assertGreater(layer.A.abs().sum().item(), 0.)
        self.assertEqual(layer.B.abs().sum().item(), 0.)
        torch.testing.assert_close(layer(x), base(x), rtol=0, atol=0)
        with torch.no_grad():
            layer.A.fill_(0.5)
            layer.B.fill_(0.25)
        torch.testing.assert_close(layer(x), base(x) + 3 * F.linear(F.linear(x, layer.A), layer.B))

    def test_lora_first_step_changes_b_not_base_or_initial_a(self):
        m = self.module()
        torch.manual_seed(7)
        layer = m.LoRALinear(nn.Linear(4, 3), rank=2, alpha=2.)
        before = {name: p.detach().clone() for name, p in layer.named_parameters()}
        optimizer = torch.optim.SGD([p for p in layer.parameters() if p.requires_grad], lr=0.1)
        loss = layer(torch.ones(2, 4)).square().mean()
        loss.backward()
        self.assertEqual(layer.A.grad.abs().sum().item(), 0.)
        self.assertGreater(layer.B.grad.abs().sum().item(), 0.)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in layer.base.parameters()))
        optimizer.step()
        changed = [name for name, p in layer.named_parameters() if not torch.equal(before[name], p)]
        self.assertEqual(changed, ["B"])

    def test_lora_rejects_invalid_rank(self):
        m = self.module()
        for rank in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                m.LoRALinear(nn.Linear(4, 3), rank=rank, alpha=2.)

    def test_dpo_equal_policy_reference_is_log_two_and_reference_is_frozen(self):
        m = self.module()
        chosen = torch.tensor([-1., -2.], requires_grad=True)
        rejected = torch.tensor([-3., -4.], requires_grad=True)
        ref_chosen = chosen.detach().clone().requires_grad_()
        ref_rejected = rejected.detach().clone().requires_grad_()
        loss = m.dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.2)
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        loss.backward()
        self.assertTrue(torch.all(chosen.grad < 0))
        self.assertTrue(torch.all(rejected.grad > 0))
        self.assertIsNone(ref_chosen.grad)
        self.assertIsNone(ref_rejected.grad)

    def test_dpo_one_step_increases_preferred_probability(self):
        m = self.module()
        scores = nn.Parameter(torch.zeros(2))
        optimizer = torch.optim.SGD([scores], lr=0.5)
        ref = scores.detach().log_softmax(-1)
        logp = scores.log_softmax(-1)
        before = logp[0].exp().item()
        loss = m.dpo_loss(logp[0:1], logp[1:2], ref[0:1], ref[1:2], beta=1.)
        loss.backward()
        optimizer.step()
        self.assertGreater(scores.softmax(-1)[0].item(), before)
        after = scores.log_softmax(-1)
        self.assertLess(m.dpo_loss(after[0:1], after[1:2], ref[0:1], ref[1:2], beta=1.).item(), loss.item())

    def test_dpo_is_stable_for_large_negative_margin(self):
        m = self.module()
        chosen = torch.tensor([-10000.], requires_grad=True)
        loss = m.dpo_loss(chosen, torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.]), beta=1.)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(chosen.grad).all())
        self.assertAlmostEqual(loss.item(), 10000.)

    def test_group_advantages_are_per_prompt_and_zero_for_tied_rewards(self):
        m = self.module()
        rewards = torch.tensor([[0., 1., 2.], [100., 101., 102.], [5., 5., 5.]])
        advantages = m.group_relative_advantages(rewards)
        torch.testing.assert_close(advantages.mean(-1), torch.zeros(3))
        torch.testing.assert_close(advantages[0], advantages[1])
        torch.testing.assert_close(advantages[0], torch.tensor([-1., 0., 1.]) / 1.0001)
        self.assertTrue(torch.equal(advantages[2], torch.zeros(3)))
        with self.assertRaises(ValueError):
            m.group_relative_advantages(torch.ones(2, 1))

    def test_clipped_surrogate_handles_both_advantage_signs_and_freezes_old_policy(self):
        m = self.module()
        new = torch.log(torch.tensor([1.5, 0.5, 0.5, 1.5])).requires_grad_()
        old = torch.zeros(4, requires_grad=True)
        advantages = torch.tensor([1., -1., 1., -1.], requires_grad=True)
        loss = m.clipped_policy_loss(new, old, advantages, clip_epsilon=0.2)
        # min(1.5, 1.2), min(-0.5, -0.8), min(0.5, 0.8), min(-1.5, -1.2)
        self.assertAlmostEqual(loss.item(), -(1.2 - 0.8 + 0.5 - 1.5) / 4, places=6)
        loss.backward()
        self.assertEqual(new.grad[0].item(), 0.)
        self.assertEqual(new.grad[1].item(), 0.)
        self.assertLess(new.grad[2].item(), 0.)
        self.assertGreater(new.grad[3].item(), 0.)
        self.assertIsNone(old.grad)
        self.assertIsNone(advantages.grad)

    def test_cli_reports_measured_results_and_scope(self):
        self.module()
        result = subprocess.run([sys.executable, str(EXAMPLE)], check=True, capture_output=True, text=True)
        for label in ("SFT", "LoRA", "DPO", "GRPO", "not a complete", "supervised_tokens=", "base_max_change=0.000000"):
            self.assertIn(label, result.stdout)


if __name__ == "__main__":
    unittest.main()
