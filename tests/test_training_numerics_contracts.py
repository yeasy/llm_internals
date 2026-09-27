"""Numerical counterexamples and prose guards for training precision claims."""
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


def last_adam_update(history, last_gradient):
    parameter = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    optimizer = torch.optim.Adam([parameter], lr=1.0, betas=(0.9, 0.95), eps=1e-8)
    for gradient in [history] * 999 + [last_gradient]:
        before = parameter.item()
        parameter.grad = torch.tensor(gradient, dtype=parameter.dtype)
        optimizer.step()
    return before - parameter.item()


class TrainingNumericsContracts(unittest.TestCase):
    def test_adam_prose_does_not_turn_scale_invariance_into_a_step_bound(self):
        text = (ROOT / "06_training_techniques/6.1_loss_optimizer.md").read_text()
        for claim in ("每步最多挪动这么多", "幅度上界约为", "一次大梯度改变的是方向而不是步长",
                      "裁剪并不是靠“把这一步缩小”起作用的"):
            self.assertFalse(claim in text, claim)
        self.assertIn("全部历史梯度", text)

    def test_adam_update_can_exceed_learning_rate(self):
        update = last_adam_update(1.0, 2.0)
        self.assertGreater(update, 1.0)
        self.assertAlmostEqual(update, 1.0257552795, places=9)

    def test_clipping_current_gradient_changes_current_adam_step(self):
        original = last_adam_update(0.02, 0.4)
        clipped = last_adam_update(0.02, 0.04)
        self.assertAlmostEqual(original, 0.6335865317, places=9)
        self.assertAlmostEqual(clipped, 1.0257548108, places=9)
        self.assertGreater(clipped, original)

    def test_clipping_prose_qualifies_moment_and_jacobian_arguments(self):
        text = (ROOT / "06_training_techniques/6.3_regularization.md").read_text()
        for claim in ("裁剪保护的不是这一步", "Adam 的更新一位数字都不变",
                      "近乎停止学习", "谱范数）普遍大于 1，多层连乘后梯度范数就会指数级放大"):
            self.assertFalse(claim in text, claim)
        self.assertIn("上界", text)
        self.assertIn("分子", text)

    def test_native_amp_retains_fp32_parameter_gradient_and_adam_states(self):
        parameter = torch.nn.Parameter(torch.ones(2, 2))
        optimizer = torch.optim.AdamW([parameter])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = torch.ones(2, 2) @ parameter
            loss = output.square().mean()
        loss.backward()
        optimizer.step()
        self.assertEqual(output.dtype, torch.bfloat16)
        for tensor in (parameter, parameter.grad, optimizer.state[parameter]["exp_avg"],
                       optimizer.state[parameter]["exp_avg_sq"]):
            self.assertEqual(tensor.dtype, torch.float32)

    def test_amp_prose_separates_compute_storage_and_communication(self):
        text = (ROOT / "07_distributed_training/7.6_mixed_precision.md").read_text()
        self.assertNotIn("消息都以 16 位发送", text)
        self.assertIn("原生 AMP", text)
        self.assertIn("unscale_", text)
        self.assertIn("裁剪", text)


if __name__ == "__main__":
    unittest.main()
