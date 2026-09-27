"""Focused guards for mathematical boundaries and framework prerequisites."""
import math
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


class CoreExplanationContracts(unittest.TestCase):
    def test_residual_text_does_not_promise_nonvanishing_gradients(self):
        text = (ROOT / "03_components/3.5_residual.md").read_text()
        for claim in ("梯度也不会整体塌到零", "恒等项使信号只增不减", "通路只加不减"):
            self.assertFalse(claim in text, f"Residual claim needs qualification: {claim}")
        self.assertTrue("上界" in text, "Distinguish a norm bound from actual scaling")

    def test_residual_branch_can_cancel_identity(self):
        x = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        y = x
        for _ in range(96):
            y = y - 0.9 * y
        (gradient,) = torch.autograd.grad(y, x)
        self.assertTrue(math.isclose(gradient.item(), 1e-96, rel_tol=1e-12))
        a = torch.diag(torch.tensor([1.1, 0.0], dtype=torch.float64))
        b = torch.diag(torch.tensor([0.0, 1.1], dtype=torch.float64))
        self.assertGreater(torch.linalg.matrix_norm(a, ord=2).item(), 1)
        self.assertGreater(torch.linalg.matrix_norm(b, ord=2).item(), 1)
        self.assertEqual(torch.linalg.matrix_norm(b @ a, ord=2).item(), 0)

    def test_multihead_projection_uses_unprojected_input(self):
        text = (ROOT / "02_attention/2.3_multi_head.md").read_text()
        self.assertTrue(r"\text{Attention}(XW_i^Q, XW_i^K, XW_i^V)" in text,
                        "Project the unprojected self-attention input X")
        self.assertFalse(r"\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)" in text,
                         "Do not silently reuse projected Q/K/V as module inputs")
        self.assertFalse("每个头拿到一个连续的" in text,
                         "A transpose view does not guarantee contiguous storage")

    def test_head_shapes_and_transpose_strides(self):
        x = torch.arange(3 * 8, dtype=torch.float64).reshape(3, 8)
        w = torch.eye(8, dtype=torch.float64)
        projected = x @ w
        heads = projected.view(3, 2, 4).transpose(0, 1)
        self.assertEqual(tuple(heads.shape), (2, 3, 4))
        self.assertFalse(heads[0].is_contiguous())
        for i in range(2):
            torch.testing.assert_close(heads[i], x @ w[:, i * 4:(i + 1) * 4])
        torch.testing.assert_close(heads.transpose(0, 1).reshape(3, 8), projected)

    def test_rope_text_keeps_relative_identity_at_low_frequencies(self):
        text = (ROOT / "04_position_encoding/4.3_rope.md").read_text()
        for claim in ("只对一部分维度成立", "才只携带相对信息"):
            self.assertFalse(claim in text, f"Relative identity holds at all frequencies: {claim}")

    def test_rope_shift_invariance_for_each_frequency(self):
        q = torch.tensor([0.3, -0.7], dtype=torch.float64)
        k = torch.tensor([0.8, 0.2], dtype=torch.float64)

        def rotate(vector, angle):
            c, s = math.cos(angle), math.sin(angle)
            return torch.tensor([[c, -s], [s, c]], dtype=torch.float64) @ vector

        for theta in (1.0, 1e-4):
            with self.subTest(theta=theta):
                first = rotate(q, 7 * theta) @ rotate(k, 2 * theta)
                shifted = rotate(q, 37 * theta) @ rotate(k, 32 * theta)
                relative = q @ rotate(k, (2 - 7) * theta)
                torch.testing.assert_close(first, shifted, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(first, relative, rtol=1e-12, atol=1e-12)

    def test_sft_masking_documents_config_and_template_requirements(self):
        text = (ROOT / "08_alignment/8.1_sft.md").read_text()
        for setting in ("assistant_only_loss=True", "completion_only_loss", "{% generation %}"):
            self.assertTrue(setting in text, f"Missing SFT masking prerequisite: {setting}")
        self.assertFalse("不是只训练最后一个回合，也不是把整段对话都算进去" in text,
                         "Assistant-only loss is a chosen objective, not a universal default")


if __name__ == "__main__":
    unittest.main()
