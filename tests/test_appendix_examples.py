import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class AppendixExamplesTest(unittest.TestCase):
    def _load_attention_example(self):
        text = (ROOT / "appendix" / "a2_pytorch_examples.md").read_text(encoding="utf-8")
        match = re.search(
            r"```python\n(import torch\nimport torch\.nn\.functional as F\nimport math\n"
            r"\n+def scaled_dot_product_attention\(.*?return output, attn_weights)\n```",
            text,
            re.DOTALL,
        )
        self.assertIsNotNone(match)
        namespace = {}
        try:
            exec(match.group(1), namespace)
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                self.skipTest("PyTorch is required to execute appendix attention example")
            raise
        return namespace["torch"], namespace["scaled_dot_product_attention"]

    def test_attention_example_runs_3d_4d_and_fully_masked_rows(self):
        torch, scaled_dot_product_attention = self._load_attention_example()

        Q = torch.randn(2, 3, 4)
        K = torch.randn(2, 4, 4)
        V = torch.randn(2, 4, 4)
        mask_3d = torch.ones(2, 3, 4, dtype=torch.bool)
        mask_3d[0, 1, :] = False

        output, weights = scaled_dot_product_attention(Q, K, V, attn_mask=mask_3d)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(torch.equal(weights[0, 1], torch.zeros_like(weights[0, 1])))
        self.assertTrue(torch.equal(output[0, 1], torch.zeros_like(output[0, 1])))

        Q4 = torch.randn(2, 2, 3, 4)
        K4 = torch.randn(2, 2, 4, 4)
        V4 = torch.randn(2, 2, 4, 4)
        mask_4d = torch.ones(2, 2, 3, 4, dtype=torch.bool)
        mask_4d[1, 0, 2, :] = False

        output4, weights4 = scaled_dot_product_attention(Q4, K4, V4, attn_mask=mask_4d)
        self.assertTrue(torch.isfinite(output4).all())
        self.assertTrue(torch.isfinite(weights4).all())
        self.assertTrue(torch.equal(weights4[1, 0, 2], torch.zeros_like(weights4[1, 0, 2])))
        self.assertTrue(torch.equal(output4[1, 0, 2], torch.zeros_like(output4[1, 0, 2])))


if __name__ == "__main__":
    unittest.main()
