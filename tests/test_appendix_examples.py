from __future__ import annotations

import math
import re
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
APPENDIX = ROOT / "appendix" / "a2_pytorch_examples.md"


def python_block_after(heading: str) -> str:
    text = APPENDIX.read_text(encoding="utf-8")
    match = re.search(
        rf"^### {re.escape(heading)}\s*$.*?^```python\s*$\n(.*?)^```\s*$",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"missing Python block after {heading!r}")
    return match.group(1)


def load_attention_example():
    namespace: dict[str, object] = {}
    exec(python_block_after("缩放点积注意力"), namespace)
    return namespace["scaled_dot_product_attention"]


def manual_attention(query, key, value, visible=None, additive_bias=None):
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    if additive_bias is not None:
        scores = scores + additive_bias
    if visible is not None:
        scores = scores.masked_fill(~visible, float("-inf"))
    fully_masked = torch.isneginf(scores).all(dim=-1, keepdim=True)
    safe_scores = scores.masked_fill(fully_masked, 0.0)
    weights = torch.softmax(safe_scores, dim=-1).masked_fill(fully_masked, 0.0)
    return weights @ value, weights


class AppendixExamplesTest(unittest.TestCase):
    def test_attention_matches_manual_and_torch_sdpa_oracles(self):
        attention = load_attention_example()
        query = torch.tensor([[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]])
        key = torch.tensor(
            [[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 0.0]]]
        )
        value = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [3.0, 1.0]]])

        output, weights = attention(query, key, value)
        manual_output, manual_weights = manual_attention(query, key, value)
        sdpa_output = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)

        self.assertTrue(torch.allclose(weights, manual_weights, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(output, manual_output, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(output, sdpa_output, atol=1e-6, rtol=1e-6))

    def test_attention_combines_causal_padding_and_boolean_masks(self):
        attention = load_attention_example()
        torch.manual_seed(7)
        query = torch.randn(2, 2, 3, 4)
        key = torch.randn(2, 2, 3, 4)
        value = torch.randn(2, 2, 3, 5)
        key_padding = torch.tensor([[True, True, False], [True, False, True]])
        attn_mask = torch.ones(2, 3, 3, dtype=torch.bool)
        attn_mask[0, 1, :] = False
        attn_mask[1, 2, 0] = False

        output, weights = attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding,
            is_causal=True,
        )
        causal = torch.ones(3, 3, dtype=torch.bool).tril()
        visible = (
            causal[None, None, :, :]
            & attn_mask[:, None, :, :]
            & key_padding[:, None, None, :]
        )
        manual_output, manual_weights = manual_attention(
            query, key, value, visible=visible
        )
        sdpa_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=visible, dropout_p=0.0
        )

        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(
            torch.equal(weights[0, :, 1], torch.zeros_like(weights[0, :, 1]))
        )
        self.assertTrue(torch.allclose(weights, manual_weights, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(output, manual_output, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(output, sdpa_output, atol=1e-6, rtol=1e-6))

    def test_attention_combines_additive_and_padding_masks(self):
        attention = load_attention_example()
        query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        key = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
        value = torch.tensor([[[1.0], [2.0], [4.0]]])
        additive = torch.tensor([[0.0, -0.25, -0.5], [-0.75, 0.0, -0.5]])
        key_padding = torch.tensor([[True, True, False]])

        output, weights = attention(
            query,
            key,
            value,
            attn_mask=additive,
            key_padding_mask=key_padding,
        )
        visible = key_padding[:, None, :]
        manual_output, manual_weights = manual_attention(
            query, key, value, visible=visible, additive_bias=additive
        )
        combined = additive[None, :, :].masked_fill(~visible, float("-inf"))
        sdpa_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=combined, dropout_p=0.0
        )

        self.assertTrue(torch.allclose(weights, manual_weights, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(output, manual_output, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(output, sdpa_output, atol=1e-6, rtol=1e-6))

    def test_rope_dot_product_depends_only_on_relative_position(self):
        namespace = {"torch": torch}
        exec(python_block_after("旋转位置编码（RoPE）"), namespace)
        frequencies = namespace["rope_frequencies"]
        apply_rope = namespace["apply_rope"]
        cos, sin = frequencies(8, max_len=64)
        query = torch.tensor([[0.2, -0.4, 0.7, 0.1, -0.3, 0.8, 0.5, -0.6]])
        key = torch.tensor([[-0.1, 0.6, 0.2, -0.7, 0.9, 0.3, -0.5, 0.4]])

        def rotate(vector, position):
            return apply_rope(
                vector,
                cos[position : position + 1],
                sin[position : position + 1],
            )

        first = rotate(query, 7) @ rotate(key, 2).T
        shifted = rotate(query, 37) @ rotate(key, 32).T
        different = rotate(query, 37) @ rotate(key, 31).T
        self.assertTrue(torch.allclose(first, shifted, atol=1e-5, rtol=1e-5))
        self.assertFalse(torch.allclose(first, different, atol=1e-5, rtol=1e-5))

    def test_rmsnorm_matches_direct_formula(self):
        namespace = {"torch": torch, "nn": torch.nn}
        exec(python_block_after("RMSNorm"), namespace)
        rms_norm = namespace["RMSNorm"](4, eps=1e-6)
        with torch.no_grad():
            rms_norm.weight.copy_(torch.tensor([1.0, 0.5, 2.0, -1.0]))
        values = torch.tensor([[1.0, -2.0, 3.0, -4.0], [0.5, 0.0, -0.5, 1.0]])
        expected = rms_norm.weight * (
            values
            / torch.sqrt(values.square().mean(dim=-1, keepdim=True) + rms_norm.eps)
        )
        actual = rms_norm(values)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
