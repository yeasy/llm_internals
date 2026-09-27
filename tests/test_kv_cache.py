"""Contracts for inference-only, per-layer KV caching of the teaching GPT."""
import importlib
from pathlib import Path
import unittest

import torch


class KVCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "examples/kv_cache.py"
        self.assertTrue(path.exists(), "KV cache teaching example is missing")
        self.m = importlib.import_module("examples.kv_cache")
        torch.manual_seed(7)
        self.model = self.m.TinyGPT().eval()
        self.tokens = torch.tensor([self.m.encode("2+3=5。"), self.m.encode("1+2=3。")])

    def check_chunks(self, sizes):
        cache = None
        offset = 0
        for size in sizes:
            logits, cache = self.m.cached_forward(
                self.model, self.tokens[:, offset:offset + size], cache)
            with torch.no_grad():
                expected = self.model(self.tokens[:, :offset + size])[:, offset:]
            torch.testing.assert_close(logits, expected, atol=1e-6, rtol=1e-5)
            offset += size
            self.assertFalse(logits.requires_grad)
            self.assertEqual(len(cache), 2)
            for keys, values in cache:
                self.assertEqual(keys.shape, (2, 2, offset, 16))
                self.assertEqual(values.shape, keys.shape)
                self.assertFalse(keys.requires_grad)
                self.assertFalse(values.requires_grad)
                self.assertIsNone(keys.grad_fn)
                self.assertIsNone(values.grad_fn)
        self.assertEqual(offset, self.tokens.shape[1])

    def test_prefill_and_single_token_decode_match_full_prefix(self):
        self.check_chunks([3, 1, 1, 1])

    def test_multi_token_decode_matches_full_prefix(self):
        self.check_chunks([1, 2, 3])

    def test_one_prefill_matches_full_forward(self):
        self.check_chunks([6])

    def test_decode_does_not_modify_old_cache_or_share_its_storage(self):
        _, old = self.m.cached_forward(self.model, self.tokens[:, :2])
        snapshot = [(k.clone(), v.clone()) for k, v in old]
        first, grown = self.m.cached_forward(self.model, self.tokens[:, 2:4], old)
        second, _ = self.m.cached_forward(self.model, self.tokens[:, 2:4], old)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        for (k, v), (saved_k, saved_v), (new_k, new_v) in zip(old, snapshot, grown):
            torch.testing.assert_close(k, saved_k, rtol=0, atol=0)
            torch.testing.assert_close(v, saved_v, rtol=0, atol=0)
            self.assertNotEqual(k.data_ptr(), new_k.data_ptr())
            self.assertNotEqual(v.data_ptr(), new_v.data_ptr())

    def test_invalid_input_and_context_overflow_are_rejected(self):
        invalid = [torch.empty(2, 0, dtype=torch.long),
                   torch.empty(0, 2, dtype=torch.long), self.tokens[0],
                   self.tokens.unsqueeze(0), self.tokens.float(),
                   torch.tensor([[-1]]), torch.tensor([[len(self.m.VOCAB)]]),
                   torch.zeros(2, self.m.CONTEXT + 1, dtype=torch.long)]
        for tokens in invalid:
            with self.subTest(shape=tuple(tokens.shape), dtype=tokens.dtype):
                with self.assertRaises(ValueError):
                    self.m.cached_forward(self.model, tokens)
        _, cache = self.m.cached_forward(
            self.model, torch.zeros(2, self.m.CONTEXT, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "context"):
            self.m.cached_forward(self.model, self.tokens[:, :1], cache)

    def test_invalid_cache_shape_layer_count_dtype_and_lengths_are_rejected(self):
        _, cache = self.m.cached_forward(self.model, self.tokens[:, :2])
        k, v = cache[0]
        invalid = [(), cache[:1], ((k[:, :1], v[:, :1]), cache[1]),
                   ((k[:1], v[:1]), cache[1]), ((k[..., :8], v[..., :8]), cache[1]),
                   ((k[:, :, :1], v), cache[1]),
                   ((k[:, :, :1], v[:, :, :1]), cache[1]),
                   ((k.double(), v.double()), cache[1]),
                   ((k.clone().requires_grad_(), v), cache[1]),
                   ((k[0], v[0]), cache[1])]
        for bad_cache in invalid:
            with self.subTest(cache_shapes=[tuple(pair[0].shape) for pair in bad_cache]):
                with self.assertRaises(ValueError):
                    self.m.cached_forward(self.model, self.tokens[:, 2:3], bad_cache)

    def test_training_mode_is_rejected_without_changing_model_mode(self):
        self.model.train()
        with self.assertRaisesRegex(ValueError, "eval"):
            self.m.cached_forward(self.model, self.tokens)
        self.assertTrue(self.model.training)


if __name__ == "__main__":
    unittest.main()
