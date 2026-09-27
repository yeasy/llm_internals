"""Executable contracts for the complete CPU teaching example."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch


class TinyGPTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def module(self):
        path = Path(__file__).resolve().parents[1] / "examples/tiny_gpt.py"
        self.assertTrue(path.exists(), "Complete training example is missing")
        spec = importlib.util.spec_from_file_location("tiny_gpt", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_shift_and_causality(self):
        m = self.module()
        x, y = m.training_batch()
        self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))
        model = m.TinyGPT().eval()
        self.assertEqual(len(model.blocks), 2)
        self.assertEqual(model.blocks[0].attention.heads, 2)
        changed = x.clone()
        changed[:, -1] = (changed[:, -1] + 1) % len(m.VOCAB)
        with torch.no_grad():
            logits = model(x)
            self.assertEqual(logits.shape, (*x.shape, len(m.VOCAB)))
            torch.testing.assert_close(logits[:, :-1], model(changed)[:, :-1])

    def test_training_checkpoint_and_generation(self):
        m = self.module()
        torch.manual_seed(7)
        model = m.TinyGPT()
        before = model.token_embedding.weight.detach().clone()
        losses = m.train(model, steps=200)
        self.assertLess(losses[-1], losses[0] * 0.4)
        self.assertFalse(torch.equal(before, model.token_embedding.weight))
        for text in m.CORPUS:
            self.assertEqual(m.generate(model, text[:4]), text)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            m.save_model(model, path)
            restored = m.load_model(path)
            x, _ = m.training_batch()
            with torch.no_grad():
                torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)
            self.assertEqual(m.generate(restored, "2+3="), "2+3=5。")
            self.assertEqual(m.generate(restored, "2+3=", max_new_tokens=1), "2+3=5")

    def test_save_refuses_existing_file(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            path.write_bytes(b"existing checkpoint")
            with self.assertRaises(FileExistsError):
                m.save_model(m.TinyGPT(), path)
            self.assertEqual(path.read_bytes(), b"existing checkpoint")

    def test_generation_input_limits(self):
        m = self.module()
        model = m.TinyGPT()
        for prompt in ("", "unknown", "2" * (m.CONTEXT + 1)):
            with self.assertRaises(ValueError):
                m.generate(model, prompt)
        self.assertEqual(m.generate(model, "2", max_new_tokens=0), "2")
        self.assertEqual(m.generate(model, "2" * m.CONTEXT), "2" * m.CONTEXT)
        with self.assertRaises(ValueError):
            m.generate(model, "2", max_new_tokens=-1)
        with self.assertRaises(ValueError):
            m.train(model, steps=0)


if __name__ == "__main__":
    unittest.main()
