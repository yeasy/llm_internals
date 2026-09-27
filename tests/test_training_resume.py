"""Resume must reproduce the actual optimizer trajectory, not only weights."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch


class TrainingResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def module(self):
        path = Path(__file__).resolve().parents[1] / "examples/training_resume.py"
        self.assertTrue(path.exists(), "Resumable training exercise is missing")
        spec = importlib.util.spec_from_file_location("examples.training_resume", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def assert_state_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_resume_matches_all_training_state_across_shuffle_boundary(self):
        m = self.module()
        torch.manual_seed(7)
        continuous = m.Trainer()
        history = continuous.train(11)
        expected = copy.deepcopy(continuous.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            for split in (1, 3, 5):
                with self.subTest(split=split):
                    torch.manual_seed(7)
                    interrupted = m.Trainer()
                    prefix = interrupted.train(split)
                    path = Path(directory) / f"step-{split}.pt"
                    interrupted.save(path)
                    torch.rand(19)  # Loading must restore RNG after model construction.
                    restored = m.Trainer.load(path)
                    suffix = restored.train(11 - split)
                    self.assertEqual(prefix + suffix, history)
                    self.assert_state_equal(restored.state_dict(), expected)

    def test_validation_is_held_out_and_does_not_change_training_state(self):
        m = self.module()
        self.assertFalse(set(m.TRAIN_TEXTS) & set(m.VALID_TEXTS))
        trainer = m.Trainer()
        trainer.train(2)
        before = copy.deepcopy(trainer.state_dict())
        gradients = [p.grad.clone() for p in trainer.model.parameters()]
        grad_modes = []
        hook = trainer.model.register_forward_pre_hook(
            lambda *_: grad_modes.append(torch.is_grad_enabled()))
        self.assertGreater(trainer.validate(), 0)
        hook.remove()
        self.assertEqual(grad_modes, [False])
        self.assertTrue(trainer.model.training)
        self.assert_state_equal(before, trainer.state_dict())
        self.assert_state_equal(gradients, [p.grad for p in trainer.model.parameters()])
        trainer.model.eval()
        trainer.validate()
        self.assertFalse(trainer.model.training)

    def test_real_optimizer_scheduler_clipping_and_weight_updates(self):
        m = self.module()
        torch.manual_seed(7)
        trainer = m.Trainer()
        self.assertEqual(len(trainer.model.blocks), 2)
        self.assertEqual(trainer.model.blocks[0].attention.heads, 2)
        before = trainer.model.token_embedding.weight.detach().clone()
        history = trainer.train(2)
        self.assertFalse(torch.equal(before, trainer.model.token_embedding.weight))
        self.assertTrue(trainer.optimizer.state)
        self.assertEqual(trainer.step, 2)
        self.assertLess(history[1]["lr"], history[0]["lr"])
        norm = torch.stack([p.grad.norm() for p in trainer.model.parameters()]).norm()
        self.assertLessEqual(norm.item(), 1.00001)
        self.assertGreater(history[0]["grad_norm_before_clip"], 1.0)

    def test_save_refuses_overwrite_and_load_rejects_wrong_contract(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            trainer = m.Trainer()
            trainer.save(path)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                trainer.save(path)
            self.assertEqual(original, path.read_bytes())
            bad = trainer.state_dict()
            bad["contract"]["train_texts"] = ["changed"]
            invalid = Path(directory) / "invalid.pt"
            torch.save(bad, invalid)
            with self.assertRaisesRegex(ValueError, "contract"):
                m.Trainer.load(invalid)

    def test_step_count_must_be_positive(self):
        trainer = self.module().Trainer()
        for steps in (0, -1):
            with self.assertRaises(ValueError):
                trainer.train(steps)


if __name__ == "__main__":
    unittest.main()
