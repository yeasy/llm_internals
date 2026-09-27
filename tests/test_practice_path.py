"""Keep the runnable practice route reachable from the published book."""
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PracticePathTests(unittest.TestCase):
    def test_offline_benchmark_uses_local_random_ids(self):
        text = (ROOT / "appendix/a7_framework_recipes.md").read_text()
        self.assertIn("--dataset-name random-ids", text)
        self.assertNotIn("--dataset-name random --", text)

    def test_practice_appendices_are_published_and_linked(self):
        summary = (ROOT / "SUMMARY.md").read_text()
        readme = (ROOT / "README.md").read_text()
        for name in ("a6_practice_path.md", "a7_framework_recipes.md"):
            with self.subTest(name=name):
                self.assertTrue((ROOT / "appendix" / name).is_file(), name)
                self.assertIn("appendix/" + name, summary)
                self.assertIn("appendix/" + name, readme)

    def test_all_cpu_experiments_have_reader_entrypoints(self):
        path = ROOT / "appendix/a6_practice_path.md"
        self.assertTrue(path.is_file(), "Missing practice guide")
        text = path.read_text()
        for name in ("tiny_gpt", "training_step", "training_resume", "kv_cache", "post_training"):
            with self.subTest(name=name):
                self.assertTrue((ROOT / "examples" / (name + ".py")).is_file())
                self.assertIn("examples/" + name + ".py", text)
                self.assertIn("tests.test_" + name, text)


if __name__ == "__main__":
    unittest.main()
