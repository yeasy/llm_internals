import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "check_project_rules.py"


def load_checker():
    spec = importlib.util.spec_from_file_location("check_project_rules", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CheckProjectRulesTest(unittest.TestCase):
    def test_heading_checks_catch_skipped_levels_and_wrong_first_heading(self):
        checker = load_checker()
        path = ROOT / "01_intro" / "1.1_bad.md"
        text = "# Wrong first heading\n\n### Skipped child\n"

        issues = checker.check_headings(path, text)

        self.assertIn(
            "01_intro/1.1_bad.md:1: section files must start with level-2 heading",
            issues,
        )
        self.assertIn(
            "01_intro/1.1_bad.md:3: heading level jumps from H1 to H3",
            issues,
        )

    def test_local_links_validate_anchors_and_reject_escaped_paths(self):
        checker = load_checker()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            temp_path = Path(temp_dir)
            target = temp_path / "target.md"
            target.write_text("## Good Heading\n", encoding="utf-8")
            source = temp_path / "source.md"
            text = (
                "[good](target.md#good-heading)\n"
                "[bad anchor](target.md#missing-heading)\n"
                "[escaped](../../outside.md)\n"
            )

            issues = checker.check_links(source, text)

            self.assertIn(
                f"{source.relative_to(ROOT)}:2: missing local link anchor: "
                "target.md#missing-heading",
                issues,
            )
            self.assertIn(
                f"{source.relative_to(ROOT)}:3: local link escapes book root: ../../outside.md",
                issues,
            )
            self.assertTrue(all("good-heading" not in issue for issue in issues))

    def test_summary_coverage_flags_unlisted_markdown(self):
        checker = load_checker()
        files = [
            ROOT / "README.md",
            ROOT / "SUMMARY.md",
            ROOT / "chapter.md",
            ROOT / "unlisted.md",
        ]
        summary_text = "* [Home](README.md)\n* [Chapter](chapter.md)\n"

        issues = checker.check_summary_coverage(summary_text, files)

        self.assertIn("SUMMARY.md: missing Markdown file from summary: unlisted.md", issues)


if __name__ == "__main__":
    unittest.main()
