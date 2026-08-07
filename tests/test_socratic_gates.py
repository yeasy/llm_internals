"""三条「教学接缝」门禁的测试。

每条都先证明它在该失败时确实失败，再证明它对合规写法放行——一条不可能
失败的检查不构成任何证据。第四个测试守住 PREREQ_TERMS 常量本身，防止术语
表指向一个已经不存在或已经不含该术语的文件。
"""

import importlib.util
import re
import sys
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


class SectionIntroTest(unittest.TestCase):
    def test_flags_section_that_falls_straight_into_its_first_subsection(self):
        checker = load_checker()
        path = ROOT / "07_distributed_training" / "7.4_pipeline_hybrid.md"
        text = "## 7.4 流水线并行\n\n### 7.4.1 流水线并行\n\n**流水线并行**将模型的不同层分配到不同的 GPU 上。\n"

        issues = checker.check_section_intro(path, text)

        self.assertEqual(len(issues), 1, issues)
        self.assertIn("缺少引入段", issues[0])

    def test_flags_intro_too_short_to_carry_a_positioning_and_a_handoff(self):
        checker = load_checker()
        path = ROOT / "07_distributed_training" / "7.4_pipeline_hybrid.md"
        text = "## 7.4 流水线并行\n\n本节介绍流水线并行。\n\n### 7.4.1 基本原理\n\n正文。\n"

        self.assertEqual(len(checker.check_section_intro(path, text)), 1)

    def test_accepts_a_real_lead_in(self):
        checker = load_checker()
        path = ROOT / "07_distributed_training" / "7.4_pipeline_hybrid.md"
        text = (
            "## 7.4 流水线并行\n\n"
            "**流水线并行**把模型的不同层放到不同 GPU 上。前一节已经算过它的前提："
            "跨节点带宽只有 NVLink 的约 1/9，张量并行因此走不出节点。\n\n"
            "### 7.4.1 基本原理\n\n正文。\n"
        )

        self.assertEqual(checker.check_section_intro(path, text), [])

    def test_ignores_chapter_readmes_and_summaries(self):
        checker = load_checker()
        text = "## 概览\n\n### 子节\n\n正文。\n"

        self.assertEqual(checker.check_section_intro(ROOT / "07_x" / "README.md", text), [])
        self.assertEqual(checker.check_section_intro(ROOT / "07_x" / "summary.md", text), [])


class ClosingClaimTest(unittest.TestCase):
    def test_flags_an_unhedged_closing_claim(self):
        checker = load_checker()
        path = ROOT / "07_distributed_training" / "7.3_model_tensor_parallel.md"
        text = "## 7.3 张量并行\n\n### 7.3.1 原理\n\n正文。\n\n这种组合策略是当前主流大模型训练框架的标配。\n"

        issues = checker.check_closing_claim(path, text)

        self.assertEqual(len(issues), 1, issues)
        self.assertIn("标配", issues[0])

    def test_accepts_the_same_claim_once_a_boundary_follows(self):
        checker = load_checker()
        path = ROOT / "07_distributed_training" / "7.3_model_tensor_parallel.md"
        text = (
            "## 7.3 张量并行\n\n### 7.3.1 原理\n\n正文。\n\n"
            "这种组合策略是当前主流大模型训练框架的标配。但它有一条硬边界："
            "层数这一维到这里还没有任何人管。\n"
        )

        self.assertEqual(checker.check_closing_claim(path, text), [])

    def test_only_looks_at_the_last_paragraph(self):
        checker = load_checker()
        path = ROOT / "07_distributed_training" / "7.3_model_tensor_parallel.md"
        text = (
            "## 7.3 张量并行\n\n### 7.3.1 原理\n\n"
            "RMSNorm 已成为主流选择。\n\n"
            "后面还有一段普通正文，本节到此为止。\n"
        )

        self.assertEqual(checker.check_closing_claim(path, text), [])


class PrereqTermsTest(unittest.TestCase):
    def test_the_term_table_points_at_files_that_actually_define_the_terms(self):
        """台账自身的防漂移检查：定义文件必须存在且真的含该术语。"""
        checker = load_checker()
        for term, definition in checker.PREREQ_TERMS.items():
            target = ROOT / definition
            with self.subTest(term=term):
                self.assertTrue(target.exists(), f"{definition} 不存在")
                self.assertRegex(
                    target.read_text(encoding="utf-8"),
                    f"(?i){term}",
                    f"{definition} 里已经没有「{term}」了，术语表该更新",
                )

    def test_every_term_is_defined_after_at_least_one_of_its_uses(self):
        """否则这条规则形同虚设——术语若从不被提前使用，检查永远不会触发。"""
        checker = load_checker()
        order = checker.summary_ordered_targets()
        position = {target: index for index, target in enumerate(order)}
        for term, definition in checker.PREREQ_TERMS.items():
            defined_at = position.get(definition)
            with self.subTest(term=term):
                self.assertIsNotNone(defined_at, f"{definition} 不在 SUMMARY 里")
                earlier_uses = [
                    path
                    for path in checker.iter_markdown_files()
                    if checker.is_numbered_section(path)
                    and position.get(checker.rel(path), 10**6) < defined_at
                    and re.search(
                        term, path.read_text(encoding="utf-8", errors="ignore"), re.I
                    )
                ]
                self.assertTrue(
                    earlier_uses,
                    f"「{term}」从未在 {definition} 之前出现，这条规则是空转的",
                )

    def test_repository_passes_all_three_gates(self):
        checker = load_checker()
        files = checker.iter_markdown_files()
        issues: list[str] = []
        for path in files:
            text = path.read_text(encoding="utf-8", errors="ignore")
            issues.extend(checker.check_section_intro(path, text))
            issues.extend(checker.check_closing_claim(path, text))
        issues.extend(checker.check_prereq_terms(files))

        self.assertEqual(issues, [])


if __name__ == "__main__":
    unittest.main()
