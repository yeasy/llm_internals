"""3.8 的总图与逐步展开顺序，避免重排后重新混淆层次。"""

from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "03_components/3.8_gpt_inference_flow.md"
TOPICS = ["整体结构", "构造输入", "多头注意力", "完成一层", "从最终表示",
          "Decode", "从教学模型", "计算开销"]


def has_expected_order(text):
    headings = re.findall(r"^### 3\.8\.(\d+) (.+)$", text, re.M)
    return len(headings) == 8 and all(
        number == str(i) and topic in title
        for i, ((number, title), topic) in enumerate(zip(headings, TOPICS), 1)
    )


class InferenceFlowStructureTests(unittest.TestCase):
    def test_general_shapes_and_norm_residual_contract_are_explained(self):
        text = SOURCE.read_text()
        for term in ("d_ff", "[n, d_h] × [d_h, m]", "[n, m] × [m, d_h]",
                     "[n, n_h × d_h]", "RMSNorm", "2L + 1",
                     "归一化前", "不是运行时"):
            self.assertIn(term, text)

    def test_matrix_values_are_not_equated_to_vector_spaces(self):
        bad = re.search(r"\\in\s*\\mathbb\{R\}\^\{[^}]+\}\s*=", SOURCE.read_text())
        self.assertIsNone(bad, "应以等号连接数值矩阵，维度另行标注")

    def test_overview_precedes_local_diagrams(self):
        images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", SOURCE.read_text())
        self.assertEqual(images[0], "_images/gpt_inference_overview.png")
        for image in images:
            self.assertTrue((SOURCE.parent / image).is_file(), image)

    def test_sections_follow_computation(self):
        self.assertTrue(has_expected_order(SOURCE.read_text()))

    def test_order_check_rejects_swapped_topics(self):
        correct = "\n".join(f"### 3.8.{i} {topic}" for i, topic in enumerate(TOPICS, 1))
        wrong = correct.replace("3.8.3 多头注意力", "3.8.3 从最终表示")
        self.assertTrue(has_expected_order(correct))
        self.assertFalse(has_expected_order(wrong))


if __name__ == "__main__":
    unittest.main()
