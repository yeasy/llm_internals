"""技术地图的精确导航和共享模型计算层回归。"""

import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "01_introduction/1.0_technology_map.md"


class TechnologyMapTests(unittest.TestCase):
    def test_quick_links_land_on_specific_topics(self):
        text = MAP.read_text(encoding="utf-8")
        targets = {
            "11_serving/11.12_hardware.md": ("gpu-hardware", "cluster", "data-center"),
            "11_serving/11.14_gpu_software.md": ("gpu-kernels", "kernel-fusion", "nccl"),
            "14_future_trends/14.5_agent_tool_use.md": ("agent-tool-call", "rag-workflow-agent"),
            "appendix/a7_framework_recipes.md": ("distributed-frameworks",),
            "11_serving/11.1_engines_overview.md": ("tensorrt-llm",),
        }
        for path, fragments in targets.items():
            destination = (ROOT / path).read_text(encoding="utf-8")
            for fragment in fragments:
                with self.subTest(path=path, fragment=fragment):
                    self.assertTrue(f"../{path}#{fragment}" in text, f"地图缺少 {fragment} 精确链接")
                    self.assertTrue(re.search(rf'<a\s+id="{re.escape(fragment)}"\s*></a>', destination),
                                    f"{path} 缺少 {fragment} 锚点")

    def test_model_computation_is_a_box_between_execution_and_kernels(self):
        script = ROOT / "tools/figures/llm_technology_map.py"
        tree = ast.parse(script.read_text(encoding="utf-8"))
        boxes = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "box":
                if len(node.args) >= 5 and isinstance(node.args[4], ast.Constant):
                    boxes[node.args[4].value] = ast.literal_eval(node.args[1])
        model = [y for title, y in boxes.items() if title.startswith("模型结构与计算")]
        self.assertEqual(len(model), 1, "共享模型计算必须是独立框，不能仅标成小字")
        training = next(y for title, y in boxes.items() if title.startswith("训练与后训练"))
        kernels = next(y for title, y in boxes.items() if title.startswith("算子、运行时与通信"))
        self.assertLess(training, model[0])
        self.assertLess(model[0], kernels)


if __name__ == "__main__":
    unittest.main()
