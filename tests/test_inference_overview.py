"""总图分层展开与残差连线；最终文字布局另做渲染检查。"""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

FIGURES = Path(__file__).resolve().parents[1] / "tools/figures"


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "需安装插图依赖 matplotlib")
class OverviewConnectionsTests(unittest.TestCase):
    def setUp(self):
        with patch.object(sys, "path", [str(FIGURES), *sys.path]):
            spec = importlib.util.spec_from_file_location(
                "overview", FIGURES / "gpt_inference_overview.py")
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)

    def tearDown(self):
        self.module.plt.close("all")

    def test_structure_is_separate_from_generation_timeline(self):
        with patch.object(self.module, "finish"):
            self.module.main()
        texts = "\n".join(t.get_text() for t in self.module.plt.gca().texts)
        for required in ("A  整个模型", "B  展开一层", "C  展开一个头",
                         "头 1", "头 2", "头 H", "拼接", "最后一行",
                         "[T, T]", "[T, d_h]", "虚线"):
            self.assertIn(required, texts)
        self.assertNotIn("m = 8", texts, "三轮生成应放在独立的时间线图")

    def test_local_layer_names_two_different_bypass_inputs(self):
        with patch.object(sys, "path", [str(FIGURES), *sys.path]):
            spec = importlib.util.spec_from_file_location(
                "layer", FIGURES / "transformer_layer_blocks.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        with patch.object(module, "finish"):
            module.main()
        texts = "\n".join(t.get_text() for t in module.plt.gca().texts)
        for required in ("保留 X", "保留 U", "X + 注意力输出", "U + MLP 输出",
                         "[n, d_model]", "最终 Norm", "不另加残差"):
            self.assertIn(required, texts)

    def test_layer_input_reaches_norm_and_both_residuals(self):
        edges = []
        with patch.object(self.module, "arrow", side_effect=lambda ax, p0, p1, **kw:
                          edges.append((p0, p1))), patch.object(self.module, "finish"):
            self.module.main()
        for edge in (((14.5, 3.8), (14.5, 5.0)),
                     ((14.5, 15.2), (14.5, 16.4)),
                     ((10.6, 14.3), (11.4, 14.3)),
                     ((10.6, 23.0), (11.4, 23.0))):
            self.assertIn(edge, edges)


if __name__ == "__main__":
    unittest.main()
