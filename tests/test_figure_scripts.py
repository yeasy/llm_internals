"""插图生成脚本与正文引用之间的一致性门禁。

正文不再内联 matplotlib 绘图代码，改为图题链接到 `tools/figures/` 下的生成脚本。
这带来三处可能各自漂移的引用，本模块把它们钉在一起：

1. 正文图题里链到的每个脚本，磁盘上必须存在；
2. `tools/figures/` 下的每个脚本，必须被至少一处正文图题引用（避免留下孤儿脚本）；
3. 每个脚本 `OUTPUT` 声明的插图，必须已经存在于仓库中（正文靠它显示，不靠运行脚本）。
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "tools" / "figures"

# 图题里的脚本链接形如：
# https://github.com/yeasy/llm_internals/blob/main/tools/figures/rope_phase.py
LINK_RE = re.compile(
    r"https://github\.com/yeasy/llm_internals/blob/main/tools/figures/([A-Za-z0-9_]+\.py)"
)
OUTPUT_RE = re.compile(r'^OUTPUT = image_path\("([^"]+)",\s*"([^"]+)"\)', re.M)


def book_markdown() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*.md")
        if ".git" not in path.parts and "node_modules" not in path.parts
    ]


def figure_scripts() -> list[Path]:
    return sorted(
        path
        for path in FIGURE_DIR.glob("*.py")
        if not path.name.startswith("_")
    )


class FigureScriptContractTests(unittest.TestCase):
    def test_every_referenced_script_exists(self):
        referenced = 0
        for md in book_markdown():
            text = md.read_text(encoding="utf-8")
            for name in LINK_RE.findall(text):
                referenced += 1
                self.assertTrue(
                    (FIGURE_DIR / name).is_file(),
                    f"{md.relative_to(ROOT)} 引用了不存在的插图脚本 {name}",
                )
        self.assertGreater(referenced, 0, "正文里一个插图脚本链接都没有，门禁形同虚设")

    def test_every_script_is_referenced(self):
        referenced = {
            name
            for md in book_markdown()
            for name in LINK_RE.findall(md.read_text(encoding="utf-8"))
        }
        for script in figure_scripts():
            self.assertIn(
                script.name,
                referenced,
                f"{script.name} 没有被任何正文图题引用，是孤儿脚本",
            )

    def test_every_script_declares_an_existing_image(self):
        for script in figure_scripts():
            match = OUTPUT_RE.search(script.read_text(encoding="utf-8"))
            self.assertIsNotNone(
                match, f"{script.name} 未按约定声明 OUTPUT = image_path(...)"
            )
            chapter_dir, filename = match.groups()
            image = ROOT / chapter_dir / "_images" / filename
            self.assertTrue(
                image.is_file(),
                f"{script.name} 声明的插图 {chapter_dir}/_images/{filename} 不在仓库中",
            )

    def test_scripts_are_syntactically_valid(self):
        import ast

        for script in FIGURE_DIR.glob("*.py"):
            with self.subTest(script=script.name):
                ast.parse(script.read_text(encoding="utf-8"), filename=str(script))

    def test_prose_no_longer_inlines_plotting_code(self):
        """正文不应再出现 matplotlib 绘图代码块（本轮清理的回归护栏）。"""
        offenders = []
        for md in book_markdown():
            text = md.read_text(encoding="utf-8")
            for block in re.findall(r"^```(?:python|py)\n(.*?)^```", text, re.S | re.M):
                if "matplotlib" in block or re.search(r"\bplt\.", block):
                    offenders.append(str(md.relative_to(ROOT)))
        self.assertEqual(
            offenders,
            [],
            "正文重新出现了绘图代码块，请改放到 tools/figures/ 并由图题链接："
            + ", ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
