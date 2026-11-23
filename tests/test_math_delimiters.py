"""Inline formulas must use $$...$$, the only delimiter GitBook renders.

GitBook draws a $$...$$ that shares its line with other text as inline math and
shows single-dollar $...$ as literal text. mdPress follows the same convention
(a $$...$$ is display math only when it stands on its own line), so $$ is the
one inline delimiter that renders in every published format.
"""
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
PAIR = re.compile(r"(?<![\\$])\$\$(?!\$)(.+?)(?<!\\)\$\$(?!\$)")
SINGLE = re.compile(r"(?<![\\$])\$([^$\n\t ][^$\n]*?[^$\n\t \\]|[^$\n\t \\])\$(?!\$)")
CODE = re.compile(r"(`+).*?\1")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def single_dollar_spans(text):
    """Return character spans of single-dollar inline formulas in prose."""
    offset, fence, in_display = 0, None, False
    for line in text.splitlines(keepends=True):
        marker = FENCE.match(line)
        if fence is not None:
            if (marker and marker[1][0] == fence[0]
                    and len(marker[1]) >= len(fence) and not marker[2].strip()):
                fence = None
        elif marker and not in_display:
            fence = marker[1]
        elif line.strip().lstrip("> ").strip() == "$$":
            in_display = not in_display
        elif not in_display:
            masked = PAIR.sub(lambda m: " " * len(m[0]), line)
            code_spans = [match.span() for match in CODE.finditer(masked)]
            for match in SINGLE.finditer(masked):
                if any(start < match.end() and end > match.start()
                       for start, end in code_spans):
                    continue
                yield offset + match.start(), offset + match.end()
        offset += len(line)


class MathDelimiterTests(unittest.TestCase):
    def test_detects_single_dollar_in_prose_and_tables(self):
        text = "维度 $d$，矩阵 $X_1$。\n| $Q$ | 输入 |\n"
        self.assertEqual([text[a:b] for a, b in single_dollar_spans(text)],
                         ["$d$", "$X_1$", "$Q$"])

    def test_preserves_double_dollar_display_code_and_prices(self):
        text = ("维度 $$d$$，矩阵 $$X_1$$。\n$$x+y$$\n$$\nx+$y$\n$$\n"
                "```bash\necho '$x$'\n```\n"
                "~~~~text\n$x$ in code\n~~~\n~~~~\n"
                "代码 `$x$` 和 ``$y$``。\n价格 \\$8/\\$40。\n")
        self.assertEqual(list(single_dollar_spans(text)), [])

    def test_detection_resumes_after_display_and_code(self):
        text = "$$\nx\n$$\n```text\nx\n```\n输入 $X$。\n"
        self.assertEqual([text[a:b] for a, b in single_dollar_spans(text)], ["$X$"])

    def test_book_has_no_single_dollar_inline_math(self):
        paths = [ROOT / "README.md"]
        paths += sorted(ROOT.glob("[0-9][0-9]_*/*.md"))
        paths += sorted((ROOT / "appendix").glob("*.md"))
        failures = []
        for path in paths:
            text = path.read_text()
            for start, _ in single_dollar_spans(text):
                failures.append(f"{path.relative_to(ROOT)}:{text[:start].count(chr(10)) + 1}")
        self.assertEqual(len(failures), 0, "Single-dollar inline math: " + ", ".join(failures[:20]))
