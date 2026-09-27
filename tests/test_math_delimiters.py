"""Inline formulas must not become display blocks inside prose or table cells."""
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
PAIR = re.compile(r"(?<![\\$])\$\$(?!\$)([^\n$]+?)(?<!\\)\$\$(?!\$)")
CODE = re.compile(r"(`+).*?\1")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def inline_display_spans(text):
    """Return character spans of unambiguous inline double-dollar formulas."""
    offset, fence, in_display = 0, None, False
    for line in text.splitlines(keepends=True):
        marker = FENCE.match(line)
        if fence is not None:
            if (marker and marker[1][0] == fence[0]
                    and len(marker[1]) >= len(fence) and not marker[2].strip()):
                fence = None
        elif marker and not in_display:
            fence = marker[1]
        elif line.strip() == "$$":
            in_display = not in_display
        elif not in_display:
            code_spans = [match.span() for match in CODE.finditer(line)]
            for match in PAIR.finditer(line):
                if any(start < match.end() and end > match.start()
                       for start, end in code_spans):
                    continue
                if line.strip() == match[0]:
                    continue  # A complete display equation on one line.
                yield offset + match.start(), offset + match.end()
        offset += len(line)


class MathDelimiterTests(unittest.TestCase):
    def test_detects_inline_formulas_in_prose_and_tables(self):
        text = "维度 $$d$$，矩阵 $$X_1$$。\n| $$Q$$ | 输入 |\n"
        self.assertEqual([text[a:b] for a, b in inline_display_spans(text)],
                         ["$$d$$", "$$X_1$$", "$$Q$$"])

    def test_preserves_display_equations_and_code(self):
        text = ("$$x+y$$\n$$\nx+y\n$$\n"
                "```bash\necho '$$x$$'\n```\n"
                "~~~~text\n$$x$$ in code\n~~~\n~~~~\n"
                "代码 `$$x$$` 和 ``$$y$$``。\n普通 $x$。\n")
        self.assertEqual(list(inline_display_spans(text)), [])

    def test_detection_resumes_after_display_and_code(self):
        text = "$$\nx\n$$\n```text\nx\n```\n输入 $$X$$。\n"
        self.assertEqual([text[a:b] for a, b in inline_display_spans(text)], ["$$X$$"])

    def test_book_has_no_inline_display_delimiters(self):
        paths = [ROOT / "README.md"]
        paths += sorted(ROOT.glob("[0-9][0-9]_*/*.md"))
        paths += sorted((ROOT / "appendix").glob("*.md"))
        failures = []
        for path in paths:
            text = path.read_text()
            for start, _ in inline_display_spans(text):
                failures.append(f"{path.relative_to(ROOT)}:{text[:start].count(chr(10)) + 1}")
        self.assertEqual(len(failures), 0, "Inline $$ display math: " + ", ".join(failures[:20]))
