"""生成图 3-1：一段文本怎样经预分词、字节化与合并变成词元 ID。

正文位置：03_components/3.1_tokenization.md
输出：03_components/_images/ch03_bpe_pipeline.png

四行自上而下：原始文本 → 预分词正则切出的片段 → 每个片段的 UTF-8 字节 →
按合并表逐次合并后的词元与 ID。片段与 ID 由 tiktoken 的 gpt2 编码实测得到。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import ACCENT, FS_NAME, FS_SMALL, FS_TEXT, FS_TITLE, MUTED, arrow, box, finish, label
from _style import image_path, use_cjk_font

OUTPUT = image_path("03_components", "ch03_bpe_pipeline.png")

PIECES = ["I", "_love", "_模型", "_2024", "!"]
BYTES = ["73", "32 108 111\n118 101", "32 230 168 161\n229 158 139", "32 50 48 50 52", "33"]
TOKENS = [["I"], ["_love"], ["_?", "?", "?", "?", "?", "?"], ["_2024"], ["!"]]
IDS = [["40"], ["1842"], ["10545", "101", "94", "161", "252", "233"], ["48609"], ["0"]]

WIDTHS = [1.5, 3.8, 7.4, 4.0, 1.5]
GAP = 0.5
X0 = 4.6
ROWS = {"text": 13.4, "piece": 11.0, "byte": 8.4, "token": 5.8}
H = 1.4


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(11.6, 6.4))

    xs, x = [], X0
    for w in WIDTHS:
        xs.append(x)
        x += w + GAP
    total = x - GAP - X0

    label(ax, X0 + total / 2, 15.7, "一段文本到词元 ID：切片 → 字节 → 合并",
          fontsize=FS_TITLE, bold=True)

    def row_label(y, text):
        label(ax, X0 - 0.5, y + H / 2, text, fontsize=FS_TEXT, color=MUTED, ha="right")

    box(ax, X0, ROWS["text"], total, H, "I love 模型 2024!", kind="data", fontsize=FS_NAME)
    row_label(ROWS["text"], "原始文本")

    row_label(ROWS["piece"], "预分词片段\n（空格归属后一片段）")
    for x0, w, p in zip(xs, WIDTHS, PIECES):
        box(ax, x0, ROWS["piece"], w, H, p, kind="data", fontsize=FS_NAME)
        arrow(ax, (x0 + w / 2, ROWS["text"]), (x0 + w / 2, ROWS["piece"] + H), color=MUTED, lw=1.2)

    row_label(ROWS["byte"], "UTF-8 字节\n（基础词表 256 个）")
    for x0, w, b in zip(xs, WIDTHS, BYTES):
        box(ax, x0, ROWS["byte"], w, H, b, kind="neutral", fontsize=FS_SMALL)
        arrow(ax, (x0 + w / 2, ROWS["piece"]), (x0 + w / 2, ROWS["byte"] + H), color=MUTED, lw=1.2)

    row_label(ROWS["token"], "合并后的词元\n与 ID")
    for x0, w, toks, ids in zip(xs, WIDTHS, TOKENS, IDS):
        arrow(ax, (x0 + w / 2, ROWS["byte"]), (x0 + w / 2, ROWS["token"] + H), color=MUTED, lw=1.2)
        cw = w / len(toks)
        kind = "new" if len(toks) > 1 else "weight"
        fs = FS_SMALL if len(toks) == 1 else 8.8
        for k, (t, i) in enumerate(zip(toks, ids)):
            box(ax, x0 + k * cw, ROWS["token"], cw, H, f"{t}\n{i}", kind=kind,
                fontsize=fs, rounded=False)

    label(ax, X0, 4.7, "5 个片段变成 10 个词元", fontsize=FS_TEXT, color=ACCENT, ha="left")
    label(ax, X0 + total, 4.7, "下划线代表空格，问号代表半个字符",
          fontsize=FS_SMALL, color=MUTED, ha="right")

    finish(fig, ax, OUTPUT, xlim=(0, X0 + total + 0.4), ylim=(4.2, 16.4))


if __name__ == "__main__":
    main()
