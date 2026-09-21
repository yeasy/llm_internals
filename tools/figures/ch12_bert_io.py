"""生成 12.1 节插图：BERT 的输入怎样构造、四类任务头从哪里取向量。

正文位置：12_encoder_models/12.1_bert.md
输出：12_encoder_models/_images/ch12_bert_io.png

上半幅：`[CLS] A [SEP] B [SEP]` 的三张嵌入表逐元素相加，得到 X(0)[T, 768]。
下半幅：编码器交出 X(L)[T, 768] 之后，四类任务头各自从哪些位置取向量。
数值与正文 12.1.1、12.1.4 一致。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import (ACCENT, DATA_EDGE, FS_SMALL, FS_TEXT, INK, MUTED, NEW_EDGE,
                      WEIGHT_EDGE, arrow, box, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("12_encoder_models", "ch12_bert_io.png")

TOKENS = ["[CLS]", "the", "cat", "[SEP]", "a", "pet", "[SEP]"]
SEGMENT = [0, 0, 0, 0, 1, 1, 1]

CW, CH, GAP = 1.55, 0.72, 0.10
PITCH = CW + GAP


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(9.6, 6.4))

    n = len(TOKENS)
    width = n * PITCH - GAP

    # ---------- 上半幅：三张嵌入表相加 ----------
    rows = [
        ("词元", [t for t in TOKENS], "data"),
        ("词嵌入 E", ["[768]"] * n, "weight"),
        ("段嵌入 S", [f"S{s}" for s in SEGMENT], "weight"),
        ("位置嵌入 P", [f"P{i}" for i in range(n)], "weight"),
    ]
    y_top = 9.0
    for k, (name, cells, kind) in enumerate(rows):
        y = y_top - k * (CH + 0.30)
        label(ax, -0.35, y + CH / 2, name, fontsize=FS_SMALL, color=MUTED, ha="right")
        for i, text in enumerate(cells):
            box(ax, i * PITCH, y, CW, CH, text, kind=kind, fontsize=FS_SMALL)

    y_sum = y_top - 4 * (CH + 0.30) - 0.24
    label(ax, -0.35, y_sum + CH / 2, "三者相加\n再 LayerNorm", fontsize=FS_SMALL,
          color=MUTED, ha="right")
    for i in range(n):
        box(ax, i * PITCH, y_sum, CW, CH, r"$x^{(0)}$", kind="new", fontsize=FS_SMALL)
        arrow(ax, (i * PITCH + CW / 2, y_sum + CH + 0.50),
              (i * PITCH + CW / 2, y_sum + CH + 0.04), color=NEW_EDGE, lw=1.1)
    label(ax, width + 0.35, y_sum + CH / 2, r"$X^{(0)}$[7, 768]",
          fontsize=FS_TEXT, color=NEW_EDGE, ha="left")
    label(ax, width + 0.35, y_top + CH / 2,
          "三张表同宽 768\n逐元素相加，不拼接", fontsize=FS_SMALL, color=WEIGHT_EDGE,
          ha="left")

    # ---------- 中间：编码器 ----------
    y_enc = y_sum - 1.30
    box(ax, 0, y_enc, width, 0.86,
        "12 层双向编码器（无因果掩码）", kind="data", fontsize=FS_TEXT)
    arrow(ax, (width / 2, y_sum - 0.04), (width / 2, y_enc + 0.86), color=DATA_EDGE, lw=1.4)

    y_out = y_enc - 1.16
    label(ax, -0.35, y_out + CH / 2, r"$X^{(L)}$", fontsize=FS_SMALL, color=MUTED, ha="right")
    for i in range(n):
        box(ax, i * PITCH, y_out, CW, CH, f"h{i}", kind="data", fontsize=FS_SMALL)
    arrow(ax, (width / 2, y_enc - 0.04), (width / 2, y_out + CH), color=DATA_EDGE, lw=1.4)
    label(ax, width + 0.35, y_out + CH / 2, r"$X^{(L)}$[7, 768]",
          fontsize=FS_TEXT, color=DATA_EDGE, ha="left")

    # ---------- 下半幅：四类任务头 ----------
    heads = [
        ("句级分类", [0], "只取 h0\n过 pooler\n再 [768, C]"),
        ("词元级标注", list(range(n)), "取每个 h_i\n共享一个头\n再 [768, C]"),
        ("抽取式问答", [4, 5], "两个向量 S、E\n与各 h 点积\n取 i ≤ j 最大"),
        ("句对任务", [0], "仍只取 h0\n两句靠段嵌入\n区分"),
    ]
    y_head = y_out - 2.85
    span_left, span_right = -3.4, width + 3.4
    hw = (span_right - span_left - 3 * 0.40) / 4
    hh = 2.20
    for k, (name, picks, note) in enumerate(heads):
        x = span_left + k * (hw + 0.40)
        box(ax, x, y_head, hw, hh, "", kind="plain")
        label(ax, x + hw / 2, y_head + hh - 0.34, name, fontsize=FS_SMALL,
              color=ACCENT, bold=True)
        label(ax, x + hw / 2, y_head + 0.82, note, fontsize=FS_SMALL, color=INK)
        for i in picks:
            arrow(ax, (i * PITCH + CW / 2, y_out - 0.02),
                  (x + hw / 2, y_head + hh + 0.04), color=ACCENT, lw=0.8, rad=0.05)

    finish(fig, ax, OUTPUT,
           xlim=(span_left - 0.4, span_right + 0.4), ylim=(y_head - 0.34, y_top + CH + 0.90))


if __name__ == "__main__":
    main()
