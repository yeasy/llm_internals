"""生成 9.1 节的图：批量生成时右填充与左填充的对照，以及左填充下的位置索引。

正文位置：09_decoding/9.1_autoregressive_decode.md
输出：09_decoding/_images/ch09_padding_sides.png

三条 Prompt 长 5、3、2，填到 5 列；第 6 列是本轮新选出的词元。
位置索引按 Hugging Face generate() 的算法：attention_mask.cumsum(-1) - 1，填充位再置 0。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _diagram import (ACCENT, FS_SMALL, FS_TEXT, FS_TITLE, INK, KINDS, MASK_EDGE, MASK_FACE,
                      MUTED, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("09_decoding", "ch09_padding_sides.png")

LENS = [5, 3, 2]
NAMES = ["a", "b", "c"]
W = 5
CELL = 1.15


def cell(ax, x, y, text, kind, *, accent=False):
    if kind == "pad":
        face, edge = MASK_FACE, MASK_EDGE
    else:
        face, edge = KINDS[kind]
    ax.add_patch(Rectangle((x, y), CELL, CELL, facecolor=face,
                           edgecolor=ACCENT if accent else edge,
                           linewidth=2.4 if accent else 1.0, zorder=3 if accent else 2))
    ax.text(x + CELL / 2, y + CELL / 2, text, ha="center", va="center", fontsize=FS_SMALL,
            color=INK, zorder=4)


def panel(ax, x0, top, title, side):
    label(ax, x0, top + 0.9, title, fontsize=FS_TEXT, bold=True, ha="left")
    for c in range(W):
        label(ax, x0 + (c + 0.5) * CELL, top + 0.25, f"列 {c + 1}", fontsize=FS_SMALL, color=MUTED)
    label(ax, x0 + (W + 0.5) * CELL, top + 0.25, "新词元", fontsize=FS_SMALL, color=MUTED)
    for r, (n, name) in enumerate(zip(LENS, NAMES)):
        y = top - (r + 1) * CELL
        pad = W - n
        for c in range(W):
            is_pad = c >= n if side == "right" else c < pad
            k = c if side == "right" else c - pad
            last = c == W - 1
            if is_pad:
                cell(ax, x0 + c * CELL, y, "PAD", "pad", accent=last)
            else:
                cell(ax, x0 + c * CELL, y, f"{name}{k + 1}", "data", accent=last)
        cell(ax, x0 + W * CELL, y, f"{name}{n + 1}", "new")
    return top - 3 * CELL


def main():
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(11.6, 7.2))
    label(ax, 0.2, 12.6, "三条 Prompt 长 5、3、2，拼成一批：每轮都从最后一列（紫框）取 logits",
          fontsize=FS_TITLE, bold=True, ha="left")

    xl, xr, top = 0.4, 10.2, 10.6
    yb = panel(ax, xl, top, "右填充：短序列的最后一列是 PAD", "right")
    label(ax, xl, yb - 0.55, "b、c 由 PAD 位置的 logits 选词，\n新词元还被接在 PAD 之后", fontsize=FS_SMALL,
          color=MUTED, ha="left", va="top")
    yb = panel(ax, xr, top, "左填充：最后一列全是真实词元", "left")
    label(ax, xr, yb - 0.55, "真实内容右端对齐，新词元紧接其后；\n左侧 PAD 由注意力掩码遮掉", fontsize=FS_SMALL,
          color=MUTED, ha="left", va="top")

    # 位置索引：以左填充的序列 b 为例
    y0 = 2.9
    label(ax, 0.4, y0 + 1.75, "左填充下，序列 b 的位置索引怎样算出", fontsize=FS_TEXT, bold=True, ha="left")
    rows = [
        ("attention_mask", [0, 0, 1, 1, 1], "plain"),
        ("cumsum(-1) - 1", [-1, -1, 0, 1, 2], "plain"),
        ("填充位改写为 0 → position_ids", [0, 0, 0, 1, 2], "data"),
    ]
    xg = 8.2
    for i, (name, vals, kind) in enumerate(rows):
        y = y0 - i * CELL
        label(ax, xg - 0.3, y + CELL / 2, name, fontsize=FS_SMALL, ha="right")
        for c, v in enumerate(vals):
            cell(ax, xg + c * CELL, y, str(v), "pad" if c < 2 else kind)
    label(ax, xg + 5 * CELL + 0.4, y0 - 2 * CELL + CELL / 2, "b1 的位置是 0，\n与单条推理一致",
          fontsize=FS_SMALL, color=MUTED, ha="left")

    finish(fig, ax, OUTPUT, xlim=(0, 19.6), ylim=(-0.1, 13.2))


if __name__ == "__main__":
    main()
