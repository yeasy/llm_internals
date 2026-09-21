"""生成 2.4 节的图：五种掩码都是同一张 [T, T] 的 0/−∞ 矩阵。

正文位置：02_attention/2.4_self_cross_causal.md
输出：02_attention/_images/ch02_mask_family.png

每幅都取 T = 8：行是 Query 位置，列是 Key 位置，深色格表示该位置被屏蔽。
滑动窗口取 w = 4（含自身），块对角取 3 + 5 两篇文档，前缀 LM 取前 3 个位置为前缀。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _diagram import (DATA_EDGE, DATA_FACE, FS_SMALL, FS_TEXT, INK, MASK_EDGE, MASK_FACE,
                      MUTED, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("02_attention", "ch02_mask_family.png")

T = 8
CELL = 0.46
GAP = 1.35          # 相邻两幅之间的空白
PREFIX = 3          # 前缀 LM 的前缀长度
WINDOW = 4          # 滑动窗口可见的位置数（含自身）
DOCS = (3, 5)       # 块对角：两篇文档的长度


def doc_of(i: int) -> int:
    return 0 if i < DOCS[0] else 1


MASKS = {
    "（a）双向\n编码器自注意力": lambda i, j: True,
    "（b）因果\n仅解码器": lambda i, j: j <= i,
    "（c）前缀 LM\n前 3 个位置互相可见": lambda i, j: j <= i or (i < PREFIX and j < PREFIX),
    "（d）滑动窗口\n窗口 4，含自身": lambda i, j: i - WINDOW < j <= i,
    "（e）块对角因果\n两篇文档 3 + 5": lambda i, j: j <= i and doc_of(i) == doc_of(j),
}


def panel(ax, x0, y0, title, visible):
    n_vis = []
    for i in range(T):
        row = 0
        for j in range(T):
            ok = visible(i, j)
            row += ok
            face, edge = (DATA_FACE, DATA_EDGE) if ok else (MASK_FACE, MASK_EDGE)
            ax.add_patch(Rectangle((x0 + j * CELL, y0 - (i + 1) * CELL), CELL, CELL,
                                   facecolor=face, edgecolor=edge, linewidth=0.7, zorder=2))
        n_vis.append(row)
    ax.add_patch(Rectangle((x0, y0 - T * CELL), T * CELL, T * CELL, fill=False,
                           edgecolor=INK, linewidth=1.2, zorder=3))
    label(ax, x0 + T * CELL / 2, y0 + 0.62, title, fontsize=FS_TEXT, bold=True)
    label(ax, x0 + T * CELL / 2, y0 - T * CELL - 0.42,
          "最后一行可见 %d 个" % n_vis[-1], fontsize=FS_SMALL, color=MUTED)


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(9.9, 2.9))
    width = T * CELL
    for k, (title, fn) in enumerate(MASKS.items()):
        panel(ax, k * (width + GAP), 0.0, title, fn)
    # 坐标轴说明只标在第一幅上
    ax.text(-0.3, -T * CELL / 2, "Query 位置 1 到 8", fontsize=FS_SMALL, color=MUTED,
            ha="center", va="center", rotation=90, zorder=6)
    label(ax, width / 2, -T * CELL - 0.95, "Key 位置 1 到 8", fontsize=FS_SMALL, color=MUTED)
    legend = [(DATA_FACE, DATA_EDGE, "可见：加 0"), (MASK_FACE, MASK_EDGE, "屏蔽：加 -∞")]
    total = 5 * width + 4 * GAP
    lx = total / 2 + 0.2
    for m, (face, edge, text) in enumerate(legend):
        ax.add_patch(Rectangle((lx + m * 3.8, -T * CELL - 1.55), CELL, CELL, facecolor=face,
                               edgecolor=edge, linewidth=0.9, zorder=2))
        label(ax, lx + m * 3.8 + CELL + 0.15, -T * CELL - 1.55 + CELL / 2, text,
              fontsize=FS_SMALL, color=MUTED, ha="left")
    finish(fig, ax, OUTPUT,
           xlim=(-0.6, total + 0.2), ylim=(-T * CELL - 1.9, 1.25))


if __name__ == "__main__":
    main()
