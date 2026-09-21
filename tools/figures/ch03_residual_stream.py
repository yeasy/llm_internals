"""生成图 3-4：残差流——各子层从同一条通路读、向同一条通路写。

正文位置：03_components/3.5_residual.md
输出：03_components/_images/ch03_residual_stream.png

横向一条粗带表示宽度固定为 d_model 的残差流；每个子层先从带上读一份并归一化，
再把自己的输出加回带上。右端标出 h_L = h_0 + 各子层输出之和。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from _diagram import (ACCENT, DATA_EDGE, DATA_FACE, FS_NAME, FS_SMALL, FS_TEXT, FS_TITLE,
                      INK, MUTED, NEW_EDGE, arrow, box, dots, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("03_components", "ch03_residual_stream.png")

SY, SH = 6.6, 1.2          # 残差流的下沿与高度
GAP, NORM_H, SUB_H = 1.1, 0.9, 1.2
SUBS = [(3.6, "注意力", True), (7.8, "MLP", False),
        (12.0, "注意力", True), (16.2, "MLP", False)]


def plus(ax, x):
    ax.add_patch(Circle((x, SY + SH / 2), 0.45, facecolor="white",
                        edgecolor=INK, lw=1.5, zorder=5))
    label(ax, x, SY + SH / 2, "+", fontsize=FS_NAME, zorder=6)


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(11.0, 5.6))

    label(ax, 10.0, 14.3, "残差流：一条宽度固定的通路，每个子层读一次、写一次",
          fontsize=FS_TITLE, bold=True)

    ax.add_patch(Rectangle((0.6, SY), 18.8, SH, facecolor=DATA_FACE,
                           edgecolor=DATA_EDGE, lw=1.6, zorder=1))
    label(ax, 1.6, SY + SH / 2, "$h_0$", fontsize=FS_NAME, zorder=3)

    for x, name, up in SUBS:
        read_x, write_x = x - 0.9, x + 1.1
        if up:
            ny = SY + SH + GAP
            sy = ny + NORM_H + 0.9
            box(ax, x - 1.9, ny, 2.0, NORM_H, "归一化", kind="neutral", fontsize=FS_SMALL)
            box(ax, x - 1.9, sy, 3.6, SUB_H, name, kind="weight", fontsize=FS_TEXT)
            arrow(ax, (read_x, SY + SH), (read_x, ny), color=DATA_EDGE, lw=1.4)
            arrow(ax, (read_x, ny + NORM_H), (read_x, sy), color=DATA_EDGE, lw=1.4)
            arrow(ax, (write_x, sy), (write_x, SY + SH + 0.45), color=NEW_EDGE, lw=1.4)
        else:
            ny = SY - GAP - NORM_H
            sy = ny - 0.9 - SUB_H
            box(ax, x - 1.9, ny, 2.0, NORM_H, "归一化", kind="neutral", fontsize=FS_SMALL)
            box(ax, x - 1.9, sy, 3.6, SUB_H, name, kind="weight", fontsize=FS_TEXT)
            arrow(ax, (read_x, SY), (read_x, ny + NORM_H), color=DATA_EDGE, lw=1.4)
            arrow(ax, (read_x, ny), (read_x, sy + SUB_H), color=DATA_EDGE, lw=1.4)
            arrow(ax, (write_x, sy + SUB_H), (write_x, SY - 0.45), color=NEW_EDGE, lw=1.4)
        plus(ax, write_x)

    dots(ax, 18.8, SY + SH / 2, horizontal=True, color=INK, spread=0.42, r=0.11)

    label(ax, 3.6, 12.6, "第 1 层", fontsize=FS_SMALL, color=MUTED)
    label(ax, 12.0, 12.6, "第 2 层", fontsize=FS_SMALL, color=MUTED)
    label(ax, 0.6, 5.6, "嵌入 + 位置", fontsize=FS_SMALL, color=MUTED, ha="left")
    label(ax, 19.4, 12.6, "末端归一化 → LM head", fontsize=FS_SMALL, color=MUTED, ha="right")

    label(ax, 0.6, 0.8, "蓝：读　青绿：写　通路宽度始终是 d_model",
          fontsize=FS_SMALL, color=MUTED, ha="left")
    label(ax, 19.4, 0.8, "$h_L = h_0 + \\sum_\\ell F_\\ell$", fontsize=FS_NAME,
          color=ACCENT, ha="right")

    finish(fig, ax, OUTPUT, xlim=(0, 20.0), ylim=(0.2, 14.8))


if __name__ == "__main__":
    main()
