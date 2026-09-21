"""生成图 3-14：归一化放在哪里——Post-Norm、Pre-Norm 与前后双归一化。

正文位置：03_components/3.6_layer_norm.md
输出：03_components/_images/ch03_norm_placement.png

三块并排，每块画一个子层：左边竖直的粗线是恒等路径（残差）。关键差别只有一处——
恒等路径上有没有归一化。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from _diagram import (ACCENT, DATA_EDGE, FS_NAME, FS_SMALL, FS_TEXT, FS_TITLE, INK, MUTED,
                      arrow, box, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("03_components", "ch03_norm_placement.png")

W = 3.0


def plus(ax, x, y):
    ax.add_patch(Circle((x, y), 0.4, facecolor="white", edgecolor=INK, lw=1.5, zorder=5))
    label(ax, x, y, "+", fontsize=FS_NAME, zorder=6)


def panel(ax, cx, title, kind, note):
    label(ax, cx, 15.0, title, fontsize=FS_NAME, bold=True)
    bx = cx - W / 2
    sx = cx - 2.2  # 恒等路径的竖线

    label(ax, cx, 1.1, note, fontsize=FS_SMALL, color=MUTED)
    label(ax, cx, 2.1, "输入 x", fontsize=FS_TEXT)
    # 主干：自下而上
    if kind == "post":
        box(ax, bx, 4.4, W, 1.2, "子层", kind="weight", fontsize=FS_TEXT)
        plus(ax, cx, 7.2)
        box(ax, bx, 8.9, W, 1.2, "归一化", kind="neutral", fontsize=FS_TEXT)
        arrow(ax, (cx, 2.6), (cx, 4.4), color=DATA_EDGE)
        arrow(ax, (cx, 5.6), (cx, 6.8), color=DATA_EDGE)
        arrow(ax, (cx, 7.6), (cx, 8.9), color=DATA_EDGE)
        arrow(ax, (cx, 10.1), (cx, 12.0), color=DATA_EDGE)
        # 恒等路径从输入绕到加号，但随后穿过归一化
        arrow(ax, (sx, 2.9), (sx, 7.2), color=ACCENT, lw=2.2)
        arrow(ax, (sx, 7.2), (cx - 0.4, 7.2), color=ACCENT, lw=2.2)
        label(ax, sx - 0.2, 5.0, "恒等\n路径", fontsize=FS_SMALL, color=ACCENT, ha="right")
        label(ax, cx, 12.6, "输出：仍要过一次归一化", fontsize=FS_SMALL, color=ACCENT)
    elif kind == "pre":
        box(ax, bx, 4.4, W, 1.2, "归一化", kind="neutral", fontsize=FS_TEXT)
        box(ax, bx, 6.8, W, 1.2, "子层", kind="weight", fontsize=FS_TEXT)
        plus(ax, cx, 9.6)
        arrow(ax, (cx, 2.6), (cx, 4.4), color=DATA_EDGE)
        arrow(ax, (cx, 5.6), (cx, 6.8), color=DATA_EDGE)
        arrow(ax, (cx, 8.0), (cx, 9.2), color=DATA_EDGE)
        arrow(ax, (cx, 10.0), (cx, 12.0), color=DATA_EDGE)
        arrow(ax, (sx, 2.9), (sx, 9.6), color=ACCENT, lw=2.2)
        arrow(ax, (sx, 9.6), (cx - 0.4, 9.6), color=ACCENT, lw=2.2)
        label(ax, sx - 0.2, 5.6, "恒等\n路径", fontsize=FS_SMALL, color=ACCENT, ha="right")
        label(ax, cx, 12.6, "输出：一路加法，不过归一化", fontsize=FS_SMALL, color=ACCENT)
    else:
        box(ax, bx, 3.6, W, 1.1, "归一化", kind="neutral", fontsize=FS_TEXT)
        box(ax, bx, 5.8, W, 1.1, "子层", kind="weight", fontsize=FS_TEXT)
        box(ax, bx, 8.0, W, 1.1, "归一化", kind="neutral", fontsize=FS_TEXT)
        plus(ax, cx, 10.4)
        arrow(ax, (cx, 2.6), (cx, 3.6), color=DATA_EDGE)
        arrow(ax, (cx, 4.7), (cx, 5.8), color=DATA_EDGE)
        arrow(ax, (cx, 6.9), (cx, 8.0), color=DATA_EDGE)
        arrow(ax, (cx, 9.1), (cx, 10.0), color=DATA_EDGE)
        arrow(ax, (cx, 10.8), (cx, 12.0), color=DATA_EDGE)
        arrow(ax, (sx, 2.9), (sx, 10.4), color=ACCENT, lw=2.2)
        arrow(ax, (sx, 10.4), (cx - 0.4, 10.4), color=ACCENT, lw=2.2)
        label(ax, sx - 0.2, 5.6, "恒等\n路径", fontsize=FS_SMALL, color=ACCENT, ha="right")
        label(ax, cx, 12.6, "输出：加法之前先限住分支幅度", fontsize=FS_SMALL, color=ACCENT)


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(11.6, 6.4))
    label(ax, 11.4, 16.4, "归一化放在哪里：恒等路径上有没有它，是唯一的结构差别",
          fontsize=FS_TITLE, bold=True)
    panel(ax, 3.6, "Post-Norm", "post", "原始 Transformer、BERT")
    panel(ax, 11.4, "Pre-Norm", "pre", "GPT-2 起、Llama")
    panel(ax, 19.2, "前后双归一化", "sandwich", "Gemma 2；OLMo 2 只留后一次")
    finish(fig, ax, OUTPUT, xlim=(0, 22.8), ylim=(0.4, 16.9))


if __name__ == "__main__":
    main()
