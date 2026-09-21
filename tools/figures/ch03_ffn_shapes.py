"""生成图 3-3：两矩阵 FFN 与三矩阵 SwiGLU 的形状对照。

正文位置：03_components/3.4_feedforward.md
输出：03_components/_images/ch03_ffn_shapes.png

左：GPT-3 Small 的两矩阵 FFN（d_ff = 4d）。右：Llama 3 8B 的三矩阵 SwiGLU
（gate 与 up 并行投影，逐元素相乘后过 down）。每个方框下方标形状，箭头旁标参数量。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from _diagram import (ACCENT, DATA_EDGE, FS_NAME, FS_SMALL, FS_TEXT, FS_TITLE, INK, MUTED,
                      arrow, box, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("03_components", "ch03_ffn_shapes.png")


def shape(ax, x, y, text):
    label(ax, x, y, text, fontsize=FS_SMALL, color=DATA_EDGE, ha="left")


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(12.0, 7.0))

    # ---------------- 左：两矩阵 FFN ----------------
    LX = 1.0
    label(ax, LX + 4.0, 15.4, "两矩阵 FFN（GPT-3 Small）", fontsize=FS_NAME, bold=True)
    box(ax, LX + 2.6, 13.2, 2.8, 1.2, "x", kind="data", fontsize=FS_NAME)
    shape(ax, LX + 5.7, 13.8, "[T, 768]")

    box(ax, LX + 2.0, 11.0, 4.0, 1.2, "W_1  升维", kind="weight", fontsize=FS_TEXT)
    shape(ax, LX + 6.3, 11.6, "[768, 3072]")
    arrow(ax, (LX + 4.0, 13.2), (LX + 4.0, 12.2), color=MUTED)

    box(ax, LX + 2.0, 8.8, 4.0, 1.2, "GELU 逐元素", kind="neutral", fontsize=FS_TEXT)
    shape(ax, LX + 6.3, 9.4, "[T, 3072]")
    arrow(ax, (LX + 4.0, 11.0), (LX + 4.0, 10.0), color=MUTED)

    box(ax, LX + 2.0, 6.6, 4.0, 1.2, "W_2  降维", kind="weight", fontsize=FS_TEXT)
    shape(ax, LX + 6.3, 7.2, "[3072, 768]")
    arrow(ax, (LX + 4.0, 8.8), (LX + 4.0, 7.8), color=MUTED)

    box(ax, LX + 2.6, 4.4, 2.8, 1.2, "输出", kind="data", fontsize=FS_NAME)
    shape(ax, LX + 5.7, 5.0, "[T, 768]")
    arrow(ax, (LX + 4.0, 6.6), (LX + 4.0, 5.6), color=MUTED)

    label(ax, LX + 4.0, 3.3, "参数 2 d d_ff = 8 $d^2$", fontsize=FS_TEXT, color=ACCENT)

    # ---------------- 右：三矩阵 SwiGLU ----------------
    RX = 13.0
    label(ax, RX + 5.0, 15.4, "三矩阵 SwiGLU（Llama 3 8B）", fontsize=FS_NAME, bold=True)
    box(ax, RX + 3.6, 13.2, 2.8, 1.2, "x", kind="data", fontsize=FS_NAME)
    shape(ax, RX + 6.7, 13.8, "[T, 4096]")

    box(ax, RX + 0.4, 11.0, 4.2, 1.2, "W_gate", kind="weight", fontsize=FS_TEXT)
    box(ax, RX + 5.4, 11.0, 4.2, 1.2, "W_up", kind="weight", fontsize=FS_TEXT)
    label(ax, RX + 2.5, 10.4, "[4096, 14336]", fontsize=FS_SMALL, color=DATA_EDGE)
    label(ax, RX + 7.5, 10.4, "[4096, 14336]", fontsize=FS_SMALL, color=DATA_EDGE)
    arrow(ax, (RX + 4.4, 13.2), (RX + 2.5, 12.2), color=MUTED)
    arrow(ax, (RX + 5.6, 13.2), (RX + 7.5, 12.2), color=MUTED)

    box(ax, RX + 0.4, 8.6, 4.2, 1.2, "SiLU 逐元素", kind="neutral", fontsize=FS_TEXT)
    arrow(ax, (RX + 2.5, 11.0), (RX + 2.5, 9.8), color=MUTED)

    ax.add_patch(Circle((RX + 5.0, 7.4), 0.5, facecolor="white", edgecolor=INK, lw=1.5, zorder=3))
    label(ax, RX + 5.0, 7.4, "×", fontsize=FS_NAME, zorder=4)
    arrow(ax, (RX + 2.5, 8.6), (RX + 4.6, 7.7), color=MUTED)
    arrow(ax, (RX + 7.5, 11.0), (RX + 5.4, 7.7), color=MUTED)
    label(ax, RX + 6.2, 8.5, "逐元素相乘", fontsize=FS_SMALL, color=MUTED, ha="left")
    shape(ax, RX + 0.4, 6.9, "[T, 14336]")

    box(ax, RX + 3.0, 5.2, 4.0, 1.2, "W_down", kind="weight", fontsize=FS_TEXT)
    shape(ax, RX + 7.3, 5.8, "[14336, 4096]")
    arrow(ax, (RX + 5.0, 6.9), (RX + 5.0, 6.4), color=MUTED)

    box(ax, RX + 3.6, 3.0, 2.8, 1.2, "输出", kind="data", fontsize=FS_NAME)
    shape(ax, RX + 6.7, 3.6, "[T, 4096]")
    arrow(ax, (RX + 5.0, 5.2), (RX + 5.0, 4.2), color=MUTED)

    label(ax, RX + 5.0, 1.9, "参数 3 d d_ff = 10.5 $d^2$", fontsize=FS_TEXT, color=ACCENT)

    finish(fig, ax, OUTPUT, xlim=(0, 24.2), ylim=(1.2, 16.1))


if __name__ == "__main__":
    main()
