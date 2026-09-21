"""生成 2.3 节插图：多头注意力在实现里是“一次大投影，两次变形”。

正文位置：02_attention/2.3_multi_head.md
输出：02_attention/_images/ch02_multihead_reshape.png

形状写法与 3.8.3 的清单一致，省略 batch 维。图中画 4 个头只为画得下，
正文算例用的是 GPT-3 6.7B 的 n_h = 32、d_h = 128、d_model = 4096。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _diagram import (ACCENT, DATA_EDGE, DATA_FACE, FS_NAME, FS_SMALL, FS_TEXT, INK, MUTED,
                      NEW_EDGE, NEW_FACE, WEIGHT_EDGE, WEIGHT_FACE, arrow, box, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("02_attention", "ch02_multihead_reshape.png")

N_DRAW = 4          # 图上画几个头
SEG_W = 1.35        # 每个头占的宽度
BAND_H = 1.5


def band(ax, x, y, n_seg, *, face, edge, seg_labels, width=SEG_W, height=BAND_H):
    for k in range(n_seg):
        ax.add_patch(Rectangle((x + k * width, y), width, height, facecolor=face,
                               edgecolor=edge, linewidth=1.2, zorder=2))
        label(ax, x + (k + 0.5) * width, y + height / 2, seg_labels[k], fontsize=FS_SMALL)
    return x + n_seg * width


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(9.6, 6.6))
    heads = ["头 1", "头 2", "头 3", "头 4"]

    # 第一行：一次大投影
    label(ax, 0.0, 11.5, "第 1 步  一次大矩阵乘法，不是 n_h 次小的", fontsize=FS_TEXT,
          bold=True, ha="left")
    box(ax, 0.0, 9.5, 2.0, BAND_H, "X\n[T, 4096]", kind="data", fontsize=FS_SMALL)
    label(ax, 2.35, 9.5 + BAND_H / 2, "×", fontsize=FS_NAME, color=MUTED)
    box(ax, 2.7, 9.5, 3.0, BAND_H, "W_Q\n[4096, 32×128]", kind="weight", fontsize=FS_SMALL)
    label(ax, 6.05, 9.5 + BAND_H / 2, "→", fontsize=FS_NAME, color=MUTED)
    right = band(ax, 6.4, 9.5, N_DRAW, face=DATA_FACE, edge=DATA_EDGE, seg_labels=heads)
    label(ax, 6.4 + (right - 6.4) / 2, 9.2, "Q[T, 32×128]：第 h 段就是第 h 个头的 Q",
          fontsize=FS_SMALL, color=MUTED)
    label(ax, right + 0.15, 9.5 + BAND_H / 2, "…", fontsize=FS_NAME, color=MUTED, ha="left")

    # 第二行：两次变形
    arrow(ax, (8.4, 8.9), (8.4, 8.1), color=ACCENT)
    label(ax, 8.65, 8.5, "view 成 [T, 32, 128]，再 transpose 成 [32, T, 128]",
          fontsize=FS_SMALL, color=ACCENT, ha="left")

    label(ax, 0.0, 7.4, "第 2 步  每个头各自做一遍注意力，互不相干", fontsize=FS_TEXT,
          bold=True, ha="left")
    for k in range(N_DRAW):
        x = k * 3.3
        box(ax, x, 5.4, 1.35, 1.2, f"Q{k+1},K{k+1}\n[T, 128]", kind="data", fontsize=FS_SMALL)
        label(ax, x + 1.6, 6.0, "→", fontsize=FS_TEXT, color=MUTED)
        box(ax, x + 1.85, 5.4, 1.1, 1.2, f"A{k+1}\n[T, T]", kind="data", fontsize=FS_SMALL)
    label(ax, 0.0, 4.95, "分数矩阵有 n_h 张：显存按 n_h × T × T 增长，而计算量与 n_h 无关",
          fontsize=FS_SMALL, color=MUTED, ha="left")

    # 第三行：拼接与输出投影
    label(ax, 0.0, 3.9, "第 3 步  拼回一条，再由 W_O 把各头混合", fontsize=FS_TEXT,
          bold=True, ha="left")
    right = band(ax, 0.0, 1.9, N_DRAW, face=DATA_FACE, edge=DATA_EDGE,
                 seg_labels=[f"C{k+1}" for k in range(N_DRAW)])
    label(ax, right + 0.15, 1.9 + BAND_H / 2, "…", fontsize=FS_NAME, color=MUTED, ha="left")
    label(ax, (right) / 2, 1.5, "Concat[T, 32×128]", fontsize=FS_SMALL, color=MUTED)
    label(ax, right + 0.95, 1.9 + BAND_H / 2, "×", fontsize=FS_NAME, color=MUTED)
    box(ax, right + 1.35, 1.9, 3.0, BAND_H, "W_O\n[32×128, 4096]", kind="weight",
        fontsize=FS_SMALL)
    label(ax, right + 4.7, 1.9 + BAND_H / 2, "→", fontsize=FS_NAME, color=MUTED)
    box(ax, right + 5.05, 1.9, 2.4, BAND_H, "注意力输出\n[T, 4096]", kind="data",
        fontsize=FS_SMALL)
    label(ax, 0.0, 1.0,
          "没有 W_O，第 h 个头就只能写进输出向量的第 h 段；W_O 的第 h 段行决定它写到哪里",
          fontsize=FS_SMALL, color=MUTED, ha="left")

    finish(fig, ax, OUTPUT, xlim=(-0.4, 14.6), ylim=(0.6, 12.0))


if __name__ == "__main__":
    main()
