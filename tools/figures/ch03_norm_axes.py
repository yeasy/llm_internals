"""生成图 3-5：BatchNorm 与 LayerNorm 在 [B, T, d] 上沿哪个轴求统计量。

正文位置：03_components/3.6_layer_norm.md
输出：03_components/_images/ch03_norm_axes.png

把 [B, T, d] 摊成一张表：每一行是一个词元（B 个样本 × T 个位置），每一列是一个通道。
左图给 LayerNorm/RMSNorm 涂一行，右图给 BatchNorm 涂一列，并标出各自的统计量个数。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _diagram import (ACCENT, DATA_EDGE, DATA_FACE, FS_NAME, FS_SMALL, FS_TEXT, FS_TITLE,
                      MUTED, NEUTRAL_EDGE, NEUTRAL_FACE, NEW_EDGE, NEW_FACE, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("03_components", "ch03_norm_axes.png")

ROWS = [("样本 1", "位置 1"), ("样本 1", "位置 2"), ("样本 1", "位置 3"),
        ("样本 2", "位置 1"), ("样本 2", "位置 2"), ("样本 2", "位置 3")]
NCOL = 6
CW, CH = 0.92, 0.92


def panel(ax, x0, y0, title, mode):
    label(ax, x0 + NCOL * CW / 2, y0 + len(ROWS) * CH + 1.5, title,
          fontsize=FS_NAME, bold=True)
    label(ax, x0 + NCOL * CW / 2, y0 + len(ROWS) * CH + 0.62,
          "d 个通道 →", fontsize=FS_SMALL, color=MUTED)
    for i, (s, p) in enumerate(ROWS):
        y = y0 + (len(ROWS) - 1 - i) * CH
        for j in range(NCOL):
            hit = (mode == "ln" and i == 1) or (mode == "bn" and j == 2)
            face, edge = (NEW_FACE, NEW_EDGE) if hit else (DATA_FACE, DATA_EDGE)
            ax.add_patch(Rectangle((x0 + j * CW, y), CW, CH, facecolor=face,
                                   edgecolor=edge, lw=1.0))
        label(ax, x0 - 0.3, y + CH / 2, f"{s} {p}", fontsize=FS_SMALL,
              color=MUTED, ha="right")
    if mode == "ln":
        ax.add_patch(Rectangle((x0, y0 + (len(ROWS) - 2) * CH), NCOL * CW, CH,
                               fill=False, edgecolor=NEW_EDGE, lw=2.4))
        label(ax, x0 + NCOL * CW / 2, y0 - 0.9,
              "每个词元自己算一对 (μ, σ)，共 B × T 对", fontsize=FS_TEXT, color=ACCENT)
        label(ax, x0 + NCOL * CW / 2, y0 - 1.9,
              "γ、β 形状 [d]，与 B、T 无关", fontsize=FS_SMALL, color=MUTED)
    else:
        ax.add_patch(Rectangle((x0 + 2 * CW, y0), CW, len(ROWS) * CH,
                               fill=False, edgecolor=NEW_EDGE, lw=2.4))
        label(ax, x0 + NCOL * CW / 2, y0 - 0.9,
              "每个通道跨全批算一对 (μ, σ)，共 d 对", fontsize=FS_TEXT, color=ACCENT)
        label(ax, x0 + NCOL * CW / 2, y0 - 1.9,
              "批变小、有填充位、推理只有一条时都会失真", fontsize=FS_SMALL, color=MUTED)


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(10.6, 5.4))
    label(ax, 10.6, 10.9, "同一张 [B, T, d]：两种归一化沿不同的轴求统计量",
          fontsize=FS_TITLE, bold=True)
    panel(ax, 3.2, 2.6, "LayerNorm / RMSNorm：沿最后一维", "ln")
    panel(ax, 13.6, 2.6, "BatchNorm：沿批次与位置", "bn")
    finish(fig, ax, OUTPUT, xlim=(0, 21.2), ylim=(0.4, 11.4))


if __name__ == "__main__":
    main()
