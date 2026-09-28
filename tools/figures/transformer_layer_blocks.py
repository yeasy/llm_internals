"""图 3-15：串行 Pre-Norm 的两条计算分支与两条残差旁路。"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import ACCENT, MUTED, arrow, box, finish, label
from _style import image_path, use_cjk_font

OUTPUT = image_path("03_components", "transformer_layer_blocks.png")


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(8.4, 11.5))
    cx = 10.0

    def block(y, h, text, kind="neutral"):
        box(ax, 6, y, 8, h, text, kind, fontsize=13, linespacing=1.25)

    def down(y1, y2):
        arrow(ax, (cx, y1), (cx, y2))

    label(ax, 8, -0.4, "一层的规律：分支先归一化，算完加回原输入",
          fontsize=16, bold=True)
    label(ax, 8, 0.4, "串行 Pre-Norm；n 是本轮输入位置数", fontsize=12, color=MUTED)
    block(1.2, 1.4, "层输入 X\n[n, d_model]", "data")
    down(2.6, 3.8)
    block(3.8, 1.5, "Norm ①\n调整注意力分支的输入尺度", "weight")
    down(5.3, 5.9)
    block(5.9, 1.7, "多头注意力 + 输出投影\n输出 [n, d_model]", "weight")
    down(7.6, 8.3)
    block(8.3, 1.2, "X + 注意力输出")
    down(9.5, 10.2)
    block(10.2, 1.4, "中间结果 U\n[n, d_model]", "data")
    down(11.6, 12.8)
    block(12.8, 1.5, "Norm ②\n调整 MLP 分支的输入尺度", "weight")
    down(14.3, 14.9)
    block(14.9, 1.7, "MLP：加工后投回原宽度\n输出 [n, d_model]", "weight")
    down(16.6, 17.3)
    block(17.3, 1.2, "U + MLP 输出")
    down(18.5, 19.2)
    block(19.2, 1.4, "本层输出 Y\n[n, d_model]", "data")

    # 从各自归一化前的主干分叉，不从另一个子层的旧输入分叉。
    for fork_y, add_y, name in ((3.0, 8.9, "X"), (12.0, 17.9, "U")):
        ax.plot([cx, 1.7, 1.7], [fork_y, fork_y, add_y],
                color=ACCENT, lw=1.7)
        ax.plot(cx, fork_y, "o", color=ACCENT, markersize=4)
        arrow(ax, (1.7, add_y), (6, add_y), color=ACCENT, lw=1.7)
        label(ax, 3.5, fork_y + 2.9,
              f"保留 {name}\n[n, d_model]\n不经过 Norm",
              fontsize=12, color=ACCENT)
    label(ax, 3.5, 18.9, "第二次加回 U，\n不是层输入 X", fontsize=12, color=ACCENT)
    down(20.6, 21.3)
    label(ax, cx, 21.9, "还有下一层：Y 成为下一层的 X", fontsize=12)
    label(ax, 8, 23.2, "全部 L 层结束：最终 Norm → 取最后一行 → LM head",
          fontsize=12, bold=True)
    label(ax, 8, 24.1, "最终 Norm 不另加残差；Norm 的算法由模型决定",
          fontsize=12, color=MUTED)
    finish(fig, ax, OUTPUT, xlim=(0, 16), ylim=(25, -1.2))


if __name__ == "__main__":
    main()
