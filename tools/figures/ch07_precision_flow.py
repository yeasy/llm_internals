"""生成 7.6 节的图：混合精度训练的一步里，每个张量是什么精度、各占每参数几个字节。

正文位置：07_distributed_training/7.6_mixed_precision.md
输出：07_distributed_training/_images/ch07_precision_flow.png

一圈四步：FP32 主权重转成 16 位副本；前向与反向在 16 位上做，得到 16 位梯度；
梯度归约后交给优化器；优化器在 FP32 上更新一阶矩、二阶矩与主权重。
橙色是长期驻留的模型状态，方框下标出每参数字节数，合计 2 + 2 + 4 + 4 + 4 = 16；
蓝色是随批次变化的激活。紫色标注只在 FP16 下需要：反向之前损失乘 S，更新之前梯度除以 S 并检查 inf/NaN。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import (ACCENT, FS_SMALL, FS_TEXT, FS_TITLE, MUTED, arrow, box, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("07_distributed_training", "ch07_precision_flow.png")

W, H = 5.9, 2.0


def node(ax, x, y, text, kind, note):
    box(ax, x, y, W, H, text, kind, fontsize=FS_TEXT)
    label(ax, x + W / 2, y - 0.45, note, fontsize=FS_SMALL, color=MUTED)


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(9.8, 6.0))
    label(ax, 0.0, 12.6, "混合精度的一步：16 位负责算，FP32 负责记账", ha="left", fontsize=FS_TITLE,
          bold=True)
    xs = (0.0, 9.4, 18.8)
    top, bot = 8.6, 2.6
    node(ax, xs[0], top, "FP32 主权重", "weight", "4 字节/参数")
    node(ax, xs[1], top, "16 位权重副本\n（BF16 或 FP16）", "weight", "2 字节/参数")
    node(ax, xs[2], top, "前向与反向\n激活为 16 位", "data", "随批量与序列长度变化")
    node(ax, xs[2], bot, "16 位梯度", "weight", "2 字节/参数")
    node(ax, xs[1], bot, "梯度归约\n（可升到 FP32）", "neutral", "AllReduce 或 Reduce-Scatter")
    node(ax, xs[0], bot, "Adam 更新（FP32）\n一阶矩 + 二阶矩", "weight", "4 + 4 字节/参数")

    ym, yb = top + H / 2, bot + H / 2
    arrow(ax, (xs[0] + W, ym), (xs[1], ym))
    label(ax, (xs[0] + W + xs[1]) / 2, ym + 0.45, "舍入到 16 位", fontsize=FS_SMALL)
    arrow(ax, (xs[1] + W, ym), (xs[2], ym))
    arrow(ax, (xs[2] + W / 2, top - 0.9), (xs[2] + W / 2, bot + H))
    arrow(ax, (xs[2], yb), (xs[1] + W, yb))
    arrow(ax, (xs[1], yb), (xs[0] + W, yb))
    arrow(ax, (xs[0] + W / 2, bot + H), (xs[0] + W / 2, top - 0.9))
    label(ax, xs[0] + W / 2 + 0.3, (top + bot + H) / 2 + 0.35, "在 FP32 上累加更新量", ha="left",
          fontsize=FS_SMALL)

    label(ax, xs[2] + W / 2 - 0.3, (top + bot + H) / 2 - 0.2, "仅 FP16：损失先乘 S",
          ha="right", fontsize=FS_SMALL, color=ACCENT)
    label(ax, (xs[0] + W + xs[1]) / 2, bot + H + 1.0, "仅 FP16：梯度除以 S，", fontsize=FS_SMALL,
          color=ACCENT)
    label(ax, (xs[0] + W + xs[1]) / 2, bot + H + 0.45, "遇 inf/NaN 跳过本步", fontsize=FS_SMALL,
          color=ACCENT)

    label(ax, 0.0, 0.75,
          "橙色为长期驻留的模型状态：2 + 2 + 4 + 4 + 4 = 16 字节/参数，与纯 FP32 训练的 4 + 4 + 4 + 4 相同。",
          ha="left", fontsize=FS_SMALL)
    label(ax, 0.0, 0.15, "混合精度省下的是激活显存、通信字节和矩阵乘法的时间，不是模型状态。",
          ha="left", fontsize=FS_SMALL, color=ACCENT)
    finish(fig, ax, OUTPUT, xlim=(-0.3, xs[2] + W + 0.3), ylim=(-0.3, 13.3))


if __name__ == "__main__":
    main()
