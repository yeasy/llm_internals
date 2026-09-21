"""生成图 3-14：真实 GPT 中一层 Transformer 的结构与各处的形状。

正文位置：03_components/3.8_gpt_inference_flow.md
输出：03_components/_images/transformer_layer_blocks.png

自下而上画出预归一化（Pre-Norm）的一层：归一化 → n_h 个头并行的注意力 → 拼接 →
输出投影 → 残差相加 → 归一化 → MLP → 残差相加；箭头旁标出数据的形状。右侧说明这一层
重复 L 次，最后一层的最后一行经 LM head 得到 logits。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from _diagram import (ACCENT, DATA_EDGE, FS_NAME, FS_SMALL, FS_TEXT, FS_TITLE, INK, MUTED,
                      arrow, box, dots, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("03_components", "transformer_layer_blocks.png")

CX = 12.0  # 主干的中心线


def plus(ax, x, y):
    ax.add_patch(Circle((x, y), 0.45, facecolor="white", edgecolor=INK, lw=1.5, zorder=3))
    label(ax, x, y, "+", fontsize=FS_NAME)


def shape(ax, x, y, text):
    label(ax, x, y, text, fontsize=FS_SMALL, color=DATA_EDGE, ha="left")


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(12.6, 10.0))

    label(ax, CX, 30.6, "第 $\ell$ 层：输入和输出都是 [T, d_model]，所以可以一层接一层",
          fontsize=FS_TITLE, bold=True)

    # 输入
    box(ax, CX - 4, 0.2, 8, 1.2, "第 $\ell$ 层的输入 X", "data", fontsize=FS_TEXT, bold=True)
    shape(ax, CX + 4.3, 0.8, "[T, d_model]")
    arrow(ax, (CX, 1.45), (CX, 2.95))

    # 归一化 1
    box(ax, CX - 3, 3.0, 6, 1.1, "归一化", "neutral")
    arrow(ax, (CX, 4.15), (CX, 5.0))

    # 分给各个头
    HW = 7.0
    heads_x = [CX - 11.6, CX - 3.9, CX + 4.6]
    names = ["头 1", "头 2", "头 n_h"]
    ax.plot([heads_x[0] + HW / 2, heads_x[2] + HW / 2], [5.0, 5.0], color=INK, lw=1.5)
    for hx, name in zip(heads_x, names):
        mid = hx + HW / 2
        arrow(ax, (mid, 5.0), (mid, 5.85))
        box(ax, hx, 5.9, HW, 1.7, "乘三个投影矩阵\n得到 Q、K、V", "weight", fontsize=FS_SMALL)
        arrow(ax, (mid, 7.65), (mid, 8.45))
        box(ax, hx, 8.5, HW, 2.6, f"{name}\n打分、掩码、\nSoftmax、读取 V", "neutral",
            fontsize=FS_SMALL)
        arrow(ax, (mid, 11.15), (mid, 12.75))
    dots(ax, CX + 3.85, 8.4, horizontal=True, color=INK, spread=0.3, r=0.09)
    label(ax, heads_x[0] + HW / 2 + 0.3, 8.05, "各为 [T, d_h]", fontsize=FS_SMALL,
          color=DATA_EDGE, ha="left")
    label(ax, heads_x[0] + HW / 2 + 0.3, 11.95, "context [T, d_h]", fontsize=FS_SMALL,
          color=DATA_EDGE, ha="left")
    label(ax, heads_x[2] + HW + 0.9, 6.75, "各头的 K、V\n写入本层缓存", ha="left",
          fontsize=FS_SMALL, color=ACCENT)
    arrow(ax, (heads_x[2] + HW + 0.05, 6.75), (heads_x[2] + HW + 0.8, 6.75), color=ACCENT, lw=1.2)
    label(ax, CX + 3.2, 4.55, "n_h 个头并行，各用各的权重", ha="left", fontsize=FS_SMALL,
          color=MUTED)

    # 拼接
    box(ax, CX - 11.6, 12.8, 23.2, 1.1, "拼接各头的 context", "neutral", fontsize=FS_TEXT)
    shape(ax, CX + 11.9, 13.35, "[T, n_h × d_h]")
    arrow(ax, (CX, 13.95), (CX, 14.85))
    box(ax, CX - 3, 14.9, 6, 1.1, "乘输出投影 W_O", "weight")
    shape(ax, CX + 3.3, 15.45, "[T, d_model]")
    arrow(ax, (CX, 16.05), (CX, 16.9))
    plus(ax, CX, 17.4)
    label(ax, CX + 0.8, 17.4, "残差相加", ha="left", fontsize=FS_SMALL, color=MUTED)

    # 残差旁路 1：从输入绕到第一个加号
    ax.plot([CX - 4, CX - 13.4, CX - 13.4], [0.8, 0.8, 17.4], color=INK, lw=1.5)
    arrow(ax, (CX - 13.4, 17.4), (CX - 0.5, 17.4))
    label(ax, CX - 13.7, 9.0, "原样绕过\n注意力", ha="right", fontsize=FS_SMALL, color=MUTED)

    # MLP 子层
    arrow(ax, (CX, 17.9), (CX, 19.35))
    box(ax, CX - 3, 19.4, 6, 1.1, "归一化", "neutral")
    arrow(ax, (CX, 20.55), (CX, 21.35))
    box(ax, CX - 4.5, 21.4, 9, 1.1, "乘 W_1，扩到约 4 倍宽", "weight")
    shape(ax, CX + 4.8, 21.95, "[T, 4 × d_model]")
    arrow(ax, (CX, 22.55), (CX, 23.15))
    box(ax, CX - 3, 23.2, 6, 1.0, "逐格激活", "neutral")
    arrow(ax, (CX, 24.25), (CX, 24.85))
    box(ax, CX - 4.5, 24.9, 9, 1.1, "乘 W_2，压回原宽", "weight")
    shape(ax, CX + 4.8, 25.45, "[T, d_model]")
    arrow(ax, (CX, 26.05), (CX, 26.9))
    plus(ax, CX, 27.4)
    label(ax, CX + 0.8, 27.4, "残差相加", ha="left", fontsize=FS_SMALL, color=MUTED)
    label(ax, CX - 5.0, 22.9, "MLP：\n每个位置\n各算各的", ha="right", fontsize=FS_SMALL,
          color=MUTED)

    # 残差旁路 2
    ax.plot([CX, CX - 8.6, CX - 8.6], [18.6, 18.6, 27.4], color=INK, lw=1.5)
    arrow(ax, (CX - 8.6, 27.4), (CX - 0.5, 27.4))
    label(ax, CX - 8.9, 23.6, "原样绕过\nMLP", ha="right", fontsize=FS_SMALL, color=MUTED)

    # 输出
    arrow(ax, (CX, 27.9), (CX, 28.65))
    box(ax, CX - 4, 28.7, 8, 1.2, "第 $\ell$ 层的输出", "data", fontsize=FS_TEXT, bold=True)
    shape(ax, CX + 4.3, 29.3, "[T, d_model]")

    # 右侧：重复 L 层，再进 LM head
    rx = CX + 18.6
    label(ax, rx + 2.6, 26.5, "整个模型", fontsize=FS_NAME, bold=True)
    box(ax, rx, 12.4, 5.2, 1.2, "初始表示 $X^{(0)}$", "data", fontsize=FS_SMALL)
    ys = [14.6, 16.7, 20.2]
    for y, name in zip(ys, ["第 1 层", "第 2 层", "第 L 层"]):
        box(ax, rx, y, 5.2, 1.3, name, "weight", fontsize=FS_SMALL)
    arrow(ax, (rx + 2.6, 13.65), (rx + 2.6, 14.55))
    arrow(ax, (rx + 2.6, 15.95), (rx + 2.6, 16.65))
    dots(ax, rx + 2.6, 19.1, horizontal=False, color=INK, spread=0.36, r=0.09)
    arrow(ax, (rx + 2.6, 21.55), (rx + 2.6, 22.25))
    box(ax, rx, 22.3, 5.2, 1.0, "最终归一化", "neutral", fontsize=FS_SMALL)
    arrow(ax, (rx + 2.6, 23.35), (rx + 2.6, 24.05))
    box(ax, rx, 24.1, 5.2, 1.5, "取最后一行\n→ LM head", "weight", fontsize=FS_SMALL)
    label(ax, rx + 2.6, 11.5, "每层结构相同，\n权重各不相同", fontsize=FS_SMALL, color=MUTED)

    # 图例
    for i, (kind, text) in enumerate((("data", "数据"), ("weight", "含权重的运算"),
                                      ("neutral", "不含权重的运算"))):
        box(ax, rx, 4.4 - i * 1.5, 0.9, 0.8, "", kind, rounded=False, lw=1.2)
        label(ax, rx + 1.3, 4.8 - i * 1.5, text, ha="left", fontsize=FS_SMALL)
    label(ax, rx, 0.2, "蓝字：数据的形状", ha="left", fontsize=FS_SMALL, color=DATA_EDGE)

    finish(fig, ax, OUTPUT, xlim=(-5.4, rx + 6.4), ylim=(-0.4, 31.6))


if __name__ == "__main__":
    main()
