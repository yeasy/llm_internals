"""生成 6.3 节的图：四类 logits 约束各自插在计算图的哪个位置。

正文位置：06_training_techniques/6.3_regularization.md
输出：06_training_techniques/_images/ch06_logit_guards.png

三条泳道对应三个互不相同的落点：
- 注意力内部的前向：QK-Norm 在点积之前，attention logit soft-cap 在 Softmax 之前；
- 最终输出与损失：final logit soft-cap 在末层 Softmax 之前，z-loss 只加一项损失，不改前向；
- 优化器更新之后：QK-clip 不进前向，按前向记下的逐头最大 logit 反过来缩放 W_Q 与 W_K。

配色沿用 _diagram 的约定：蓝=数据、橙=权重、紫=本节要强调的四类约束。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import (ACCENT, FS_SMALL, FS_TEXT, FS_TITLE, MUTED,
                      arrow, box, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("06_training_techniques", "ch06_logit_guards.png")

H = 2.2          # 方框高度
GAP = 1.0        # 方框之间的水平间隔


def guard(ax, x, y, w, text):
    """紫色描边的约束框。"""
    ax.add_patch(plt.Rectangle((x, y), w, H, facecolor="#e9e6f6", edgecolor=ACCENT,
                               linewidth=2.0, zorder=3))
    ax.text(x + w / 2, y + H / 2, text, ha="center", va="center", fontsize=FS_TEXT,
            color=ACCENT, fontweight="bold", linespacing=1.35, zorder=4)
    return x + w


def chain(ax, y, items, x0=0.0):
    x, prev = x0, None
    for text, kind, w in items:
        if prev is not None:
            arrow(ax, (prev, y + H / 2), (x, y + H / 2))
        if kind == "guard":
            right = guard(ax, x, y, w, text)
        else:
            box(ax, x, y, w, H, text, kind, fontsize=FS_TEXT)
            right = x + w
        prev, x = right, right + GAP
    return prev


def lane_title(ax, y, title, note):
    label(ax, 0.0, y, title, ha="left", fontsize=FS_TEXT, bold=True)
    label(ax, 0.0, y - 1.1, note, ha="left", fontsize=FS_SMALL, color=MUTED)


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(9.8, 5.9))

    label(ax, 0.0, 21.6, "四类约束落在三个互不相同的位置（soft-cap 分两个落点）", ha="left",
          fontsize=FS_TITLE, bold=True)

    lane_title(ax, 19.4, "① 注意力内部的前向", "每层、每个头各做一次")
    chain(ax, 15.6, [
        ("q, k", "data", 3.2),
        ("QK-Norm\n先归一化 q、k", "guard", 7.0),
        ("$qk^{\\mathsf{T}}/\\sqrt{d_h}$\nattention logits", "data", 7.4),
        ("attention\nlogit soft-cap", "guard", 6.8),
        ("Softmax", "neutral", 4.4),
    ])

    lane_title(ax, 13.2, "② 最终输出与损失", "每步一次，作用在词表维上")
    chain(ax, 9.4, [
        ("末层输出\n× W_vocab", "weight", 5.6),
        ("logits\n[B, T, V]", "data", 5.6),
        ("final\nlogit soft-cap", "guard", 6.8),
        ("Softmax\n配分函数 Z", "neutral", 5.8),
    ])
    guard(ax, 17.2, 5.0, 7.4, "z-loss\n把 log Z 拉回 0")
    arrow(ax, (23.6, 9.4), (21.8, 7.2), color=ACCENT)
    label(ax, 25.2, 6.1, "不改前向，只多一项损失", ha="left",
          fontsize=FS_SMALL, color=MUTED)

    lane_title(ax, 3.4, "③ 优化器更新之后", "不进前向，直接改权重")
    chain(ax, -0.4, [
        ("本步刚更新的\nW_Q、W_K", "weight", 6.4),
        ("QK-clip\n$\\gamma=\\min(1,\\ \\tau/S_{\\max})$", "guard", 9.4),
        ("W_Q、W_K\n各乘 $\\sqrt{\\gamma}$", "weight", 6.4),
    ])
    label(ax, 25.4, 0.7, "输入信号 $S_{\\max}$ 由泳道 ① 的前向顺带记下：", ha="left",
          fontsize=FS_SMALL, color=MUTED)
    label(ax, 25.4, -0.4, "逐头的最大 attention logit", ha="left",
          fontsize=FS_SMALL, color=MUTED)

    finish(fig, ax, OUTPUT, xlim=(-0.6, 39.0), ylim=(-1.6, 22.6))


if __name__ == "__main__":
    main()
