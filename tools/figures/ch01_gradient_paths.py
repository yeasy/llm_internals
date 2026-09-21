"""生成 1.2 节插图：梯度沿 h 通路与沿 c 通路的两种衰减。

正文位置：01_introduction/1.2_rnn_cnn_limits.md
输出：01_introduction/_images/ch01_gradient_paths.png

左：一步回传的雅可比。h 通路每步要乘 diag(tanh') W_h，c 通路每步只乘 diag(f_t)。
右：把两种通路的连乘画成对数纵轴的衰减曲线，代入正文用到的四个系数。
    0.5 一条对应 Jozefowicz 等 2015 指出的默认初始化：遗忘门初值约 0.5。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _diagram import (ACCENT, DATA_EDGE, FS_NAME, FS_SMALL, FS_TEXT, INK, MUTED,
                      NEW_EDGE, WEIGHT_EDGE, arrow, box, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("01_introduction", "ch01_gradient_paths.png")


def draw_paths(ax) -> None:
    """左图：相邻两个时间步之间的两条通路。"""
    ax.set_xlim(-0.2, 9.4)
    ax.set_ylim(-1.6, 8.3)
    ax.set_aspect("equal")
    ax.axis("off")

    y_c, y_h = 4.6, 0.8
    w, h = 2.2, 1.0
    x0, x1 = 0.4, 6.4

    for x, name in ((x0, "t-1"), (x1, "t")):
        box(ax, x, y_c, w, h, f"$c_{{{name}}}$", kind="new", fontsize=FS_TEXT)
        box(ax, x, y_h, w, h, f"$h_{{{name}}}$", kind="data", fontsize=FS_TEXT)

    # c 通路：只乘一个门向量，逐元素。
    arrow(ax, (x0 + w, y_c + h / 2), (x1, y_c + h / 2), color=NEW_EDGE, lw=2.4)
    label(ax, (x0 + w + x1) / 2, y_c + h + 0.95,
          "加法通路\n$\\partial c_t/\\partial c_{t-1}=\\mathrm{diag}(f_t)$\n"
          "逐元素相乘，不过激活",
          fontsize=FS_SMALL, color=NEW_EDGE, linespacing=1.5)

    # h 通路：乘权重矩阵再过激活。
    arrow(ax, (x0 + w, y_h + h / 2), (x1, y_h + h / 2), color=DATA_EDGE, lw=2.4)
    label(ax, (x0 + w + x1) / 2, y_h - 0.95,
          "乘法通路\n$\\mathrm{diag}(\\tanh')\\,W_h$\n满矩阵，再过激活",
          fontsize=FS_SMALL, color=DATA_EDGE, linespacing=1.5)

    # 两条通路的耦合：h 由 c 读出，门由 h 算出。
    arrow(ax, (x1 + 0.55, y_c), (x1 + 0.55, y_h + h), color=MUTED, lw=1.2)
    label(ax, x1 + 0.78, (y_c + y_h + h) / 2, "$h_t=o_t\\odot\\tanh(c_t)$",
          fontsize=FS_SMALL, color=MUTED, ha="left")
    arrow(ax, (x0 + w * 0.6, y_h + h), (x0 + w * 0.6, y_c), color=MUTED, lw=1.2)
    label(ax, x0 + w * 0.6 - 0.25, (y_c + y_h + h) / 2,
          "三个门都由\n$h_{t-1}$ 与 $x_t$ 算出", fontsize=FS_SMALL, color=MUTED,
          ha="right", linespacing=1.45)

    label(ax, 4.6, 8.0, "一步回传各要乘什么", fontsize=FS_NAME, color=INK, bold=True)


def draw_decay(ax) -> None:
    """右图：连乘若干次之后还剩多少。横轴是连乘次数，与正文 1.2.3、1.2.4 同口径。"""
    steps = np.arange(1, 1001)
    curves = [
        (0.5, WEIGHT_EDGE, "-", "c 通路，f = 0.5（偏置按默认初始化）"),
        (0.9, DATA_EDGE, "-", "h 通路，每步收缩 0.9"),
        (0.99, NEW_EDGE, "-", "c 通路，f = 0.99"),
        (0.999, ACCENT, "--", "c 通路，f = 0.999"),
    ]
    for rho, color, ls, text in curves:
        ax.plot(steps, rho ** steps, color=color, lw=2.2, ls=ls, label=text)

    ax.axhline(1e-7, color=MUTED, lw=1.0, ls=":")
    ax.text(1.3, 1.6e-7, "1e-7", fontsize=FS_SMALL, color=MUTED, ha="left", va="bottom")

    for rho, color, text, xy_text in (
        (0.9, DATA_EDGE, "连乘 99 次剩 3.0e-5", (2.2, 2e-6)),
        (0.99, NEW_EDGE, "连乘 99 次剩 0.37", (1.25, 0.02)),
    ):
        ax.scatter([99], [rho ** 99], s=58, color=color, edgecolor="white",
                   linewidth=1.3, zorder=5)
        ax.annotate(text, xy=(99, rho ** 99), xytext=xy_text, fontsize=FS_SMALL,
                    color=color, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=color, lw=1.0))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1000)
    ax.set_ylim(1e-12, 6)
    ax.set_xlabel("连乘次数", fontsize=FS_TEXT, color=INK)
    ax.set_ylabel("剩余的梯度倍率", fontsize=FS_TEXT, color=INK)
    ax.set_title("连乘若干次之后还剩多少", fontsize=FS_NAME, color=INK)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1000"])
    ax.set_yticks([1, 1e-3, 1e-6, 1e-9, 1e-12])
    ax.set_yticklabels(["1", "1e-3", "1e-6", "1e-9", "1e-12"])
    ax.legend(fontsize=FS_SMALL, frameon=False, loc="lower left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=FS_SMALL)


def main() -> None:
    use_cjk_font()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.2),
                                   gridspec_kw={"width_ratios": [1.0, 1.0]})
    draw_paths(ax1)
    draw_decay(ax2)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"已写入 {OUTPUT}")


if __name__ == "__main__":
    main()
