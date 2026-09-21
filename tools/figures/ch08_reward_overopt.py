"""生成 8.2 节的示意图：代理奖励与真实奖励随 KL 距离分道。

正文位置：08_alignment/8.2_rlhf.md
输出：08_alignment/_images/ch08_reward_overopt.png

横轴是 d = sqrt(KL(pi || pi_init))，纵轴是奖励。真实奖励用 Gao 等人给出的
函数形式 R(d) = d(alpha - beta log d) 画出形状，代理奖励按论文观察到的
"大致随 sqrt(KL) 线性增长"画成直线。alpha、beta 取教学值，只表形状不表量级。
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt

from _diagram import ACCENT, DATA_EDGE, FS_NAME, FS_SMALL, FS_TEXT, INK, MUTED, NEW_EDGE
from _style import image_path, use_cjk_font

OUTPUT = image_path("08_alignment", "ch08_reward_overopt.png")

ALPHA, BETA = 1.0, 0.5          # 教学取值；论文的拟合系数随奖励模型规模变化


def gold(d):
    return d * (ALPHA - BETA * math.log(d))


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(8.0, 4.4))

    ds = [0.02 + 0.02 * i for i in range(400)]
    ax.plot(ds, [gold(d) for d in ds], color=NEW_EDGE, lw=2.4, label="真实目标（金奖励）")
    ax.plot(ds, [0.62 * d for d in ds], color=DATA_EDGE, lw=2.4, ls="--",
            label="代理目标（奖励模型给的分）")

    dstar = math.exp(ALPHA / BETA - 1)
    ax.plot([dstar, dstar], [0, gold(dstar)], color=ACCENT, lw=1.4, ls=":")
    ax.plot([dstar], [gold(dstar)], "o", color=ACCENT, ms=7, zorder=5)
    ax.annotate("峰值：d* = exp(a/b - 1)\n此处 KL = d*$^2$",
                xy=(dstar, gold(dstar)), xytext=(dstar + 0.9, gold(dstar) + 0.55),
                fontsize=FS_SMALL, color=ACCENT,
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.2))

    ax.annotate("越过峰值之后：\n代理分仍在涨，真实质量在掉",
                xy=(6.55, gold(6.55)), xytext=(3.1, 0.22),
                fontsize=FS_SMALL, color=MUTED,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))

    ax.set_xlabel("d = $\\sqrt{\\mathrm{KL}(\\pi\\,\\|\\,\\pi_{init})}$", fontsize=FS_TEXT)
    ax.set_ylabel("奖励", fontsize=FS_TEXT)
    ax.set_xlim(0, 8.2)
    ax.set_ylim(0, 4.3)
    ax.set_title("优化得越远，代理与真实越分道", fontsize=FS_NAME, color=INK, pad=10)
    ax.legend(fontsize=FS_SMALL, loc="upper left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=FS_SMALL)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"已写入 {OUTPUT}")


if __name__ == "__main__":
    main()
