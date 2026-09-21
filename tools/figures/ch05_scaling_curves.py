"""生成 5.4 节插图：幂律在 log-log 下何时是直线，以及 IsoFLOP 剖面。

正文位置：05_pretraining/5.4_data_scaling.md
输出：05_pretraining/_images/ch05_scaling_curves.png

左图用 Chinchilla 论文正文公布的一组系数，右图用 arXiv 2404.10102 的重拟合系数
（原因见第 28-29 行的注释）。

左：固定 D = 1.4T，画 L(N) 与 A / N^0.34。固定 D 时 B / D^0.28 也是常数，
    所以减去 E 还不够直；把全部与 N 无关的项都减掉，才得到斜率 -0.34 的直线。
右：把 D = C / (6N) 代回去，对三个算力预算各画一条 IsoFLOP 剖面，
    标出抛物线最低点，即该预算下的计算最优配比。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _diagram import (ACCENT, DATA_EDGE, FS_NAME, FS_SMALL, FS_TEXT, INK, MUTED,
                      NEW_EDGE, WEIGHT_EDGE)
from _style import image_path, use_cjk_font

OUTPUT = image_path("05_pretraining", "ch05_scaling_curves.png")

# 左图：Chinchilla 论文正文公布的一组系数。
E, A, B, ALPHA, BETA = 1.69, 406.4, 410.7, 0.34, 0.28
# 右图：arXiv 2404.10102 复现该文 Approach 3 后重拟合的一组系数。用它是因为
# 正文公布的那一组已知偏差较大，反解出的最优配比与论文自己用的 20 词元/参数不符。
E2, A2, B2, ALPHA2, BETA2 = 1.8172, 482.01, 2085.43, 0.3478, 0.3658


def loss(n, d):
    return E + A / n ** ALPHA + B / d ** BETA


def loss_refit(n, d):
    return E2 + A2 / n ** ALPHA2 + B2 / d ** BETA2


def main() -> None:
    use_cjk_font()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))

    # ---------------- 左：L 弯、L - E 直 ----------------
    n = np.logspace(7, 12, 300)
    d_fixed = 1.4e12
    total = loss(n, d_fixed)
    data_term = B / d_fixed ** BETA
    ax1.plot(n, total, color=DATA_EDGE, lw=2.4, label="L（含 E 与数据项）")
    ax1.plot(n, total - E, color=NEW_EDGE, lw=2.2, ls="-.",
             label="L - E（仍含数据项，弯）")
    ax1.plot(n, A / n ** ALPHA, color=ACCENT, lw=2.4, ls="--",
             label="A / N^0.34（斜率 -0.34，直）")
    ax1.axhline(E, color=MUTED, lw=1.0, ls=":")
    ax1.text(1.5e9, E * 0.92, "E = 1.69：拟合下限，不是理论极限",
             fontsize=FS_SMALL, color=MUTED, ha="left", va="top")
    ax1.plot([5e10, 1.2e12], [data_term, data_term], color=NEW_EDGE, lw=1.0, ls=":")
    ax1.text(1.4e7, 24,
             f"D 固定时 B / D^0.28 = {data_term:.3f} 也是常数，\n"
             f"L - E 因此压平到 {data_term:.3f}，并不是直线",
             fontsize=FS_SMALL, color=NEW_EDGE, ha="left", va="top",
             linespacing=1.45)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("参数量 N", fontsize=FS_TEXT, color=INK)
    ax1.set_ylabel("损失（每词元 nat）", fontsize=FS_TEXT, color=INK)
    ax1.set_title("同一组数据，画法不同", fontsize=FS_NAME, color=INK)
    ax1.legend(fontsize=FS_SMALL, frameon=False, loc="lower left")
    ax1.set_ylim(0.03, 30)
    ax1.set_yticks([0.03, 0.1, 0.3, 1, 3, 10])
    ax1.set_yticklabels(["0.03", "0.1", "0.3", "1", "3", "10"])
    ax1.set_xticks([1e7, 1e8, 1e9, 1e10, 1e11, 1e12])
    ax1.set_xticklabels(["1e7", "1e8", "1e9", "1e10", "1e11", "1e12"])
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)
    ax1.tick_params(labelsize=FS_SMALL)

    # ---------------- 右：IsoFLOP 剖面 ----------------
    budgets = [(1e21, DATA_EDGE, (9e8, 2.60)), (1e23, NEW_EDGE, (1.3e11, 2.30)),
               (1e25, WEIGHT_EDGE, (4e9, 1.87))]
    for c, color, tpos in budgets:
        nn = np.logspace(8, 12.4, 2000)
        dd = c / (6 * nn)
        ll = loss_refit(nn, dd)
        ax2.plot(nn, ll, color=color, lw=2.2)
        k = int(np.argmin(ll))
        ax2.scatter([nn[k]], [ll[k]], s=70, color=color, edgecolor="white",
                    linewidth=1.4, zorder=5)
        exp = int(np.log10(c))
        ax2.text(tpos[0], tpos[1], f"C = 1e{exp}\n{dd[k] / nn[k]:.0f} 词元/参数",
                 fontsize=FS_SMALL, color=color, ha="center", va="center",
                 linespacing=1.35)
    ax2.set_xscale("log")
    ax2.set_xlabel("参数量 N（数据量由 D = C / 6N 定）", fontsize=FS_TEXT, color=INK)
    ax2.set_ylabel("损失（每词元 nat）", fontsize=FS_TEXT, color=INK)
    ax2.set_title("固定算力时，最优点是抛物线的底", fontsize=FS_NAME, color=INK)
    ax2.set_ylim(1.75, 3.6)
    ax2.set_xticks([1e8, 1e9, 1e10, 1e11, 1e12])
    ax2.set_xticklabels(["1e8", "1e9", "1e10", "1e11", "1e12"])
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)
    ax2.tick_params(labelsize=FS_SMALL)

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"已写入 {OUTPUT}")


if __name__ == "__main__":
    main()
