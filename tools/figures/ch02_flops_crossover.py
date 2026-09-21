"""生成 2.5 节插图：一层 Transformer 中，线性项与平方项各占多少。

正文位置：02_attention/2.5_complexity_limits.md
输出：02_attention/_images/ch02_flops_crossover.png

口径与 3.8.7 一致：一层约 24nd^2（与权重相乘）加 4n^2 d（双向的打分与读取），
因果掩码下平方项折半为 2n^2 d。取 d = 4096（GPT-3 6.7B 与 Llama 3 8B 的宽度），
两条交点分别在 n = 6d = 24,576 与 n = 12d = 49,152。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _diagram import ACCENT, DATA_EDGE, FS_SMALL, FS_TEXT, INK, MUTED, WEIGHT_EDGE
from _style import image_path, use_cjk_font

OUTPUT = image_path("02_attention", "ch02_flops_crossover.png")

D = 4096
MARKS = (2048, 8192, 32768, 131072)


def main() -> None:
    use_cjk_font()
    n = np.logspace(np.log10(512), np.log10(1 << 20), 400)
    linear = 24 * n * D * D
    quad_bi = 4 * n * n * D
    quad_causal = 2 * n * n * D

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(n, linear / 1e12, color=WEIGHT_EDGE, lw=2.2, label="与权重相乘：$24nd^2$")
    ax.plot(n, quad_bi / 1e12, color=DATA_EDGE, lw=2.2, label="打分与读取，双向：$4n^2d$")
    ax.plot(n, quad_causal / 1e12, color=DATA_EDGE, lw=2.0, linestyle=(0, (5, 2)),
            label="打分与读取，因果：$2n^2d$")

    crossings = ((6 * D, "双向的交点\nn = 6d = 24,576", 0.11, 5.5, "left"),
                 (12 * D, "因果的交点\nn = 12d = 49,152", 1.5, 0.12, "left"))
    for x, text, fx, fy, ha in crossings:
        y = 24 * x * D * D / 1e12
        ax.scatter([x], [y], s=70, color=ACCENT, zorder=5, edgecolor="white", linewidth=1.4)
        ax.annotate(text, xy=(x, y), xytext=(x * fx, y * fy), fontsize=FS_SMALL, color=ACCENT,
                    ha=ha, arrowprops=dict(arrowstyle="-", color=ACCENT, lw=1.0))

    for x in MARKS:
        ax.axvline(x, color="#e4e4e1", lw=1.0, zorder=0)
    ax.set_xticks(list(MARKS))
    ax.set_xticklabels(["2K", "8K", "32K", "128K"])
    ax.set_ylim(0.15, 2e7)
    ax.set_yticks([1e0, 1e2, 1e4, 1e6])
    ax.set_yticklabels(["1", "$10^2$", "$10^4$", "$10^6$"])
    ax.set_xlabel("序列长度 n（词元）", fontsize=FS_TEXT, color=INK)
    ax.set_ylabel("一层的计算量（TFLOPs）", fontsize=FS_TEXT, color=INK)
    ax.set_title("d = 4096 时，平方项在哪里追上线性项", fontsize=FS_TEXT, color=INK, pad=10)
    ax.tick_params(labelsize=FS_SMALL, colors=MUTED)
    ax.grid(True, which="major", axis="y", color="#e4e4e1", lw=0.8)
    ax.minorticks_off()
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=FS_SMALL, loc="upper left", frameon=False)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="white")
    ratios = {int(x): round(4 * x * x * D / (24 * x * D * D) * 100, 1) for x in MARKS}
    print(f"已写入 {OUTPUT}  双向平方项占线性项百分比={ratios}")


if __name__ == "__main__":
    main()
