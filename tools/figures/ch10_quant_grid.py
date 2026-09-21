"""生成 10.4 节插图：同一行权重在三种量化方案下落到哪些格点上。

正文位置：10_inference_optimization/10.4_quantization.md
输出：10_inference_optimization/_images/ch10_quant_grid.png

权重取正文 10.4.1 的手算例 [0.1, -0.2, 0.3, 8.0]，对称 absmax 量化。
只画 0 附近的窗口 [-0.45, 0.45]：离群值 8.0 在窗口外，但它决定了整行共用的步长。
蓝 = 原始值，青绿 = 量化后还原的值，灰色竖线 = 该方案可表示的格点。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import ACCENT, DATA_EDGE, FS_TEXT, INK, MASK_EDGE, MUTED, NEW_EDGE
from _style import image_path, use_cjk_font

OUTPUT = image_path("10_inference_optimization", "ch10_quant_grid.png")
FS = 10
WEIGHTS = [0.1, -0.2, 0.3]
LO, HI = -0.45, 0.45


def quantize(values, amax, bits):
    qmax = 2 ** (bits - 1) - 1
    step = amax / qmax
    return step, [max(-qmax, min(qmax, round(v / step))) * step for v in values]


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    rows = [
        ("INT8，整行一个缩放因子", 8.0, 8, "步长 8 ÷ 127 = 0.063"),
        ("INT4，整行一个缩放因子", 8.0, 4, "步长 8 ÷ 7 = 1.143，窗口内只有 0 这一个格点"),
        ("INT4，前三个数单独成组", 0.3, 4, "步长 0.3 ÷ 7 = 0.043"),
    ]
    for k, (name, amax, bits, note) in enumerate(rows):
        y = 2 - k
        step, deq = quantize(WEIGHTS, amax, bits)
        ax.plot([LO, HI], [y, y], color="#b9b8b2", lw=1.0, zorder=1)
        n = min(int(HI / step), 2 ** (bits - 1) - 1)
        for g in range(-n, n + 1):
            ax.plot([g * step, g * step], [y - 0.13, y + 0.13], color=MASK_EDGE, lw=1.0, zorder=2)
        for w, d in zip(WEIGHTS, deq):
            ax.annotate("", xy=(d, y - 0.02), xytext=(w, y + 0.3),
                        arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.2), zorder=3)
            ax.scatter([w], [y + 0.3], s=60, color=DATA_EDGE, edgecolor="white", linewidth=1.2, zorder=4)
            ax.scatter([d], [y], s=70, color=NEW_EDGE, edgecolor="white", linewidth=1.2, zorder=5)
        ax.text(LO, y + 0.52, name, fontsize=FS_TEXT, color=INK, ha="left", va="center", fontweight="bold")
        ax.text(HI, y + 0.52, note, fontsize=FS, color=MUTED, ha="right", va="center")
        ax.text(HI + 0.02, y, "还原值\n" + "、".join(f"{d:.3f}".rstrip("0").rstrip(".") if d else "0" for d in deq),
                fontsize=FS, color=NEW_EDGE, ha="left", va="center", linespacing=1.4)

    ax.scatter([LO + 0.01], [-0.62], s=60, color=DATA_EDGE, edgecolor="white", linewidth=1.2)
    ax.text(LO + 0.035, -0.62, "原始值 0.1、-0.2、0.3", fontsize=FS, color=INK, va="center")
    ax.scatter([-0.08], [-0.62], s=70, color=NEW_EDGE, edgecolor="white", linewidth=1.2)
    ax.text(-0.055, -0.62, "量化后还原的值", fontsize=FS, color=INK, va="center")
    ax.plot([0.17, 0.17], [-0.74, -0.5], color=MASK_EDGE, lw=1.0)
    ax.text(0.19, -0.62, "可表示的格点", fontsize=FS, color=INK, va="center")

    ax.set_xlim(LO - 0.02, HI + 0.26)
    ax.set_ylim(-0.95, 2.85)
    ax.set_yticks([])
    ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
    ax.set_xticklabels(["-0.4", "-0.2", "0", "0.2", "0.4"])
    ax.tick_params(labelsize=FS, colors=MUTED, length=0)
    ax.set_xlabel("权重取值（只画 0 附近；同一行里的离群值 8.0 在窗口之外）", fontsize=FS_TEXT, color=INK)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"已写入 {OUTPUT}")


if __name__ == "__main__":
    main()
