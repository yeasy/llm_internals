"""生成 6.4 节的图：训练显存里模型状态与激活各占多少。

正文位置：06_training_techniques/6.4_batch_sequence.md
输出：06_training_techniques/_images/ch06_memory_stack.png

算例是 GPT-3 6.7B（L = 32、h = 4096、a = 32，micro-batch 1），与 3.8.6 的表 3-11 同一配置。
模型状态按混合精度 AdamW 的 16 字节/参数：16 x 6.7e9 = 107 GB，与序列长度无关。
激活按 Korthikanti 等人的每层 sbh(34 + 5as/h) 字节：
- 朴素：两项都留；
- FlashAttention：不物化 [s, s] 的分数矩阵，去掉 5as/h 这一项；
- 全量重计算：每层只留输入 2sbh。
纵轴取对数，因为三档之间差两个数量级。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import ACCENT, DATA_EDGE, INK, MUTED, NEW_EDGE, WEIGHT_EDGE
from _style import image_path, use_cjk_font

OUTPUT = image_path("06_training_techniques", "ch06_memory_stack.png")

GB = 1e9
N_PARAM, LAYERS, H, A = 6.7e9, 32, 4096, 32
MODEL_STATE = 16 * N_PARAM / GB
CARD = 80.0        # 一张 80 GB 的 H100，见 7.1 节的集群设定

SEQS = (2048, 8192)
STRATEGIES = (
    ("朴素实现\n两项都留", DATA_EDGE),
    ("FlashAttention\n去掉二次项", NEW_EDGE),
    ("全量重计算\n每层只留输入", WEIGHT_EDGE),
)


def activation_gb(s: int, mode: int) -> float:
    linear = 34 * s * H
    quad = 5 * A * s * s
    if mode == 0:
        per_layer = linear + quad
    elif mode == 1:
        per_layer = linear
    else:
        per_layer = 2 * s * H
    return per_layer * LAYERS / GB


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(8.8, 4.8))

    width, xs, labels, colors, values = 0.62, [], [], [], []
    for gi, s in enumerate(SEQS):
        for mi, (name, color) in enumerate(STRATEGIES):
            xs.append(gi * 3.6 + mi)
            labels.append(name)
            colors.append(color)
            values.append(activation_gb(s, mi))

    bars = ax.bar(xs, values, width=width, color=colors, edgecolor=INK, linewidth=0.6)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.15,
                f"{v:,.1f} GB", ha="center", fontsize=10, color=INK)

    ax.axhline(MODEL_STATE, color=ACCENT, linestyle="--", linewidth=1.5)
    ax.text(-0.42, MODEL_STATE * 1.13,
            f"模型状态 {MODEL_STATE:.0f} GB（16 字节/参数，与序列长度无关）",
            fontsize=10, color=ACCENT)
    ax.axhline(CARD, color=MUTED, linestyle=":", linewidth=1.3)
    ax.text(-0.42, CARD * 0.72, "单卡 80 GB", fontsize=10, color=MUTED)

    ax.set_yscale("log")
    ax.set_ylim(0.3, 1600)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("激活显存（GB，对数轴）", fontsize=11, color=INK)
    for gi, s in enumerate(SEQS):
        ax.text(gi * 3.6 + 1, 900, f"序列长度 s = {s:,}", ha="center",
                fontsize=12, color=INK, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150, facecolor="white")
    print(f"已写入 {OUTPUT}")
    print(f"模型状态 {MODEL_STATE:.1f} GB")
    for x, lab, v in zip(xs, labels, values):
        print(f"  x={x} {lab.replace(chr(10), ' ')}: {v:,.2f} GB")


if __name__ == "__main__":
    main()
