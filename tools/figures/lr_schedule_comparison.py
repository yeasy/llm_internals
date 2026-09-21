"""生成 6.2 节的图：三种学习率调度策略的曲线对比。

正文位置：06_training_techniques/6.2_lr_schedule.md
输出：06_training_techniques/_images/lr_schedule_comparison.png

三条曲线的公式与正文 6.2.1、6.2.4 一致，纵轴统一归一化为「占峰值学习率的比例」，
以便三种调度并排比较：

- 逆平方根：6.2.1 的原始公式，按同一个 warmup 归一化到自己的峰值；
- 余弦退火：衰减到峰值的 10%，取的是 GPT-3 的口径（该文说明余弦衰减到初值的 10%）；
- WSD：预热后保持峰值，最后 10% 的步数用余弦降到峰值的 10%，
  衰减段占比取 MiniCPM 的「10% 步数足够」结论。

图里不画具体的学习率数值：各模型的峰值差两个数量级，正文的表格给真实配置。
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt

from _diagram import ACCENT, DATA_EDGE, INK, MUTED, NEW_EDGE, WEIGHT_EDGE
from _style import image_path, use_cjk_font

OUTPUT = image_path("06_training_techniques", "lr_schedule_comparison.png")

TOTAL_STEPS = 20_000
WARMUP_STEPS = 2_000
FLOOR = 0.10           # 衰减终点占峰值的比例
DECAY_FRAC = 0.10      # WSD 衰减段占总步数的比例


def inverse_sqrt(step: int) -> float:
    """6.2.1 的原始调度，除以它自己的峰值后得到相对值。"""
    raw = min(step ** -0.5, step * WARMUP_STEPS ** -1.5)
    return raw / WARMUP_STEPS ** -0.5


def cosine(step: int) -> float:
    if step <= WARMUP_STEPS:
        return step / WARMUP_STEPS
    p = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
    return FLOOR + 0.5 * (1 - FLOOR) * (1 + math.cos(math.pi * p))


def wsd(step: int) -> float:
    decay_start = int(TOTAL_STEPS * (1 - DECAY_FRAC))
    if step <= WARMUP_STEPS:
        return step / WARMUP_STEPS
    if step < decay_start:
        return 1.0
    p = (step - decay_start) / (TOTAL_STEPS - decay_start)
    return FLOOR + 0.5 * (1 - FLOOR) * (1 + math.cos(math.pi * p))


def main() -> None:
    use_cjk_font()
    steps = list(range(1, TOTAL_STEPS + 1))
    fig, ax = plt.subplots(figsize=(9.2, 4.4))

    ax.plot(steps, [inverse_sqrt(s) for s in steps], color=MUTED, linewidth=1.8,
            label="逆平方根（原始 Transformer）")
    ax.plot(steps, [cosine(s) for s in steps], color=DATA_EDGE, linewidth=1.8,
            label="余弦退火，降到峰值的 10%")
    ax.plot(steps, [wsd(s) for s in steps], color=NEW_EDGE, linewidth=1.8,
            label="WSD：末 10% 步数衰减")

    ax.axvline(WARMUP_STEPS, color=WEIGHT_EDGE, linestyle=":", linewidth=1.4)
    ax.annotate("预热结束", xy=(WARMUP_STEPS, 1.02), xytext=(WARMUP_STEPS + 600, 1.10),
                fontsize=10, color=WEIGHT_EDGE,
                arrowprops=dict(arrowstyle="-", color=WEIGHT_EDGE, lw=1.0))
    ax.axhline(FLOOR, color=ACCENT, linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(TOTAL_STEPS * 0.02, FLOOR + 0.03, "峰值的 10%", fontsize=10, color=ACCENT)

    decay_start = int(TOTAL_STEPS * (1 - DECAY_FRAC))
    ax.annotate("WSD 的衰减段\n（损失在这里骤降）",
                xy=(19_150, 0.50), xytext=(TOTAL_STEPS * 0.60, 0.62),
                fontsize=10, color=NEW_EDGE, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=NEW_EDGE, lw=1.1))

    ax.set_xlabel("训练步数（示意：共 2 万步，预热 2 千步）", fontsize=11, color=INK)
    ax.set_ylabel("学习率 / 峰值学习率", fontsize=11, color=INK)
    ax.set_ylim(0, 1.22)
    ax.set_xlim(0, TOTAL_STEPS)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150, facecolor="white")
    print(f"已写入 {OUTPUT}")


if __name__ == "__main__":
    main()
