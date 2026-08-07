"""生成图 6-1：三种学习率调度策略的曲线对比。

正文位置：06_training_techniques/6.2_lr_schedule.md
输出：06_training_techniques/_images/lr_schedule_comparison.png

三条曲线分别是 6.2.1 的原始逆平方根调度，以及 6.2.4 描述的余弦退火与
WSD 三阶段调度；公式与正文一致。
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import torch

from _style import image_path, use_cjk_font

OUTPUT = image_path("06_training_techniques", "lr_schedule_comparison.png")

TOTAL_STEPS = 20000
WARMUP_STEPS = 2000
D_MODEL = 512
PEAK_LR = 1e-3


def inverse_sqrt(steps: torch.Tensor) -> torch.Tensor:
    """原始 Transformer 的逆平方根调度（6.2.1 的公式）。"""
    return D_MODEL ** (-0.5) * torch.minimum(
        steps ** (-0.5),
        steps * WARMUP_STEPS ** (-1.5)
    )


def cosine_with_warmup() -> torch.Tensor:
    """余弦退火（带线性预热）。"""
    lr = torch.zeros(TOTAL_STEPS)
    for s in range(TOTAL_STEPS):
        if s < WARMUP_STEPS:
            lr[s] = PEAK_LR * (s + 1) / WARMUP_STEPS
        else:
            progress = (s - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
            lr[s] = PEAK_LR * 0.5 * (1 + math.cos(math.pi * progress))
    return lr


def wsd() -> torch.Tensor:
    """WSD 三阶段：预热 → 恒定峰值 → 末段快速衰减。"""
    # 稳定阶段持续到总步数的 80% 处（最后 20% 用于衰减）
    decay_start = int(TOTAL_STEPS * 0.8)
    lr = torch.zeros(TOTAL_STEPS)
    for s in range(TOTAL_STEPS):
        if s < WARMUP_STEPS:
            lr[s] = PEAK_LR * (s + 1) / WARMUP_STEPS
        elif s < decay_start:
            lr[s] = PEAK_LR
        else:
            progress = (s - decay_start) / (TOTAL_STEPS - decay_start)
            lr[s] = PEAK_LR * 0.5 * (1 + math.cos(math.pi * progress))
    return lr


def main() -> None:
    use_cjk_font()
    steps = torch.arange(1, TOTAL_STEPS + 1).float()

    plt.figure(figsize=(10, 5))
    plt.plot(steps.numpy(), inverse_sqrt(steps).numpy(),
             label="逆平方根（原始 Transformer）", linewidth=1.5)
    plt.plot(steps.numpy(), cosine_with_warmup().numpy(),
             label="余弦退火", linewidth=1.5)
    plt.plot(steps.numpy(), wsd().numpy(),
             label="WSD 三阶段", linewidth=1.5)
    plt.axvline(x=WARMUP_STEPS, color="gray", linestyle=":", alpha=0.5,
                label=f"预热结束（step {WARMUP_STEPS}）")
    plt.xlabel("训练步数")
    plt.ylabel("学习率")
    plt.title("三种学习率调度策略对比")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=150)
    print(f"已写入 {OUTPUT}")


if __name__ == "__main__":
    main()
