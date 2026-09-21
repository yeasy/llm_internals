"""生成 12.2 节插图：ELECTRA 的生成器 → 采样 → 判别器数据流。

正文位置：12_encoder_models/12.2_roberta_albert.md
输出：12_encoder_models/_images/ch12_electra_flow.png

四行分别是原始序列、遮盖后送进生成器的序列、采样填回后送进判别器的序列、
判别器的二分类标签。右侧标出两处损失各覆盖多少个位置，与正文 12.2.3 的算例一致。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import (ACCENT, DATA_EDGE, FS_SMALL, FS_TEXT, INK, MUTED,
                      NEW_EDGE, WEIGHT_EDGE, arrow, box, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("12_encoder_models", "ch12_electra_flow.png")

ORIG = ["the", "chef", "cooked", "the", "meal", "."]
MASKED_AT = (2, 3)
SAMPLED = {2: "ate", 3: "the"}   # 位置 3 恰好采样回原词，标签因此是 original

CW, CH, GAP = 1.95, 0.80, 0.14
PITCH = CW + GAP


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(9.4, 5.6))

    n = len(ORIG)
    width = n * PITCH - GAP

    def row(y, texts, kinds, name):
        label(ax, -0.4, y + CH / 2, name, fontsize=FS_SMALL, color=MUTED, ha="right")
        for i, (t, k) in enumerate(zip(texts, kinds)):
            box(ax, i * PITCH, y, CW, CH, t, kind=k, fontsize=FS_SMALL)

    y1, y2, y3, y4 = 6.60, 4.90, 2.30, 0.60

    row(y1, ORIG, ["data"] * n, "原始序列 x")

    masked = [("[MASK]" if i in MASKED_AT else t) for i, t in enumerate(ORIG)]
    kinds2 = ["neutral" if i in MASKED_AT else "data" for i in range(n)]
    row(y2, masked, kinds2, "遮 15%\n→ 生成器输入")

    corrupted = [SAMPLED.get(i, t) for i, t in enumerate(ORIG)]
    kinds3 = ["new" if i in MASKED_AT else "data" for i in range(n)]
    row(y3, corrupted, kinds3, "按分布采样填回\n→ 判别器输入")

    labels = ["replaced" if corrupted[i] != ORIG[i] else "original" for i in range(n)]
    row(y4, labels, ["plain"] * n, "判别器标签")

    # 生成器方框
    gen_y = y2 - 1.28
    box(ax, MASKED_AT[0] * PITCH - 0.72, gen_y, 2 * PITCH - GAP + 1.44, 0.92,
        "小生成器 MLM + 采样", kind="weight", fontsize=FS_SMALL)
    for i in MASKED_AT:
        arrow(ax, (i * PITCH + CW / 2, y2 - 0.04), (i * PITCH + CW / 2, gen_y + 0.92),
              color=WEIGHT_EDGE, lw=1.2)
        arrow(ax, (i * PITCH + CW / 2, gen_y - 0.04), (i * PITCH + CW / 2, y3 + CH),
              color=NEW_EDGE, lw=1.2)
    for i in range(n):
        if i not in MASKED_AT:
            arrow(ax, (i * PITCH + CW / 2, y2 - 0.04), (i * PITCH + CW / 2, y3 + CH),
                  color=DATA_EDGE, lw=1.0)
        arrow(ax, (i * PITCH + CW / 2, y1 - 0.04), (i * PITCH + CW / 2, y2 + CH),
              color=DATA_EDGE, lw=1.0)
        arrow(ax, (i * PITCH + CW / 2, y3 - 0.04), (i * PITCH + CW / 2, y4 + CH),
              color=ACCENT, lw=1.0)

    # 右侧注解
    right = width + 0.45
    label(ax, right, y2 + CH / 2, "生成器损失\n只在 2 个位置", fontsize=FS_SMALL,
          color=WEIGHT_EDGE, ha="left")
    label(ax, right, gen_y + 0.46,
          "从输出分布采样，不取 argmax\n采样是离散的，梯度不回传",
          fontsize=FS_SMALL, color=MUTED, ha="left")
    label(ax, right, y3 + CH / 2, "判别器读全部\n6 个位置", fontsize=FS_SMALL,
          color=NEW_EDGE, ha="left")
    label(ax, right, y4 + CH / 2,
          "判别器损失铺满 6 个位置\n采样恰回原词的位置标 original",
          fontsize=FS_SMALL, color=ACCENT, ha="left")
    label(ax, width / 2, y4 - 0.55,
          r"总损失 $\mathcal{L}=\mathcal{L}_{MLM}(\theta_G)+50\cdot\mathcal{L}_{Disc}(\theta_D)$；"
          "下游只保留判别器",
          fontsize=FS_TEXT, color=INK)
    finish(fig, ax, OUTPUT, xlim=(-5.2, width + 5.4), ylim=(y4 - 1.10, y1 + CH + 0.40))


if __name__ == "__main__":
    main()
