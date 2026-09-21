"""生成 5.2 节插图：一条句子经 80/10/10 改写后，哪些位置计损失。

正文位置：05_pretraining/5.2_masked_lm.md
输出：05_pretraining/_images/ch05_mlm_masking.png

与正文 5.2.3 的算例一致：12 个词元的序列，选中 2 个位置（约 15%），
一个换成 [MASK]、一个保持原样；未选中的 10 个位置照常前向但不计损失。
上半幅是改写前后的输入，下半幅标出损失位置与自回归的对照。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import (ACCENT, DATA_EDGE, FS_SMALL, INK, MASK_EDGE, MUTED, NEW_EDGE,
                      arrow, box, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("05_pretraining", "ch05_mlm_masking.png")

ORIG = ["[CLS]", "the", "man", "went", "to", "the", "store", "to", "buy", "a", "hat", "[SEP]"]
# 选中 2 个位置作为预测目标（索引从 0 起）：6 号换 [MASK]，10 号保持原样。
REPLACED = dict(ORIG_IDX for ORIG_IDX in [(6, "[MASK]")])
KEEP_SAME = {10}
SELECTED = {6, 10}

CW, CH, GAP = 1.62, 0.82, 0.07
FS_CELL = 10.0


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(11.6, 5.2))

    y_orig, y_in, y_out = 3.30, 1.85, 0.30

    def draw(y, texts, kinds):
        centers = []
        for i, (t, k) in enumerate(zip(texts, kinds)):
            cx, _ = box(ax, i * (CW + GAP), y, CW, CH, t, kind=k, fontsize=FS_CELL)
            centers.append(cx)
        return centers

    # 原文
    label(ax, -0.4, y_orig + CH / 2, "原文", fontsize=FS_SMALL, color=MUTED, ha="right")
    kinds = ["new" if i in SELECTED else "data" for i in range(len(ORIG))]
    c0 = draw(y_orig, ORIG, kinds)
    label(ax, c0[6], y_orig + CH + 0.26, "选中·80% 换 [MASK]", fontsize=FS_SMALL,
          color=NEW_EDGE)
    label(ax, c0[10], y_orig + CH + 0.26, "选中·10% 原样", fontsize=FS_SMALL,
          color=NEW_EDGE)

    # 改写后的输入
    label(ax, -0.4, y_in + CH / 2, "送进编码器\n的输入", fontsize=FS_SMALL, color=MUTED,
          ha="right")
    texts = [REPLACED.get(i, ORIG[i]) for i in range(len(ORIG))]
    kinds = ["neutral" if i in REPLACED else ("new" if i in SELECTED else "data")
             for i in range(len(ORIG))]
    c1 = draw(y_in, texts, kinds)
    for i in range(len(ORIG)):
        arrow(ax, (c0[i], y_orig), (c1[i], y_in + CH), color=MUTED, lw=0.9)

    # 输出与损失
    label(ax, -0.4, y_out + CH / 2, "每个位置\n都有输出", fontsize=FS_SMALL, color=MUTED,
          ha="right")
    for i in range(len(ORIG)):
        if i in SELECTED:
            box(ax, i * (CW + GAP), y_out, CW, CH, "计损失", kind="new", fontsize=FS_CELL)
            arrow(ax, (c1[i], y_in), (c1[i], y_out + CH), color=ACCENT, lw=1.2)
        else:
            box(ax, i * (CW + GAP), y_out, CW, CH, "丢弃", kind="plain", fontsize=FS_CELL,
                color=MUTED)

    x_right = len(ORIG) * (CW + GAP) - GAP
    label(ax, x_right + 0.35, y_orig + CH / 2,
          "注意力无因果掩码\n每个位置可读全序列", fontsize=FS_SMALL, color=DATA_EDGE, ha="left")
    label(ax, x_right + 0.35, y_in + CH / 2,
          "12 个位置全部\n参与前向计算", fontsize=FS_SMALL, color=INK, ha="left")
    label(ax, x_right + 0.35, y_out + CH / 2,
          "只有 2 个位置出损失\n自回归同长度出 11 个", fontsize=FS_SMALL, color=ACCENT,
          ha="left")

    finish(fig, ax, OUTPUT, xlim=(-3.6, x_right + 4.6), ylim=(-0.2, y_orig + CH + 0.72))


if __name__ == "__main__":
    main()
