"""生成 5.1 节插图：一条序列怎样错一位变成 T-1 个训练样本。

正文位置：05_pretraining/5.1_autoregressive.md
输出：05_pretraining/_images/ch05_shift_labels.png

口径沿用 3.8 节的教学序列：`〈user〉 2 + 3 = 〈assistant〉 5 。 EOS` 共 9 个位置。
输入取前 8 个位置，标签取后 8 个位置；位置 6、7、8 的损失数值取自 3.8 节
表 3-7 与 3.8.5 两轮 Decode 给出的概率（0.462、0.477、0.313）。
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt

from _diagram import (ACCENT, DATA_EDGE, DATA_FACE, FS_NAME, FS_SMALL, FS_TEXT, INK,
                      MASK_EDGE, MASK_FACE, MUTED, NEW_EDGE, arrow, box, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("05_pretraining", "ch05_shift_labels.png")

TOKENS = ["〈user〉", "2", "+", "3", "=", "〈assistant〉", "5", "。", "EOS"]
PROBS = {6: 0.462, 7: 0.477, 8: 0.313}

CW, CH, GAP = 2.2, 0.80, 0.07
FS_CELL = 10.5


def row(ax, y, cells, *, kind="data", edge=None, fontsize=FS_CELL, start=0):
    centers = []
    for i, text in enumerate(cells):
        x = (start + i) * (CW + GAP)
        if text is None:
            centers.append(None)
            continue
        cx, cy = box(ax, x, y, CW, CH, text, kind=kind, fontsize=fontsize)
        centers.append((cx, cy))
    if edge:
        pass
    return centers


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(11.6, 5.6))

    y_seq, y_in, y_lab, y_loss = 4.35, 2.75, 1.45, 0.05

    # 第一行：完整序列
    label(ax, -0.35, y_seq + CH / 2, "整条序列\n（9 个位置）", fontsize=FS_SMALL,
          color=MUTED, ha="right")
    seq = row(ax, y_seq, TOKENS)
    for i in range(len(TOKENS)):
        label(ax, seq[i][0], y_seq + CH + 0.22, str(i + 1), fontsize=FS_SMALL, color=MUTED)

    # 第二行：输入 = 前 8 个位置
    label(ax, -0.35, y_in + CH / 2, "输入\ninput_ids[:, :-1]", fontsize=FS_SMALL,
          color=MUTED, ha="right")
    inp = row(ax, y_in, TOKENS[:-1])

    # 第三行：标签 = 后 8 个位置，左对齐到同一列
    label(ax, -0.35, y_lab + CH / 2, "标签\ninput_ids[:, 1:]", fontsize=FS_SMALL,
          color=MUTED, ha="right")
    lab = row(ax, y_lab, TOKENS[1:], kind="new")

    # 输入与标签之间的“预测”箭头
    for i in range(8):
        arrow(ax, (inp[i][0], y_in), (lab[i][0], y_lab + CH), color=ACCENT, lw=1.1)

    # 第四行：每个位置一个损失
    label(ax, -0.35, y_loss + CH / 2, "本位置的损失\n-ln p(标签)", fontsize=FS_SMALL,
          color=MUTED, ha="right")
    for i in range(8):
        pos = i + 1
        if pos in PROBS:
            text = f"{-math.log(PROBS[pos]):.3f}"
            kind = "data"
        else:
            text = "?"
            kind = "neutral"
        box(ax, i * (CW + GAP), y_loss, CW, CH, text, kind=kind, fontsize=FS_SMALL)
        arrow(ax, (lab[i][0], y_lab), (lab[i][0], y_loss + CH), color=MUTED, lw=1.0)

    x_right = 9 * (CW + GAP) - GAP
    label(ax, x_right + 0.4, y_loss + CH / 2,
          "三个已算出的位置\n平均 0.891",
          fontsize=FS_SMALL, color=INK, ha="left")
    label(ax, x_right + 0.4, y_lab + CH / 2,
          "每个位置各出\n一个标签",
          fontsize=FS_SMALL, color=NEW_EDGE, ha="left")
    label(ax, x_right + 0.4, y_in + CH / 2,
          "一次前向\n同时算 8 个位置",
          fontsize=FS_SMALL, color=DATA_EDGE, ha="left")

    # 最后一个位置没有输入、第一个位置没有标签，各画一个空槽
    box(ax, 8 * (CW + GAP), y_in, CW, CH, "无输入", kind="plain", fontsize=FS_CELL,
        color=MUTED)
    label(ax, seq[8][0], y_seq - 0.3, "只当标签", fontsize=FS_SMALL, color=MUTED)
    label(ax, seq[0][0], y_seq - 0.3, "只当输入", fontsize=FS_SMALL, color=MUTED)

    finish(fig, ax, OUTPUT, xlim=(-4.0, x_right + 4.2), ylim=(-0.55, y_seq + CH + 0.75))


if __name__ == "__main__":
    main()
