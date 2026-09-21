"""生成 8.1 节的示意图：一条 SFT 样本怎样变成带掩码的损失。

正文位置：08_alignment/8.1_sft.md
输出：08_alignment/_images/ch08_sft_loss_mask.png

三行对齐：输入词元、该位置要预测的目标词元、掩码 m。掩码对齐的是被预测的
词元而不是输入词元，所以最后一个提示词位置 ⟨assistant⟩ 的 m 已经是 1。
词元序列沿用 3.8 节的教学模板。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import (ACCENT, DATA_EDGE, DATA_FACE, FS_NAME, FS_SMALL, FS_TEXT, INK, MASK_EDGE,
                      MASK_FACE, MUTED, NEW_EDGE, NEW_FACE, arrow, box, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("08_alignment", "ch08_sft_loss_mask.png")

TOKENS = ["〈user〉", "2", "+", "3", "=", "〈assistant〉", "5", "。", "EOS"]
TARGET = ["2", "+", "3", "=", "〈assistant〉", "5", "。", "EOS", "—"]
MASK = [0, 0, 0, 0, 0, 1, 1, 1, None]
NLL = {5: "0.36", 6: "0.11", 7: "0.92"}

CW, CH = 1.55, 0.86


def row(ax, y, values, kinds, *, fontsize=FS_SMALL):
    for j, v in enumerate(values):
        fs = fontsize - 2.5 if len(v) > 7 else fontsize
        box(ax, 1.9 + j * CW, y, CW - 0.08, CH, v, kind=kinds[j], fontsize=fs, rounded=False)


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(9.2, 4.5))

    top = 8.2
    label(ax, 8.9, top + 1.85, "一条 SFT 样本：提示词 6 个位置，回答 3 个位置（含 EOS）",
          fontsize=FS_NAME, bold=True)

    # 位置号
    for j in range(9):
        label(ax, 1.9 + j * CW + (CW - 0.08) / 2, top + 0.42, f"位置 {j + 1}",
              fontsize=FS_SMALL, color=MUTED)

    # 第一行：输入词元
    y1 = top - CH
    label(ax, 1.75, y1 + CH / 2, "输入词元", fontsize=FS_TEXT, ha="right")
    row(ax, y1, TOKENS, ["data"] * 6 + ["new"] * 3)

    # 第二行：该位置预测的目标
    y2 = y1 - CH - 0.75
    label(ax, 1.75, y2 + CH / 2, "要预测的目标", fontsize=FS_TEXT, ha="right")
    row(ax, y2, TARGET, ["plain"] * 5 + ["new"] * 3 + ["neutral"])

    # 错位箭头：位置 j 的输入 -> 位置 j 的目标（即输入 j+1）
    for j in range(9):
        x0 = 1.9 + j * CW + (CW - 0.08) / 2
        arrow(ax, (x0, y1), (x0, y2 + CH), color=MASK_EDGE, lw=1.0)

    # 第三行：掩码
    y3 = y2 - CH - 0.75
    label(ax, 1.75, y3 + CH / 2, "掩码 m", fontsize=FS_TEXT, ha="right")
    for j, m in enumerate(MASK):
        if m is None:
            box(ax, 1.9 + j * CW, y3, CW - 0.08, CH, "—", kind="neutral",
                fontsize=FS_SMALL, rounded=False)
        else:
            box(ax, 1.9 + j * CW, y3, CW - 0.08, CH, str(m),
                kind="new" if m else "plain", fontsize=FS_TEXT,
                bold=bool(m), color=NEW_EDGE if m else MUTED, rounded=False)

    # 第四行：计入损失的三项
    y4 = y3 - CH - 0.75
    label(ax, 1.75, y4 + CH / 2, "本位的 -log p", fontsize=FS_TEXT, ha="right")
    for j in range(9):
        txt = NLL.get(j, "不计")
        box(ax, 1.9 + j * CW, y4, CW - 0.08, CH, txt,
            kind="new" if j in NLL else "plain", fontsize=FS_SMALL,
            color=INK if j in NLL else MUTED, rounded=False)

    label(ax, 8.9, y4 - 1.0,
          "损失 = (0.36 + 0.11 + 0.92) / 3 = 0.463；分母是 m 之和，不是序列长度",
          fontsize=FS_TEXT, color=ACCENT)

    # 提示词 / 回答 的分区标注
    label(ax, 1.9 + 3.0 * CW, top + 1.08, "提示词：只读，不计损失", fontsize=FS_SMALL,
          color=DATA_EDGE)
    label(ax, 1.9 + 7.5 * CW, top + 1.08, "回答：计损失", fontsize=FS_SMALL, color=NEW_EDGE)

    label(ax, 8.9, y4 - 1.75,
          "注意错位：位置 6 的输入仍是提示词，但它预测的是回答的第一个词元，所以 m 已经是 1",
          fontsize=FS_SMALL, color=MUTED)

    finish(fig, ax, OUTPUT, xlim=(-0.6, 18.4), ylim=(y4 - 2.4, top + 2.4))


if __name__ == "__main__":
    main()
