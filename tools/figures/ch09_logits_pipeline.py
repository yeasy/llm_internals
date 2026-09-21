"""生成 9.1 节的图：一轮 Decode 里，从最后一行隐藏向量到选出词元的处理链。

正文位置：09_decoding/9.1_autoregressive_decode.md
输出：09_decoding/_images/ch09_logits_pipeline.png

形状按 Llama 3 8B：d_model = 4096，n_vocab = 128256（嵌入表行数）。
链上各环节的先后取 vLLM 采样器文档字符串的顺序；各引擎的差异见 9.3.7。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import (ACCENT, FS_SMALL, FS_TEXT, FS_TITLE, INK, MUTED, arrow, box, finish,
                      label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("09_decoding", "ch09_logits_pipeline.png")


def main():
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(11.6, 7.6))
    label(ax, 0.2, 14.3, "一轮 Decode 的后半段：从隐藏向量到下一个词元", fontsize=FS_TITLE,
          bold=True, ha="left")
    label(ax, 0.2, 13.55, "蓝色是数据，橙色是权重，青绿色是本轮新选出的词元；紫色标出各环节在本章的位置",
          fontsize=FS_SMALL, color=MUTED, ha="left")

    # 第一行：LM head
    y1, h = 11.2, 1.5
    box(ax, 0.2, y1, 4.4, h, "最后一层、最后一行\nh[B, 4096]", "data", fontsize=FS_SMALL)
    label(ax, 5.0, y1 + h / 2, "×", fontsize=17)
    box(ax, 5.4, y1, 5.0, h, "LM head\nW_vocab[4096, 128256]", "weight", fontsize=FS_SMALL)
    arrow(ax, (10.5, y1 + h / 2), (11.5, y1 + h / 2))
    box(ax, 11.6, y1, 5.2, h, "logits[B, 128256]\nFP32 每行约 501 KiB", "data", fontsize=FS_SMALL)
    label(ax, 17.1, y1 + h / 2, "前向到此结束", fontsize=FS_SMALL, color=MUTED, ha="left")

    # 第二行：logits 处理链
    y2 = 7.6
    steps = [
        ("① 约束与偏置\n语法掩码、禁词\nlogit_bias", "9.4"),
        ("② 惩罚\n重复惩罚\n频次、存在惩罚", "9.3.6"),
        ("③ 温度\nz / T", "9.3.1"),
        ("④ 截断\nTop-k、Top-p\nMin-p", "9.3.2–9.3.5"),
    ]
    x, w, gap = 0.2, 4.0, 0.75
    arrow(ax, (14.2, y1), (14.2, 10.35))
    ax.plot([14.2, 2.2], [10.35, 10.35], color=INK, lw=1.5, zorder=1)
    arrow(ax, (2.2, 10.35), (2.2, y2 + 2.0))
    for i, (text, sec) in enumerate(steps):
        box(ax, x, y2, w, 2.0, text, "data", fontsize=FS_SMALL)
        label(ax, x + w / 2, y2 - 0.45, sec, fontsize=FS_SMALL, color=ACCENT, bold=True)
        if i < len(steps) - 1:
            arrow(ax, (x + w, y2 + 1.0), (x + w + gap, y2 + 1.0))
        x += w + gap
    label(ax, 19.4, y2 + 1.0, "每一环仍是\n[B, 128256]", fontsize=FS_SMALL, color=MUTED, ha="left")

    # 第三行：选词与收尾
    y3 = 3.2
    x_last = 0.2 + 3 * (w + gap) + w / 2
    arrow(ax, (x_last, y2 - 0.85), (x_last, y3 + 2.0))
    box(ax, x_last - 2.6, y3, 5.2, 2.0, "⑤ 选词\n贪心：argmax\n采样：Softmax 后抽取", "data",
        fontsize=FS_SMALL)
    label(ax, x_last, y3 - 0.45, "9.2、9.3.8", fontsize=FS_SMALL, color=ACCENT, bold=True)
    arrow(ax, (x_last - 2.6, y3 + 1.0), (x_last - 3.5, y3 + 1.0))
    box(ax, x_last - 7.3, y3, 3.8, 2.0, "词元 ID\ny[B]", "new", fontsize=FS_SMALL)
    arrow(ax, (x_last - 7.3, y3 + 1.0), (x_last - 8.2, y3 + 1.0))
    box(ax, x_last - 13.4, y3, 5.2, 2.0, "⑥ 停止判定\nEOS、长度上限\n停止字符串", "neutral",
        fontsize=FS_SMALL)
    label(ax, x_last - 10.8, y3 + 2.45, "9.1.3", fontsize=FS_SMALL, color=ACCENT, bold=True)

    # 回路与输出
    x_stop = x_last - 13.4
    arrow(ax, (x_stop + 1.3, y3), (x_stop + 1.3, 1.2))
    label(ax, x_stop + 1.3, 0.75, "未停：y 作为下一轮\nDecode 的输入", fontsize=FS_SMALL)
    arrow(ax, (x_stop + 3.9, y3), (x_stop + 3.9, 1.2))
    label(ax, x_stop + 4.3, 0.75, "同时：增量反分词，\n流式输出（9.1.5）", fontsize=FS_SMALL, ha="left")

    finish(fig, ax, OUTPUT, xlim=(0, 22.6), ylim=(0, 14.9))


if __name__ == "__main__":
    main()
