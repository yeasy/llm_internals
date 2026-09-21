"""生成 14.4 节的图：一张图怎样变成视觉词元，再排进 LLM 的输入序列。

正文位置：14_future_trends/14.4_multimodal.md
输出：14_future_trends/_images/ch14a_visual_tokens.png

数值全部由除法得到，口径取 LLaVA-1.5：CLIP ViT-L/14 @336，(336/14)^2 = 576 个
patch；投影器是 1024 -> 4096 -> 4096 的两层 MLP（liuhaotian/llava-v1.5-7b 的
config.json：mm_hidden_size=1024、hidden_size=4096、mm_projector_type=mlp2x_gelu）。
下半幅说明占位符替换：文本侧的一个 <image> 位置，在嵌入层被换成 576 行。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _diagram import (ACCENT, DATA_EDGE, DATA_FACE, FS_NAME, FS_SMALL, FS_TEXT,
                      FS_TITLE, MUTED, NEW_FACE, NEW_EDGE, arrow, box, dots,
                      finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("14_future_trends", "ch14a_visual_tokens.png")


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(11.9, 7.0))

    label(ax, 0.0, 14.6, "一张 336×336 的图怎样变成 576 行，再排进序列",
          ha="left", fontsize=FS_TITLE, bold=True)

    # ---------- 上半幅：形状链 ----------
    top = 12.9
    bh, bw, gap = 2.1, 5.2, 1.5

    # 图像方格
    g = 6
    cell = 0.42
    gx, gy = 0.0, top - bh
    for i in range(g):
        for j in range(g):
            ax.add_patch(Rectangle((gx + j * cell, gy + bh - (i + 1) * cell), cell, cell,
                                   facecolor=DATA_FACE, edgecolor=DATA_EDGE, lw=0.7))
    label(ax, gx + g * cell / 2, gy - 0.5, "切成 14×14 的块，24 行 24 列",
          fontsize=FS_SMALL, color=MUTED)

    x = gx + g * cell + gap
    arrow(ax, (x - gap + 0.15, top - bh / 2), (x - 0.15, top - bh / 2), color=MUTED)
    box(ax, x, top - bh, bw, bh, "ViT-L/14 编码\n576 个 patch\n[576, 1024]", "data",
        fontsize=FS_SMALL)
    x += bw
    arrow(ax, (x + 0.15, top - bh / 2), (x + gap - 0.15, top - bh / 2), color=MUTED)
    x += gap
    box(ax, x, top - bh, bw, bh, "两层 MLP 投影\n1024 → 4096 → 4096\n共 2098 万参数",
        "weight", fontsize=FS_SMALL)
    x += bw
    arrow(ax, (x + 0.15, top - bh / 2), (x + gap - 0.15, top - bh / 2), color=MUTED)
    x += gap
    box(ax, x, top - bh, bw, bh, "视觉词元\n[576, 4096]\n与词嵌入同宽", "new",
        fontsize=FS_SMALL)
    label(ax, x + bw / 2, top - bh - 0.55, "没有经过分词器，也没有词表里的 ID",
          fontsize=FS_SMALL, color=ACCENT)

    # ---------- 下半幅：占位符替换 ----------
    y = top - bh - 4.3
    label(ax, 0.0, y + 2.3, "嵌入层里的一次替换：一个占位符换成 576 行",
          ha="left", fontsize=FS_TITLE, bold=True)

    cw, ch = 1.5, 1.05
    seq = [("这", "data"), ("张", "data"), ("图", "data"), ("<image>", "new"),
           ("里", "data"), ("有", "data"), ("什", "data"), ("么", "data")]
    for i, (tok, kind) in enumerate(seq):
        w = cw * 1.8 if tok == "<image>" else cw
        xx = sum(cw * 1.8 if s[0] == "<image>" else cw for s in seq[:i])
        box(ax, xx, y, w, ch, tok, kind, fontsize=FS_SMALL, rounded=False)
    total_w = sum(cw * 1.8 if s[0] == "<image>" else cw for s in seq)
    label(ax, total_w + 0.4, y + ch / 2, "文本侧：8 个词元", ha="left", fontsize=FS_NAME)

    # 展开后的序列
    y2 = y - 2.6
    arrow(ax, (3 * cw + cw * 0.9, y - 0.15), (3 * cw + cw * 0.9, y2 + ch + 0.15),
          color=NEW_EDGE)
    xx = 0.0
    for tok in ("这", "张", "图"):
        box(ax, xx, y2, cw, ch, tok, "data", fontsize=FS_SMALL, rounded=False)
        xx += cw
    vis_w = 7.0
    ax.add_patch(Rectangle((xx, y2), vis_w, ch, facecolor=NEW_FACE, edgecolor=NEW_EDGE,
                           lw=1.4))
    label(ax, xx + vis_w / 2, y2 + ch / 2, "576 行视觉向量", fontsize=FS_SMALL)
    dots(ax, xx + vis_w / 2, y2 - 0.55, spread=0.2, r=0.06, color=MUTED)
    xx += vis_w
    for tok in ("里", "有", "什", "么"):
        box(ax, xx, y2, cw, ch, tok, "data", fontsize=FS_SMALL, rounded=False)
        xx += cw
    label(ax, xx + 0.4, y2 + ch / 2, "送进 LLM：[583, 4096]", ha="left", fontsize=FS_NAME)

    label(ax, 0.0, y2 - 1.35,
          "视觉词元占了 583 行里的 576 行，98.8%。此后的因果掩码、位置编码、KV 缓存"
          "一律按 583 个位置算。",
          ha="left", fontsize=FS_TEXT, color=ACCENT)
    label(ax, 0.0, y2 - 2.1,
          "换成 InternVL 1.5 的 12 块切图加缩略图，同一张图是 3328 行；"
          "1 fps 的 10 分钟视频是 15.36 万行。",
          ha="left", fontsize=FS_TEXT, color=MUTED)

    finish(fig, ax, OUTPUT, xlim=(-0.4, 26.0), ylim=(y2 - 2.7, 15.2))


if __name__ == "__main__":
    main()
