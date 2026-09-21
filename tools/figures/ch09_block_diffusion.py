"""生成 9.6 节的图：画布内并行去噪，画布间自回归。

正文位置：09_decoding/9.6_diffusion_lm.md
输出：09_decoding/_images/ch09_block_diffusion.png

上半幅是正文表 9-13 的 8 位置教学例：4 步，每步解开置信度最高的 2 个位置。
下半幅按 DiffusionGemma 模型卡描述的结构画出块间流程：编码器写 KV 缓存，
解码器在画布上做双向注意力并经交叉注意力读缓存；画布长 256，去噪步数上限 48。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _diagram import (FS_SMALL, FS_TEXT, FS_TITLE, INK, KINDS, MASK_EDGE, MASK_FACE, MUTED, arrow,
                      box, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("09_decoding", "ch09_block_diffusion.png")

TOKENS = ["今天", "天气", "很", "好", "，", "适合", "出门", "。"]
COMMIT = [(8, 5), (1, 3), (2, 4), (6, 7)]          # 每步解开的位置（从 1 计）
CW, CH = 1.9, 1.0


def canvas_row(ax, x0, y, done, new):
    for k, tok in enumerate(TOKENS, start=1):
        if k in new:
            face, edge = KINDS["new"]
            text = tok
        elif k in done:
            face, edge = KINDS["data"]
            text = tok
        else:
            face, edge = MASK_FACE, MASK_EDGE
            text = "M"
        ax.add_patch(Rectangle((x0 + (k - 1) * CW, y), CW, CH, facecolor=face, edgecolor=edge,
                               linewidth=1.0, zorder=2))
        ax.text(x0 + (k - 0.5) * CW, y + CH / 2, text, ha="center", va="center",
                fontsize=FS_SMALL, color=INK if text != "M" else MUTED, zorder=3)


def main():
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(11.6, 9.4))
    label(ax, 0.2, 20.3, "(a) 画布之内：每步并行预测全部掩码位，只留最有把握的，其余重新掩上",
          fontsize=FS_TEXT, bold=True, ha="left")
    x0, top = 5.2, 17.6
    for k in range(1, 9):
        label(ax, x0 + (k - 0.5) * CW, top + 1.35, f"位置 {k}", fontsize=FS_SMALL, color=MUTED)
    done = set()
    canvas_row(ax, x0, top, done, set())
    label(ax, x0 - 0.3, top + CH / 2, "起点：全掩码", fontsize=FS_SMALL, ha="right")
    for s, pair in enumerate(COMMIT, start=1):
        y = top - s * (CH + 0.35)
        canvas_row(ax, x0, y, done, set(pair))
        done |= set(pair)
        label(ax, x0 - 0.3, y + CH / 2, f"第 {s} 次前向后", fontsize=FS_SMALL, ha="right")
    label(ax, x0 + 8 * CW + 0.4, top - 2 * (CH + 0.35) + CH / 2,
          "8 个词元用 4 次前向；\n自回归要 8 次", fontsize=FS_SMALL, color=MUTED, ha="left")
    label(ax, x0, top - 4 * (CH + 0.35) - 0.55, "灰：仍是掩码 M；青绿：本步解开；蓝：此前已解开",
          fontsize=FS_SMALL, color=MUTED, ha="left")

    # (b) 画布之间
    label(ax, 0.2, 9.6, "(b) 画布之间：从左到右，已定稿的画布写入 KV 缓存，供后面的画布读取",
          fontsize=FS_TEXT, bold=True, ha="left")
    y = 6.2
    h = 2.2
    box(ax, 0.2, y, 3.6, h, "Prompt\n编码器 Prefill", "data", fontsize=FS_SMALL)
    arrow(ax, (3.8, y + h / 2), (4.6, y + h / 2))
    box(ax, 4.6, y, 5.6, h, "画布 1：解码器去噪\n256 个位置并行\n至多 48 次前向", "new",
        fontsize=FS_SMALL)
    arrow(ax, (10.2, y + h / 2), (11.0, y + h / 2))
    box(ax, 11.0, y, 3.9, h, "编码器处理\n画布 1\n写入缓存", "data", fontsize=FS_SMALL)
    arrow(ax, (14.9, y + h / 2), (15.7, y + h / 2))
    box(ax, 15.7, y, 5.6, h, "画布 2：解码器去噪\n256 个位置并行\n至多 48 次前向", "new",
        fontsize=FS_SMALL)
    arrow(ax, (21.3, y + h / 2), (22.1, y + h / 2))
    label(ax, 22.3, y + h / 2, "依此\n类推", fontsize=FS_SMALL, color=MUTED, ha="left")

    # KV 缓存条
    yk = 2.6
    label(ax, 4.1, yk + 0.5, "KV 缓存", fontsize=FS_SMALL, bold=True, ha="left")
    ax.add_patch(Rectangle((0.2, yk), 3.6, 1.0, facecolor=KINDS["data"][0],
                           edgecolor=KINDS["data"][1], linewidth=1.3))
    label(ax, 2.0, yk + 0.5, "Prompt 的 K/V", fontsize=FS_SMALL)
    ax.add_patch(Rectangle((11.0, yk), 3.9, 1.0, facecolor=KINDS["new"][0],
                           edgecolor=KINDS["new"][1], linewidth=1.3))
    label(ax, 12.95, yk + 0.5, "追加 256 行", fontsize=FS_SMALL)
    arrow(ax, (2.0, y), (2.0, yk + 1.0))
    arrow(ax, (12.95, y), (12.95, yk + 1.0))
    # 读缓存
    arrow(ax, (3.2, yk + 1.0), (6.4, y), style="-|>", rad=0.0, color=MUTED)
    arrow(ax, (13.9, yk + 1.0), (17.4, y), style="-|>", rad=0.0, color=MUTED)
    label(ax, 6.3, yk + 1.7, "交叉注意力\n读缓存", fontsize=FS_SMALL, color=MUTED, ha="left")
    label(ax, 17.3, yk + 1.7, "读 Prompt 与\n画布 1 的 K/V", fontsize=FS_SMALL, color=MUTED, ha="left")
    label(ax, 0.2, 1.6, "画布内部是双向注意力，内容每步都在变，画布自身的 K/V 每次前向重算；"
          "缓存只存已定稿的部分", fontsize=FS_SMALL, color=MUTED, ha="left")

    finish(fig, ax, OUTPUT, xlim=(0, 25.0), ylim=(0.9, 21.0))


if __name__ == "__main__":
    main()
