"""生成图 3-11：一轮 Decode 怎样对着 KV 缓存只算一行。

正文位置：03_components/3.8_gpt_inference_flow.md
输出：03_components/_images/decode_with_cache.png

上半幅用教学模型头 1 在第 1 轮 Decode（位置 7，词元 5）的真实数值：新位置只产生一行
q、k、v；k、v 追加到缓存第 7 行，q 与缓存的 7 行 K 逐行打分，Softmax 后按权重混合
7 行 V，得到 context。数值与正文表 3-9 及第 1 轮 Decode 的算式一致。
下半幅示意真实模型：新位置自左向右穿过 L 层，每层读写的是本层自己的缓存。
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt

from _diagram import (ACCENT, FS_NAME, FS_SMALL, FS_TEXT, FS_TITLE, INK, MUTED,
                      arrow, box, dots, finish, grid, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("03_components", "decode_with_cache.png")

TOKENS = ["〈user〉", "2", "+", "3", "=", "〈assistant〉", "5"]
K = [[0, 0], [0, 1], [1, 0], [1, 1], [0, 1], [0, 1], [1, 0]]  # 头 1：K 与 V 相同
Q = [1, 0]
SCORES = [(Q[0] * k[0] + Q[1] * k[1]) / math.sqrt(2) for k in K]
EXPS = [math.exp(s) for s in SCORES]
WEIGHTS = [e / sum(EXPS) for e in EXPS]
CONTEXT = [sum(w * k[j] for w, k in zip(WEIGHTS, K)) for j in (0, 1)]

CW, CH = 1.25, 0.95   # 单元格的宽和高
NEW_ROW = {6: "new"}  # 第 7 行是本轮新追加的


def f3(v):
    return "0" if abs(v) < 5e-4 else f"{v:.3f}"


def left_panel(ax):
    top = 12.2
    label(ax, 12.4, 18.3, "教学模型的头 1，第 1 轮 Decode：新位置 7，词元是 5",
          fontsize=FS_TITLE, bold=True)

    # 新位置的一行，以及它投影出的 q、k、v
    ty = 16.4
    label(ax, 0.2, ty - 0.48, "新位置这一行\n[1, 4]", ha="left", fontsize=FS_SMALL, color=MUTED)
    grid(ax, 3.6, ty, [[0, 1, 1, 0]], kind="new", cell_w=CW * 0.8, cell_h=CH, fontsize=FS_TEXT)
    arrow(ax, (7.9, ty - 0.48), (11.3, ty - 0.48))
    label(ax, 9.85, ty + 0.05, "乘头 1 的权重 [4, 2]", fontsize=FS_SMALL, color=MUTED)
    label(ax, 11.6, ty - 0.48, "q = k = v =", ha="left", fontsize=FS_TEXT)
    grid(ax, 14.8, ty, [Q], kind="new", cell_w=CW, cell_h=CH, fontsize=FS_TEXT)
    label(ax, 17.7, ty - 0.48, "只算这一行 [1, 2]，\n前 6 个位置不再重算",
          ha="left", fontsize=FS_SMALL, color=ACCENT)

    # 行标签
    x = 0.2
    label(ax, x + 0.35, top + 0.6, "位置", fontsize=FS_SMALL, color=MUTED)
    label(ax, x + 2.55, top + 0.6, "词元", fontsize=FS_SMALL, color=MUTED)
    for i, tok in enumerate(TOKENS):
        kind = "new" if i == 6 else "data"
        label(ax, x + 0.35, top - (i + 0.5) * CH, str(i + 1), fontsize=FS_TEXT, color=MUTED)
        box(ax, x + 1.0, top - (i + 1) * CH + 0.08, 3.1, CH - 0.16, tok, kind,
            fontsize=FS_SMALL, lw=1.0)

    # K 缓存
    xk = 5.2
    label(ax, xk + CW, top + 1.55, "缓存的 K", fontsize=FS_NAME, bold=True)
    label(ax, xk + CW, top + 0.6, "[7, 2]", fontsize=FS_SMALL, color=MUTED)
    grid(ax, xk, top, K, kind="data", cell_w=CW, cell_h=CH, row_kinds=NEW_ROW, fontsize=FS_TEXT)

    # 分数
    xs = 11.4
    label(ax, 9.55, top - 3.3, "q 与每一行\n做点积，\n再除以 √2", fontsize=FS_SMALL, color=MUTED)
    label(ax, xs + 0.95, top + 1.55, "分数", fontsize=FS_NAME, bold=True)
    label(ax, xs + 0.95, top + 0.6, "7 个", fontsize=FS_SMALL, color=MUTED)
    grid(ax, xs, top, [[s] for s in SCORES], kind="new", cell_w=1.9, cell_h=CH,
         fontsize=FS_TEXT, fmt=f3)

    # 权重
    xw = 16.4
    arrow(ax, (xs + 2.05, top - 3.3), (xw - 0.15, top - 3.3))
    label(ax, (xs + 1.9 + xw) / 2, top - 2.75, "Softmax", fontsize=FS_SMALL, color=MUTED)
    label(ax, xw + 0.95, top + 1.55, "权重", fontsize=FS_NAME, bold=True)
    label(ax, xw + 0.95, top + 0.6, "和为 1", fontsize=FS_SMALL, color=MUTED)
    grid(ax, xw, top, [[w] for w in WEIGHTS], kind="new", cell_w=1.9, cell_h=CH,
         fontsize=FS_TEXT, fmt=f3)

    # V 缓存
    xv = 20.4
    label(ax, xw + 2.65, top - 3.3, "×", fontsize=18)
    label(ax, xv + CW, top + 1.55, "缓存的 V", fontsize=FS_NAME, bold=True)
    label(ax, xv + CW, top + 0.6, "[7, 2]", fontsize=FS_SMALL, color=MUTED)
    grid(ax, xv, top, K, kind="data", cell_w=CW, cell_h=CH, row_kinds=NEW_ROW, fontsize=FS_TEXT)

    # context
    xc = 20.4
    arrow(ax, (xv + CW, top - 7 * CH - 0.1), (xv + CW, top - 7 * CH - 1.25))
    label(ax, xv + CW + 0.3, top - 7 * CH - 0.68, "按权重相加", ha="left",
          fontsize=FS_SMALL, color=MUTED)
    grid(ax, xc - 0.65, top - 7 * CH - 1.35, [CONTEXT], kind="new", cell_w=1.9, cell_h=CH,
         fontsize=FS_TEXT, fmt=f3)
    label(ax, xc - 0.95, top - 7 * CH - 1.83, "context [1, 2]", ha="right", fontsize=FS_TEXT,
          bold=True)

    # 图例式说明
    label(ax, 0.2, top - 7 * CH - 1.0, "蓝色 6 行：Prefill 时写入缓存，这一轮直接读取",
          ha="left", fontsize=FS_SMALL)
    label(ax, 0.2, top - 7 * CH - 1.85, "青绿：这一轮新算出的数；其中 K、V 的第 7 行追加进缓存",
          ha="left", fontsize=FS_SMALL)


def bottom_panel(ax, top):
    """真实模型：新位置自左向右穿过 L 层，每层读写本层自己的缓存。"""
    label(ax, 12.4, top, "真实模型：新位置依次穿过 L 层，每层各有一份缓存", fontsize=FS_TITLE, bold=True)
    y = top - 3.6
    items = [("新词元\n的向量", 0.2, 3.4, "new"), ("第 1 层\n本层的权重", 4.9, 3.9, "weight"),
             ("第 2 层\n本层的权重", 10.1, 3.9, "weight"), ("第 L 层\n本层的权重", 16.6, 3.9, "weight"),
             ("LM head →\n下一个词元", 21.8, 3.6, "weight")]
    for k, (text, x, w, kind) in enumerate(items):
        box(ax, x, y, w, 1.9, text, kind, fontsize=FS_SMALL)
        if kind == "weight" and "层" in text:
            cx = x + (w - 7 * 0.5) / 2
            arrow(ax, (x + w / 2, y - 0.05), (x + w / 2, y - 1.15), style="<|-|>")
            for c in range(7):
                box(ax, cx + c * 0.5, y - 2.0, 0.5, 0.8, "", "new" if c == 6 else "data",
                    rounded=False, lw=1.0)
            label(ax, x + w / 2, y - 2.55, text.split("\n")[0] + "的 KV 缓存", fontsize=FS_SMALL,
                  color=MUTED)
    mid = y + 0.95
    arrow(ax, (3.65, mid), (4.85, mid))
    arrow(ax, (8.85, mid), (10.05, mid))
    arrow(ax, (14.05, mid), (14.75, mid))
    dots(ax, 15.3, mid, horizontal=True, color=INK, spread=0.3, r=0.09)
    arrow(ax, (15.85, mid), (16.55, mid))
    arrow(ax, (20.55, mid), (21.75, mid))
    label(ax, 12.4, y - 3.75, "各层的缓存互不通用：第 2 层的 K、V 由第 2 层自己的权重算出",
          fontsize=FS_SMALL, color=ACCENT)


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(12.4, 12.5))
    left_panel(ax)
    ax.plot([0.2, 25.2], [2.5, 2.5], color="#c9c8c2", lw=1.0)
    bottom_panel(ax, 1.2)
    finish(fig, ax, OUTPUT, xlim=(-0.3, 25.8), ylim=(-7.0, 19.2))


if __name__ == "__main__":
    main()
