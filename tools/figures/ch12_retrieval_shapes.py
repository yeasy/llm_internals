"""生成 12.3 节插图：三种检索结构的形状与在线前向次数。

正文位置：12_encoder_models/12.3_longformer_bigbird.md
输出：12_encoder_models/_images/ch12_retrieval_shapes.png

自上而下是交叉编码器、双编码器、迟交互（ColBERT）。橙色是可以离线算好、存进索引
的部分，蓝色是每次查询都要现算的部分。形状与前向次数同正文 12.3.6 的算例
（N = 1,000,000 段落，查询 32 个词元、段落 256 个词元）。

方框宽度不是手填的：脚本先量出框内每条文字在保存 dpi 下的实际宽度，再定下框宽，
免得中文串在不同字体下溢出边框。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import (ACCENT, DATA_EDGE, FS_SMALL, FS_TEXT, INK, MUTED, WEIGHT_EDGE,
                      arrow, box, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("12_encoder_models", "ch12_retrieval_shapes.png")

DPI = 150
BH = 0.92
ROW_GAP = 1.05          # 同一幅里两条链之间的行距
PANEL_GAP = 2.05        # 幅与幅之间的空白

PANELS = [
    ("（a）交叉编码器", "在线前向 100 万次：查询与段落在第 1 层就相互读取",
     [("查询 + 段落，共 288 词元", "编码器", "h0 → 相关度", "data")],
     None),
    ("（b）双编码器", "在线前向 1 次：段落向量与查询无关，可离线建索引",
     [("查询 32 词元", "查询编码器", "u[1, 768]", "data"),
      ("段落 256 词元", "段落编码器（离线）", "v[1, 768]", "weight")],
     r"分数 = u $\cdot$ v"),
    ("（c）迟交互", "在线前向 1 次：索引存每个词元一个向量，大得多",
     [("查询 32 词元", "查询编码器", "Q[32, 128]", "data"),
      ("段落 256 词元", "段落编码器（离线）", "D[256, 128]", "weight")],
     r"$QD^{T}$[32, 256] 逐行取最大再求和"),
]


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(10.4, 6.2), dpi=DPI)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 24)
    ax.set_aspect("equal")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ppu = ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0]

    def units(text, fontsize=FS_SMALL):
        handle = ax.text(0, 0, text, fontsize=fontsize)
        width = handle.get_window_extent(renderer=renderer).width / ppu
        handle.remove()
        return width

    col_texts = [[], [], []]
    score_texts = []
    for _, _, chains, score in PANELS:
        for chain in chains:
            for k in range(3):
                col_texts[k].append(chain[k])
        if score:
            score_texts.append(score)
    col_w = [max(units(t) for t in group) + 0.8 for group in col_texts]
    score_w = max(units(t) for t in score_texts) + 0.8
    gap = 1.25
    x0 = [0.0]
    for w in col_w[:-1]:
        x0.append(x0[-1] + w + gap)
    x_score = x0[-1] + col_w[-1] + gap
    total = x_score + score_w

    y = 0.0
    for title, note, chains, score in PANELS:
        rows_y = []
        for chain in chains:
            y -= BH
            rows_y.append(y)
            kind = chain[3]
            edge = DATA_EDGE if kind == "data" else WEIGHT_EDGE
            for k in range(3):
                box(ax, x0[k], y, col_w[k], BH, chain[k], kind=kind, fontsize=FS_SMALL)
                if k:
                    arrow(ax, (x0[k] - gap, y + BH / 2), (x0[k] - 0.03, y + BH / 2),
                          color=edge)
            y -= ROW_GAP
        if score:
            top, bottom = rows_y[0], rows_y[-1]
            sy = (top + bottom) / 2
            box(ax, x_score, sy, score_w, BH, score, kind="new", fontsize=FS_SMALL)
            for ry in rows_y:
                arrow(ax, (x0[2] + col_w[2] + 0.05, ry + BH / 2),
                      (x_score - 0.03, sy + BH / 2), color=INK, lw=1.1, rad=0.0)
        head = rows_y[0] + BH
        label(ax, 0.0, head + 0.52, title, fontsize=FS_TEXT, bold=True, ha="left")
        label(ax, total, head + 0.52, note, fontsize=FS_SMALL, color=ACCENT, ha="right")
        y -= PANEL_GAP - ROW_GAP

    label(ax, 0.0, y + 0.55, "橙色可离线算好并存进索引，蓝色每次查询现算",
          fontsize=FS_SMALL, color=MUTED, ha="left")
    finish(fig, ax, OUTPUT, xlim=(-0.4, total + 0.4), ylim=(y - 0.1, 1.35))
    print(f"三列框宽 {[round(w, 2) for w in col_w]}、打分框 {score_w:.2f}、总宽 {total:.2f}")


if __name__ == "__main__":
    main()
