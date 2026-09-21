"""生成 9.2 节的图：两步搜索树上，贪心搜索与束宽 2 的束搜索各走哪条路。

正文位置：09_decoding/9.2_greedy_beam.md
输出：09_decoding/_images/ch09_beam_tree.png

教学分布：第 1 步 A 0.5、B 0.4、C 0.1；第 2 步的条件分布见 P2。
联合概率由脚本相乘得到，与正文表 9-4 一致。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import (ACCENT, FS_SMALL, FS_TEXT, FS_TITLE, INK, MASK_EDGE, MUTED, arrow, box,
                      finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("09_decoding", "ch09_beam_tree.png")

P1 = {"A": 0.5, "B": 0.4, "C": 0.1}
P2 = {
    "A": {"x": 0.40, "y": 0.35, "z": 0.25},
    "B": {"x": 0.90, "y": 0.06, "z": 0.04},
    "C": {"x": 0.50, "y": 0.30, "z": 0.20},
}
BEAM = 2


def main():
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(11.6, 8.6))
    label(ax, 0.2, 17.4, "两步搜索树：贪心只跟单步最大，束宽 2 多留一条，找到联合概率更高的 B x",
          fontsize=FS_TITLE, bold=True, ha="left")
    label(ax, 0.2, 16.65, "蓝色：束搜索保留；青绿：束搜索最终选中；灰色：被剪掉；紫色粗线：贪心路径",
          fontsize=FS_SMALL, color=MUTED, ha="left")

    kept1 = sorted(P1, key=P1.get, reverse=True)[:BEAM]
    cands = {(a, b): P1[a] * P2[a][b] for a in kept1 for b in P2[a]}
    kept2 = sorted(cands, key=cands.get, reverse=True)[:BEAM]
    best = kept2[0]
    greedy_a = max(P1, key=P1.get)
    greedy_b = max(P2[greedy_a], key=P2[greedy_a].get)

    x_root, x1, x2 = 0.4, 6.0, 13.2
    bw, bh = 3.0, 1.2
    y_root = 7.6
    box(ax, x_root, y_root, bw, bh, "Prompt", "neutral", fontsize=FS_SMALL)
    ys1 = {"A": 12.4, "B": 7.6, "C": 2.8}
    for a, y in ys1.items():
        keep = a in kept1
        box(ax, x1, y, bw, bh, f"{a}   {P1[a]:.1f}", "data" if keep else "plain",
            fontsize=FS_TEXT, color=INK if keep else MUTED)
        g = a == greedy_a
        arrow(ax, (x_root + bw, y_root + bh / 2), (x1, y + bh / 2),
              color=ACCENT if g else (INK if keep else MASK_EDGE), lw=3.0 if g else 1.4)
        for k, b in enumerate(P2[a]):
            yy = y + 1.45 - k * 1.45
            joint = P1[a] * P2[a][b]
            if (a, b) == best:
                kind, col = "new", INK
            elif (a, b) in kept2:
                kind, col = "data", INK
            else:
                kind, col = "plain", MUTED
            box(ax, x2, yy, 6.2, 1.1, f"{a} {b}   {P1[a]:.1f} × {P2[a][b]:.2f} = {joint:.3f}", kind,
                fontsize=FS_SMALL, color=col)
            g2 = g and b == greedy_b
            arrow(ax, (x1 + bw, y + bh / 2), (x2, yy + 0.55),
                  color=ACCENT if g2 else (INK if (a, b) in kept2 else MASK_EDGE),
                  lw=3.0 if g2 else 1.2)
    label(ax, x1 + bw / 2, 15.55, "第 1 步", fontsize=FS_TEXT, bold=True)
    label(ax, x2 + 3.1, 15.55, "第 2 步：联合概率", fontsize=FS_TEXT, bold=True)
    label(ax, x1 + bw / 2, 1.9, "C 排第 3，束宽 2 在此剪掉，\n其后继不再送入模型", fontsize=FS_SMALL, color=MUTED)
    label(ax, x2 + 6.5, ys1["A"] + 2.0, "贪心终点\n0.200", fontsize=FS_SMALL, color=ACCENT, ha="left")
    label(ax, x2 + 6.5, ys1["B"] + 2.0, "束搜索终点\n0.360", fontsize=FS_SMALL, color=INK, ha="left")

    finish(fig, ax, OUTPUT, xlim=(0, 22.4), ylim=(0.8, 18.0))


if __name__ == "__main__":
    main()
