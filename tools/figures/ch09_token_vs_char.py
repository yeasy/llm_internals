"""生成 9.4 节的图：字符级自动机与词元级转移的对照。

正文位置：09_decoding/9.4_constrained.md
输出：09_decoding/_images/ch09_token_vs_char.png

正则 [0-9]+\\.[0-9]{2}（两位小数的金额）编译成 5 个状态的字符级 DFA；
下半幅画出几个多字符词元怎样一次跨过多个状态，与正文表 9-10 一致。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from _diagram import (ACCENT, DATA_EDGE, DATA_FACE, FS_SMALL, FS_TEXT, FS_TITLE, INK, MASK_EDGE,
                      MUTED, NEW_EDGE, NEW_FACE, arrow, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("09_decoding", "ch09_token_vs_char.png")

XS = [2.0, 6.6, 11.2, 15.8, 20.4]
Y = 8.6
R = 0.95


def state(ax, i, *, accept=False):
    face, edge = (NEW_FACE, NEW_EDGE) if accept else (DATA_FACE, DATA_EDGE)
    ax.add_patch(Circle((XS[i], Y), R, facecolor=face, edgecolor=edge, linewidth=1.8, zorder=3))
    if accept:
        ax.add_patch(Circle((XS[i], Y), R - 0.2, facecolor="none", edgecolor=edge, linewidth=1.2,
                            zorder=4))
    label(ax, XS[i], Y, f"s{i}", fontsize=FS_TEXT)


def jump(ax, i, j, text, *, above=False, depth=2.1, dx0=0.0, dx1=0.0):
    """画一条词元级转移：从 s_i 到 s_j，走状态的上方或下方。"""
    sign = 1 if above else -1
    y = Y + sign * (R + 0.15)
    yb = Y + sign * depth
    x0, x1 = XS[i] + dx0, XS[j] + dx1
    ax.plot([x0, x0], [y, yb], color=ACCENT, lw=1.8, zorder=2)
    ax.plot([x0, x1], [yb, yb], color=ACCENT, lw=1.8, zorder=2)
    arrow(ax, (x1, yb), (x1, y), color=ACCENT, lw=1.8)
    label(ax, (x0 + x1) / 2, yb + sign * 0.45, text, fontsize=FS_SMALL, color=ACCENT)


def main():
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(11.6, 7.4))
    label(ax, -1.6, 14.6, "语法写在字符上，模型选的是词元：一个词元可以一次跨过几个状态",
          fontsize=FS_TITLE, bold=True, ha="left")
    label(ax, -1.6, 13.85, "黑色：正则 [0-9]+\\.[0-9]{2} 的字符级自动机，每条边吃一个字符；"
          "紫色：一个词元走过的路", fontsize=FS_SMALL, color=MUTED, ha="left")

    for i in range(5):
        state(ax, i, accept=(i == 4))
    edges = ["0-9", ".", "0-9", "0-9"]
    for i, text in enumerate(edges):
        arrow(ax, (XS[i] + R, Y), (XS[i + 1] - R, Y))
        label(ax, (XS[i] + XS[i + 1]) / 2, Y + 0.4, text, fontsize=FS_SMALL)
    arrow(ax, (XS[0] - 2.6, Y), (XS[0] - R, Y))
    label(ax, XS[0] - 2.7, Y + 0.45, "起点", fontsize=FS_SMALL, color=MUTED, ha="left")
    # s1 上的自环
    arrow(ax, (XS[1] - 0.5, Y + R - 0.12), (XS[1] + 0.5, Y + R - 0.12), rad=-2.2)
    label(ax, XS[1] + 1.25, Y + 1.85, "0-9", fontsize=FS_SMALL)
    label(ax, XS[4] + 1.3, Y, "接受状态：\n此时才允许 EOS", fontsize=FS_SMALL, color=MUTED, ha="left")

    jump(ax, 0, 2, "词元 1.：整数位加小数点", above=True, depth=3.5, dx1=-0.3)
    jump(ax, 2, 4, "词元 50：一次填满两位小数", above=True, depth=3.5, dx0=0.3)
    jump(ax, 0, 1, "词元 12：两个字符，停在 s1", dx1=-0.3)
    jump(ax, 1, 3, "词元 .5：小数点加第一位小数", dx0=0.3, dx1=-0.3)

    # 同一个词元在另一个状态不合法
    y = Y - 3.9
    x0 = XS[3] + 0.3
    ax.plot([x0, x0], [Y - R - 0.15, y], color=MASK_EDGE, lw=1.8, linestyle=(0, (4, 2)))
    ax.plot([x0, XS[4] + 0.6], [y, y], color=MASK_EDGE, lw=1.8, linestyle=(0, (4, 2)))
    label(ax, XS[4] + 0.8, y, "无路可走", fontsize=FS_SMALL, color=MUTED, ha="left")
    label(ax, x0 + 0.2, y - 0.55, "词元 50 从 s3 出发：5 到 s4，0 没有出边，不合法",
          fontsize=FS_SMALL, color=MUTED, ha="left")

    finish(fig, ax, OUTPUT, xlim=(-2.0, 27.4), ylim=(3.0, 15.2))


if __name__ == "__main__":
    main()
