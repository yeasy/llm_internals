"""生成 11.7 节的图：字符级自动机怎样变成词元级索引。

正文位置：11_serving/11.7_constrained_decoding.md
输出：11_serving/_images/ch11_7_token_index.png

上半幅是正则 \\{"age": (0|[1-9][0-9]*)\\} 的字符级 DFA（12 个状态），上方的弧线是
教学词表里的多字符词元：一个词元一次跨过几个字符状态。下半幅是预计算出的词元级
索引 [12, 16]：每个状态一行、每个词元一列，格内写的是接受该词元后到达的状态，灰格
是不合法的词元。索引由脚本现场算出，与正文表格和 scratchpad 的 toy_index.py 一致。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from _diagram import (ACCENT, DATA_EDGE, DATA_FACE, FS_SMALL, FS_TEXT, FS_TITLE,
                      INK, MASK_EDGE, MASK_FACE, MUTED, NEUTRAL_EDGE, arrow, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("11_serving", "ch11_7_token_index.png")

# ---------- 字符级 DFA ----------
DFA: dict[tuple[int, str], int] = {}
for i, c in enumerate('{"age": '):
    DFA[(i, c)] = i + 1
DFA[(8, "0")] = 9
for c in "123456789":
    DFA[(8, c)] = 10
for c in "0123456789":
    DFA[(10, c)] = 10
DFA[(9, "}")] = 11
DFA[(10, "}")] = 11
N_STATES, ACCEPT = 12, {11}

VOCAB = ["EOS", "{", '"', '{"', "a", "ge", "age", '":', ":", " ",
         "0", "1", "12", "01", "}", "2}"]


def walk(state, text):
    for c in text:
        state = DFA.get((state, c))
        if state is None:
            return None
    return state


def build_index():
    index = []
    for q in range(N_STATES):
        row = []
        for tid, tok in enumerate(VOCAB):
            if tid == 0:
                row.append(q if q in ACCEPT else None)
            else:
                row.append(walk(q, tok))
        index.append(row)
    return index


def show(tok):
    return tok.replace(" ", "_")


# ---------- 上半幅 ----------
R = 0.70
POS = {i: (1.3 + 2.3 * i, 15.2) for i in range(9)}
POS[9] = (22.1, 16.7)
POS[10] = (22.1, 13.7)
POS[11] = (25.0, 15.2)


def node(ax, q, *, ghost=False):
    x, y = POS[q]
    if ghost:
        ax.add_patch(Circle((x, y), R, facecolor="white", edgecolor=MASK_EDGE,
                            linewidth=1.4, linestyle=(0, (3, 2)), zorder=3))
        label(ax, x, y, f"q{q}", fontsize=FS_TEXT, color=MUTED)
        return
    ax.add_patch(Circle((x, y), R, facecolor=DATA_FACE, edgecolor=DATA_EDGE,
                        linewidth=1.6, zorder=3))
    if q in ACCEPT:
        ax.add_patch(Circle((x, y), R - 0.13, facecolor="none", edgecolor=DATA_EDGE,
                            linewidth=1.2, zorder=4))
    label(ax, x, y, f"q{q}", fontsize=FS_SMALL if q >= 10 else FS_TEXT)  # 两位数状态名用小一号字


def edge(ax, a, b, text, *, dy=0.42, dx=0.0):
    (x0, y0), (x1, y1) = POS[a], POS[b]
    d = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    ux, uy = (x1 - x0) / d, (y1 - y0) / d
    arrow(ax, (x0 + ux * R, y0 + uy * R), (x1 - ux * R, y1 - uy * R), lw=1.3)
    label(ax, (x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, text, fontsize=FS_TEXT)


def token_arc(ax, a, b, text, *, height):
    (x0, y0), (x1, _) = POS[a], POS[b]
    ax.annotate("", xy=(x1 - 0.18, y0 + R + 0.05), xytext=(x0 + 0.18, y0 + R + 0.05), zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.5, shrinkA=0, shrinkB=0,
                                connectionstyle="arc3,rad=-0.42"))
    ax.text((x0 + x1) / 2, y0 + height, text, ha="center", va="center", fontsize=FS_TEXT,
            color=ACCENT, zorder=6,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=ACCENT, lw=1.0))


def top_panel(ax):
    label(ax, 0.2, 19.6, "字符级 DFA：12 个状态，按字符转移（_ 表示空格，双圈是接受状态）",
          fontsize=FS_TITLE, bold=True, ha="left")
    for q in range(N_STATES):
        node(ax, q, ghost=(q == 4))
    for i, c in enumerate(['{', '"', 'a', 'g', 'e', '"', ':', '_']):
        edge(ax, i, i + 1, c, dy=-0.45)
    edge(ax, 8, 9, "0", dy=0.5, dx=-0.25)
    edge(ax, 8, 10, "1-9", dy=-0.5, dx=-0.45)
    edge(ax, 9, 11, "}", dy=0.5, dx=0.25)
    edge(ax, 10, 11, "}", dy=-0.5, dx=0.25)
    # q10 的自环
    x, y = POS[10]
    ax.annotate("", xy=(x + 0.35, y - R + 0.06), xytext=(x - 0.35, y - R + 0.06), zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3, shrinkA=0, shrinkB=0,
                                connectionstyle="arc3,rad=1.6"))
    label(ax, x, y - 1.75, "0-9", fontsize=FS_TEXT)

    token_arc(ax, 0, 2, '{"', height=2.05)
    token_arc(ax, 2, 5, "age", height=2.6)
    token_arc(ax, 5, 7, '":', height=2.05)
    label(ax, 17.0, 18.0, "紫色弧线：一个词元一次跨过几个字符状态",
          fontsize=FS_SMALL, color=ACCENT, ha="left")
    label(ax, POS[4][0], 13.75, "没有词元停在 q4", fontsize=FS_SMALL, color=MUTED)

    # 单出边链
    y = 12.75
    x0, x1 = POS[0][0] - R, POS[8][0] + R
    ax.plot([x0, x0, x1, x1], [y + 0.3, y, y, y + 0.3], color=ACCENT, lw=1.5, zorder=2)
    label(ax, (x0 + x1) / 2, y - 0.55,
          'q0 到 q8 每个状态只有一条出边：压缩成一条边，文本 {"age":_ 共 8 个字符，可跳跃前进',
          fontsize=FS_SMALL, color=ACCENT)


# ---------- 下半幅 ----------
CW, CH = 1.32, 0.78
X0, TOP = 4.2, 8.9


def bottom_panel(ax, index):
    label(ax, 0.2, 11.0, "词元级索引 [12, 16]：每个状态一行，格内是接受该词元后到达的状态",
          fontsize=FS_TITLE, bold=True, ha="left")
    label(ax, X0 - 0.25, TOP + 1.05, "词元 ID", fontsize=FS_SMALL, color=MUTED, ha="right")
    label(ax, X0 - 0.25, TOP + 0.4, "词元", fontsize=FS_SMALL, color=MUTED, ha="right")
    for j, tok in enumerate(VOCAB):
        cx = X0 + (j + 0.5) * CW
        label(ax, cx, TOP + 1.05, str(j), fontsize=FS_SMALL, color=MUTED)
        label(ax, cx, TOP + 0.4, show(tok), fontsize=FS_TEXT)
    for q, row in enumerate(index):
        y = TOP - (q + 1) * CH
        n_ok = sum(d is not None for d in row)
        label(ax, X0 - 0.25, y + CH / 2, f"q{q}", fontsize=FS_TEXT, ha="right")
        label(ax, X0 + 16 * CW + 0.3, y + CH / 2, f"允许 {n_ok} 个", fontsize=FS_SMALL,
              color=MUTED, ha="left")
        for j, dst in enumerate(row):
            x = X0 + j * CW
            ok = dst is not None
            ax.add_patch(Rectangle((x, y), CW, CH, facecolor=DATA_FACE if ok else MASK_FACE,
                                   edgecolor=DATA_EDGE if ok else "white", linewidth=0.8,
                                   zorder=2 if not ok else 3))
            if ok:
                text = "结束" if j == 0 else f"q{dst}"
                label(ax, x + CW / 2, y + CH / 2, text, fontsize=FS_SMALL)
    ax.add_patch(Rectangle((X0, TOP - 12 * CH), 16 * CW, 12 * CH, fill=False,
                           edgecolor=NEUTRAL_EDGE, linewidth=1.4, zorder=5))
    # 高亮 q8 这一行
    y8 = TOP - 9 * CH
    ax.add_patch(Rectangle((X0 - 0.06, y8 - 0.04), 16 * CW + 0.12, CH + 0.08, fill=False,
                           edgecolor=ACCENT, linewidth=2.4, zorder=7))
    label(ax, 0.2, y8 + CH / 2, "当前状态", fontsize=FS_SMALL, color=ACCENT, ha="left")
    label(ax, 0.2, TOP - 12 * CH - 0.75,
          "运行时：状态 q8 取出这一行作掩码，只有 ID 10、11、12、15 保留原 logits，"
          r"其余 12 格置为 $-\infty$；",
          fontsize=FS_SMALL, color=ACCENT, ha="left")
    label(ax, 0.2, TOP - 12 * CH - 1.4, "选中词元 12 后，查同一格得到下一状态 q10，不必再走自动机。",
          fontsize=FS_SMALL, color=ACCENT, ha="left")
    label(ax, 0.2, TOP - 12 * CH - 2.05,
          "词元 01 在 q8 不合法（0 之后不能再跟数字），在 q10 合法：同一个词元是否合法取决于状态。",
          fontsize=FS_SMALL, color=MUTED, ha="left")


def main():
    use_cjk_font()
    index = build_index()
    fig, ax = plt.subplots(figsize=(12.2, 9.6))
    top_panel(ax)
    bottom_panel(ax, index)
    finish(fig, ax, OUTPUT, xlim=(-0.2, 27.6), ylim=(-3.0, 20.4))


if __name__ == "__main__":
    main()
