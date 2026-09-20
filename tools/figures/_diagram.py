"""3.8 节各示意图共用的配色与绘图原语。

配色取自一组经过色觉障碍模拟校验的分类色（蓝、橙、青绿、紫），每种角色固定
一个颜色，在所有图里含义一致：

- 蓝：随输入变化的数据（词元、向量、Q/K/V、注意力权重……）
- 橙：训练好就固定的权重
- 青绿：本轮新加入的东西（Decode 新追加的词元、新写入缓存的那一行）
- 紫：强调，用来标出“必须相等”或“要特别留意”的数字

颜色从不单独承担含义：每个色块旁边都有文字标签。
"""

from __future__ import annotations

from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

DATA_EDGE, DATA_FACE, DATA_STRONG = "#2a78d6", "#dbe9fa", "#8db9ee"
WEIGHT_EDGE, WEIGHT_FACE = "#eb6834", "#fde3d6"
NEW_EDGE, NEW_FACE = "#12936a", "#cdeee2"
ACCENT = "#4a3aa7"
MASK_FACE, MASK_EDGE = "#e4e4e1", "#8a8a86"
NEUTRAL_EDGE, NEUTRAL_FACE = "#6b6a66", "#f1f0ec"
INK, MUTED = "#141413", "#52514e"

FS_TITLE, FS_NAME, FS_TEXT, FS_SMALL, FS_OP = 15, 13.5, 12, 11, 18

KINDS = {
    "data": (DATA_FACE, DATA_EDGE),
    "weight": (WEIGHT_FACE, WEIGHT_EDGE),
    "new": (NEW_FACE, NEW_EDGE),
    "neutral": (NEUTRAL_FACE, NEUTRAL_EDGE),
    "plain": ("white", NEUTRAL_EDGE),
}


def box(ax, x, y, w, h, text="", kind="neutral", *, fontsize=FS_TEXT, bold=False,
        color=INK, lw=1.5, rounded=True, zorder=2, linespacing=1.35):
    """以 (x, y) 为左下角画一个带文字的方框；返回中心坐标。"""
    face, edge = KINDS[kind]
    if rounded:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.18",
                               facecolor=face, edgecolor=edge, linewidth=lw, zorder=zorder)
    else:
        patch = Rectangle((x, y), w, h, facecolor=face, edgecolor=edge,
                          linewidth=lw, zorder=zorder)
    ax.add_patch(patch)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
                color=color, fontweight="bold" if bold else "normal",
                linespacing=linespacing, zorder=zorder + 1)
    return x + w / 2, y + h / 2


def arrow(ax, p0, p1, *, color=INK, lw=1.5, style="-|>", rad=0.0, zorder=1):
    ax.annotate("", xy=p1, xytext=p0, zorder=zorder,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                shrinkA=0, shrinkB=0,
                                connectionstyle=f"arc3,rad={rad}"))


def label(ax, x, y, text, *, fontsize=FS_TEXT, color=INK, ha="center", va="center",
          bold=False, linespacing=1.4, zorder=6):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, color=color,
            fontweight="bold" if bold else "normal", linespacing=linespacing, zorder=zorder)


def dots(ax, cx, cy, *, horizontal=True, color=INK, spread=0.26, r=0.075, zorder=7):
    """手工画三个点当省略号，不依赖字体里有没有 ⋯ 和 ⋮ 这两个字形。"""
    for k in (-1, 0, 1):
        dx, dy = (k * spread, 0) if horizontal else (0, k * spread)
        ax.add_patch(Circle((cx + dx, cy + dy), r, facecolor=color,
                            edgecolor="none", zorder=zorder))


def grid(ax, x, top, values, *, kind="data", cell_w=1.0, cell_h=1.0, fontsize=FS_SMALL,
         row_kinds=None, fmt=None, zorder=2):
    """画一个写着数值的矩阵，左上角在 (x, top)；values 是二维列表。

    row_kinds 可为每一行单独指定配色（例如把新追加的一行标成青绿）。
    返回 (右边界 x, 下边界 y)。
    """
    n_rows, n_cols = len(values), len(values[0])
    for i, row in enumerate(values):
        face, edge = KINDS[(row_kinds or {}).get(i, kind)]
        for j, v in enumerate(row):
            cx, cy = x + j * cell_w, top - (i + 1) * cell_h
            ax.add_patch(Rectangle((cx, cy), cell_w, cell_h, facecolor=face,
                                   edgecolor=edge, linewidth=0.8, zorder=zorder))
            if v is not None:
                text = fmt(v) if fmt else str(v)
                ax.text(cx + cell_w / 2, cy + cell_h / 2, text, ha="center", va="center",
                        fontsize=fontsize, color=INK, zorder=zorder + 1)
    _, edge = KINDS[kind]
    ax.add_patch(Rectangle((x, top - n_rows * cell_h), n_cols * cell_w, n_rows * cell_h,
                           fill=False, edgecolor=edge, linewidth=1.6, zorder=zorder + 2))
    return x + n_cols * cell_w, top - n_rows * cell_h


def finish(fig, ax, output, *, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"已写入 {output}")
