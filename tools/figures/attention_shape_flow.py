"""生成图 3-14 及随文分图：教学模型的矩阵形状。

每个格子对应一个数；各面板与正文数值计算相邻。紫色标出乘法的匹配维。
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle

from _diagram import (ACCENT, ACCENT as MATCH, DATA_EDGE, DATA_FACE, DATA_STRONG as LAST_ROW_FACE,
                      INK, MASK_FACE, MUTED, WEIGHT_EDGE, WEIGHT_FACE, arrow, box, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("03_components", "attention_shape_flow.png")

FS_NAME, FS_DIM, FS_TOY, FS_NOTE, FS_OP = 13.5, 13.5, 11.5, 11.5, 18

X0 = 10.5           # 矩阵区域的左边界，左侧留给步骤说明
LABEL_X = X0 - 2.0  # 步骤说明的右边界
ROW_GAP = 6.2       # 相邻两行矩阵之间的留白（单位：格）


@dataclass(frozen=True)
class Dim:
    """矩阵的一个维度：真实大小、图上占几格、中间是否用省略号截断。"""

    label: str
    units: int
    broken: bool = False

    @property
    def gap(self) -> tuple[float, float] | None:
        """省略号所占的那一格（从起点算起的区间）。"""
        if not self.broken:
            return None
        lo = (self.units - 1) / 2
        return lo, lo + 1


T6 = Dim("6", 6)                 # 词元数：两种规模相同，照实画
ONE = Dim("1", 1)
D_MODEL = Dim("768", 5, True)    # 2 格 + 省略号 + 2 格
D_HEAD = Dim("64", 3, True)      # 1 格 + 省略号 + 1 格
N_VOCAB = Dim("50257", 7, True)  # 3 格 + 省略号 + 3 格


class Canvas:
    """在一个坐标轴上按“格”为单位摆放矩阵。"""

    def __init__(self, ax):
        self.ax = ax

    def matrix(self, x, top, rows: Dim, cols: Dim, name, toy, *, weight=False,
               match_rows=False, match_cols=False, lower_tri=False,
               highlight_last_row=False, show_rows=True, show_cols=True):
        """画一个矩阵，左上角在 (x, top)；返回右边界的 x。"""
        face, edge = (WEIGHT_FACE, WEIGHT_EDGE) if weight else (DATA_FACE, DATA_EDGE)
        h, w = rows.units, cols.units
        y = top - h
        ax = self.ax
        ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor="none", zorder=2))
        if lower_tri:  # 右上角（未来位置）涂灰
            ax.add_patch(Polygon(_staircase(x, top, h, w), closed=True,
                                 facecolor=MASK_FACE, edgecolor="none", zorder=3))
        if highlight_last_row:
            ax.add_patch(Rectangle((x, y), w, 1, facecolor=LAST_ROW_FACE,
                                   edgecolor="none", zorder=3))
        for i in range(1, h):
            ax.plot([x, x + w], [y + i, y + i], color=edge, lw=0.6, alpha=0.55, zorder=4)
        for j in range(1, w):
            ax.plot([x + j, x + j], [y, top], color=edge, lw=0.6, alpha=0.55, zorder=4)
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=edge,
                               linewidth=1.6, zorder=5))
        # 省略号：用白色带子把矩阵“截断”，再写上 ⋯ 或 ⋮
        if cols.gap:
            lo, hi = cols.gap
            ax.add_patch(Rectangle((x + lo + 0.08, y - 0.12), hi - lo - 0.16, h + 0.24,
                                   facecolor="white", edgecolor="none", zorder=6))
        if rows.gap:
            lo, hi = rows.gap
            ax.add_patch(Rectangle((x - 0.12, top - hi + 0.08), w + 0.24, hi - lo - 0.16,
                                   facecolor="white", edgecolor="none", zorder=6))
        # 两个方向都截断时，两组点分别放在左上那一块的行中线和列中线上，避免叠成十字
        if cols.gap:
            lo, hi = cols.gap
            cy = top - rows.gap[0] / 2 if rows.gap else y + h / 2
            self.dots(x + (lo + hi) / 2, cy, horizontal=True, color=edge)
        if rows.gap:
            lo, hi = rows.gap
            cx = x + cols.gap[0] / 2 if cols.gap else x + w / 2
            self.dots(cx, top - (lo + hi) / 2, horizontal=False, color=edge)
        # 名称、列数在上，行数紧贴左侧，教学模型的形状在下
        ax.text(x + w / 2, top + 1.55, name, ha="center", va="bottom",
                fontsize=FS_NAME, color=INK, fontweight="bold")
        if show_cols:
            ax.text(x + w / 2, top + 0.15, cols.label, ha="center", va="bottom",
                    fontsize=FS_DIM, color=MATCH if match_cols else INK,
                    fontweight="bold" if match_cols else "normal")
        if show_rows:
            ax.text(x - 0.22, y + h / 2, rows.label, ha="right", va="center",
                    fontsize=FS_DIM, color=MATCH if match_rows else INK,
                    fontweight="bold" if match_rows else "normal")
        if toy:
            ax.text(x + w / 2, y - 0.3, toy, ha="center", va="top",
                    fontsize=FS_TOY, color=MUTED, linespacing=1.35)
        return x + w

    def dots(self, cx, cy, *, horizontal, color, spread=0.26):
        """手工画三个点当省略号，不依赖字体里有没有 ⋯ 和 ⋮ 这两个字形。"""
        for k in (-1, 0, 1):
            dx, dy = (k * spread, 0) if horizontal else (0, k * spread)
            self.ax.add_patch(Circle((cx + dx, cy + dy), 0.075, facecolor=color,
                                     edgecolor="none", zorder=7))

    def op(self, x, top, height, symbol, gap_after=4.3):
        """在一行的垂直中线上写运算符；与右侧矩阵的行数标签留足距离。"""
        self.ax.text(x + 1.15, top - height / 2, symbol, ha="center", va="center",
                     fontsize=FS_OP, color=INK)
        return x + gap_after

    def arrow(self, x, top, height, text, width=7.2):
        """一个带文字说明的箭头，表示不改变形状的运算。"""
        yc = top - height / 2
        self.ax.annotate("", xy=(x + width, yc), xytext=(x + 0.6, yc),
                         arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
        self.ax.text(x + (width + 0.6) / 2, yc + 0.4, text, ha="center",
                     va="bottom", fontsize=FS_NOTE, color=INK, linespacing=1.4)
        return x + width + 1.3

    def step(self, top, height, title, detail):
        yc = top - height / 2
        self.ax.text(LABEL_X, yc + 1.0, title, ha="right", va="center",
                     fontsize=FS_NAME, color=INK, fontweight="bold")
        self.ax.text(LABEL_X, yc - 1.0, detail, ha="right", va="center",
                     fontsize=FS_NOTE, color=MUTED, linespacing=1.4)

    def note(self, x, y, text):
        self.ax.text(x, y, text, ha="left", va="center", fontsize=FS_NOTE,
                     color=MUTED, linespacing=1.5)


def _staircase(x, top, rows, cols):
    """严格上三角（第 i 行只屏蔽第 i 列右边的格子）围成的阶梯形多边形。"""
    pts = [(x + 1, top), (x + cols, top), (x + cols, top - rows + 1)]
    for i in range(rows - 2, -1, -1):
        pts.append((x + i + 1, top - i - 1))
        pts.append((x + i + 1, top - i))
    return pts


def product(filename, left, right, result, *, weight=False, triangular=False, note=""):
    """单行乘法面板，所有尺寸使用教学模型的实际值。"""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    c = Canvas(ax)
    top = 0
    lr, lc, lname = left
    rr, rc, rname = right
    ar, ac, aname = result
    x = c.matrix(1, top, Dim(str(lr), lr), Dim(str(lc), lc), lname, "",
                 match_cols=True, lower_tri=triangular)
    x = c.op(x, top, 4, "×")
    x = c.matrix(x, top, Dim(str(rr), rr), Dim(str(rc), rc), rname, "",
                 weight=weight, match_rows=True)
    x = c.op(x, top, 4, "=")
    x = c.matrix(x, top, Dim(str(ar), ar), Dim(str(ac), ac), aname, "")
    depth = max(lr, rr, ar)
    ax.text((x + 1) / 2, -depth - 1.3, note, ha="center", va="top",
            fontsize=FS_NOTE, color=MUTED)
    ax.set(xlim=(-0.5, x + 1), ylim=(-depth - 3, 3.8), aspect="equal")
    ax.axis("off")
    fig.savefig(image_path("03_components", filename), dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    use_cjk_font()
    product("attention_shape_flow.png", (6, 4, "输入 X"), (4, 2, "投影权重 W_Q"),
            (6, 2, "Query Q"), weight=True,
            note="每行对应一个位置：完整的 4 维输入，经投影得到 2 维 Query。")
    product("attention_scores_shape.png", (6, 2, "Query Q"), (2, 6, "Key 的转置 $K^T$"),
            (6, 6, "点积分数"),
            note="结果第 i 行、第 j 列 = 位置 i 的 Query 与位置 j 的 Key 的点积。\n随后每格除以 √2，形状仍为 [6, 6]。")
    product("attention_read_shape.png", (6, 6, "注意力权重 A"), (6, 2, "Value V"),
            (6, 2, "上下文 C"), triangular=True,
            note="A 的一行给出 6 个权重；对 V 的 6 行加权求和，得到一个 2 维向量。")
    product("attention_merge_shape.png", (6, 4, "拼接后的 C"), (4, 4, "输出投影 W_O"),
            (6, 4, "注意力输出"), weight=True,
            note="头 1、头 2 各输出 [6, 2]，沿列拼接为 [6, 4]；拼接不是相加。")
    product("inference_lm_head_shape.png", (1, 4, "末层最后一行"), (4, 5, "词表投影 W_vocab"),
            (1, 5, "词表分数 logits"), weight=True,
            note="本例候选为 5、。、EOS、6、其他；选最大分数对应的词元。")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    c = Canvas(ax)
    x = c.matrix(1, 0, T6, T6, "缩放后的分数", "")
    x = c.arrow(x, 0, 6, "加因果掩码\n逐行 Softmax", width=8)
    x = c.matrix(x, 0, T6, T6, "注意力权重 A", "", lower_tri=True)
    ax.text((x + 1) / 2, -7.5, "灰色格子：未来位置被屏蔽，Softmax 后权重为 0。\n每一行的权重之和为 1。",
            ha="center", va="top", fontsize=FS_NOTE, color=MUTED)
    ax.set(xlim=(-0.5, x + 1), ylim=(-10, 3.8), aspect="equal")
    ax.axis("off")
    fig.savefig(image_path("03_components", "attention_weights_shape.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
