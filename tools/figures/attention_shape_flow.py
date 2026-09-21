"""生成图 3-12：一层注意力计算中各矩阵的形状怎样一步步变化。

正文位置：03_components/3.8_gpt_inference_flow.md
输出：03_components/_images/attention_shape_flow.png

各矩阵按 GPT-3 Small 的形状标注（d_model = 768，12 个头，每头 d_h = 64，
词表 50,257）。词元数 6 这一维照实画出，所以 [6, 6] 的分数矩阵和因果掩码的阶梯
是完整的；768、64、50257 这些太大的维度用省略号截断，只保持“64 < 768 < 50257”
的相对宽窄。矩阵下方的灰色方括号是同一个矩阵在教学模型中的形状。矩阵相乘时，
左边的列数必须等于右边的行数，这一对数字用同一种强调色标出。
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle

from _diagram import (ACCENT as MATCH, DATA_EDGE, DATA_FACE, DATA_STRONG as LAST_ROW_FACE,
                      INK, MASK_FACE, MUTED, WEIGHT_EDGE, WEIGHT_FACE)
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


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(13.0, 17.4))
    c = Canvas(ax)
    x0 = X0

    # ---- 第 2 步：投影 ----
    top = 0.0
    c.step(top, 6, "第 2 步 投影", "每个头做三次，\n得到 Q、K、V")
    x = c.matrix(x0, top, T6, D_MODEL, "输入 X", "教学模型 [6, 4]", match_cols=True)
    x = c.op(x, top, 6, "×")
    x = c.matrix(x, top, D_MODEL, D_HEAD, "权重 W_Q", "教学模型 [4, 2]",
                 weight=True, match_rows=True)
    x = c.op(x, top, 6, "=")
    x = c.matrix(x, top, T6, D_HEAD, "Q", "教学模型 [6, 2]")
    c.note(x + 1.6, top - 3, "K、V 同理，各用自己的\n权重 W_K、W_V；\n算出的 K、V 还会留作\nKV 缓存")

    # ---- 第 3 到 5 步：打分、掩码、Softmax ----
    top -= 6 + ROW_GAP
    c.step(top, 6, "第 3 到 5 步 打分", "64 在相乘时消掉，\n结果只和词元数有关")
    x = c.matrix(x0, top, T6, D_HEAD, "Q", "教学模型 [6, 2]", match_cols=True)
    x = c.op(x, top, 6, "×")
    x = c.matrix(x, top, D_HEAD, T6, "K 的转置", "教学模型 [2, 6]", match_rows=True)
    x = c.op(x, top, 6, "=", gap_after=3.7)
    x = c.matrix(x, top, T6, T6, "匹配分数", "教学模型 [6, 6]")
    x = c.arrow(x, top, 6, "每格除以 √64 = 8、\n加掩码、\n逐行 Softmax\n（形状不变）")
    x = c.matrix(x, top, T6, T6, "注意力权重", "教学模型 [6, 6]", lower_tri=True)
    right_edge = x

    # ---- 第 6 步：读取 Value ----
    top -= 6 + ROW_GAP
    c.step(top, 6, "第 6 步 读取", "按权重混合\n各位置的 Value")
    x = c.matrix(x0, top, T6, T6, "注意力权重", "教学模型 [6, 6]",
                 lower_tri=True, match_cols=True)
    x = c.op(x, top, 6, "×", gap_after=3.7)
    x = c.matrix(x, top, T6, D_HEAD, "V", "教学模型 [6, 2]", match_rows=True)
    x = c.op(x, top, 6, "=", gap_after=3.7)
    x = c.matrix(x, top, T6, D_HEAD, "context", "教学模型 [6, 2]")
    c.note(x + 1.6, top - 3, "12 个头各做一遍\n第 2 到 6 步，各交出\n一个 [6, 64] 的 context")

    # ---- 拼接与输出投影 ----
    top -= 6 + ROW_GAP
    c.step(top, 6, "拼接与输出投影", "各头左右并排，\n再乘 W_O 回到 d_model")
    x = c.matrix(x0, top, T6, D_HEAD, "头 1", "", show_cols=False)
    x = c.matrix(x + 0.3, top, T6, D_HEAD, "头 2", "", show_rows=False, show_cols=False)
    c.dots(x + 0.95, top - 3, horizontal=True, color=DATA_EDGE, spread=0.36)
    x = c.matrix(x + 1.9, top, T6, D_HEAD, "头 12", "", show_rows=False, show_cols=False)
    mid = (x0 + x) / 2
    ax.text(mid + 0.9, top + 0.15, "12 × 64 = ", ha="right", va="bottom",
            fontsize=FS_DIM, color=INK)
    ax.text(mid + 0.9, top + 0.15, "768", ha="left", va="bottom",
            fontsize=FS_DIM, color=MATCH, fontweight="bold")
    ax.text(mid, top - 6 - 0.3, "教学模型：2 个 [6, 2] 拼成 [6, 4]", ha="center",
            va="top", fontsize=FS_TOY, color=MUTED)
    x = c.op(x, top, 6, "×")
    x = c.matrix(x, top, D_MODEL, D_MODEL, "权重 W_O", "教学模型 [4, 4]",
                 weight=True, match_rows=True)
    x = c.op(x, top, 6, "=", gap_after=3.7)
    x = c.matrix(x, top, T6, D_MODEL, "注意力输出", "教学模型 [6, 4]",
                 highlight_last_row=True)
    c.note(x + 1.0, top - 3, "与输入 X\n同形状，\n才能做\n残差相加")

    # ---- LM head ----
    top -= 6 + ROW_GAP
    c.step(top, 5, "LM head", "残差、MLP 不改形状；\n只取最后一行去打分")
    x = c.matrix(x0, top - 2, ONE, D_MODEL, "最后位置的表示", "教学模型 [1, 4]",
                 highlight_last_row=True, match_cols=True)
    x = c.op(x, top, 5, "×")
    x = c.matrix(x, top, D_MODEL, N_VOCAB, "权重 W_vocab", "教学模型 [4, 5]",
                 weight=True, match_rows=True)
    x = c.op(x, top, 5, "=", gap_after=3.7)
    x = c.matrix(x, top - 2, ONE, N_VOCAB, "logits", "教学模型 [1, 5]")

    # ---- 图例 ----
    ly = top - 5 - 4.4
    items = (
        (DATA_FACE, DATA_EDGE, "数据：随输入变化"),
        (LAST_ROW_FACE, DATA_EDGE, "最后位置（位置 6）那一行"),
        (WEIGHT_FACE, WEIGHT_EDGE, "权重：训练好就固定"),
        (MASK_FACE, "#8a8a8a", "被因果掩码屏蔽，权重为 0"),
    )
    for k, (face, edge, text) in enumerate(items):
        lx, yy = x0 + (k % 2) * 17.0, ly - (k // 2) * 1.9
        ax.add_patch(Rectangle((lx, yy), 1.1, 1.1, facecolor=face, edgecolor=edge, lw=1.3))
        ax.text(lx + 1.6, yy + 0.55, text, ha="left", va="center", fontsize=FS_NOTE, color=INK)
    ax.text(x0, ly - 3.5, "紫色数字：相乘时，左边的列数必须等于右边的行数，\n这个数在结果里消失。",
            ha="left", va="center", fontsize=FS_NOTE, color=MATCH, linespacing=1.4)
    ax.text(x0, ly - 6.4,
            "黑色数字是 GPT-3 Small 的真实形状；三个点表示中间还有很多列或行没有画出。\n"
            "词元数 6 这一维是照实画的。灰色方括号是同一个矩阵在教学模型中的形状，\n"
            "后面手算用的就是它。",
            ha="left", va="center", fontsize=FS_NOTE, color=MUTED, linespacing=1.4)

    ax.set_xlim(x0 - 11.5, right_edge + 0.8)
    ax.set_ylim(ly - 8.4, 3.8)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"已写入 {OUTPUT}")


if __name__ == "__main__":
    main()
