"""生成 14.1 节的图：线性注意力的两种相乘顺序，以及因果情形下的递推状态。

正文位置：14_future_trends/14.1_efficient_attention.md
输出：14_future_trends/_images/ch14a_linear_state.png

上半幅并列两种结合顺序：先算 [n, n] 的打分矩阵（softmax 注意力只能这样算），
与先算 [d_k, d_v] 的状态矩阵（去掉 softmax 后才允许这样算）。形状按 Llama 3 8B
的单头取值：d_k = d_v = 128，序列长 n。
下半幅画因果情形的递推：状态 S 的形状与 n 无关，每来一个词元加一个外积。
图中数值即正文 14.1.2 第五步的 2x2 三步手算，phi 取恒等映射。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import (ACCENT, FS_NAME, FS_SMALL, FS_TEXT, FS_TITLE, INK, MUTED,
                      arrow, box, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("14_future_trends", "ch14a_linear_state.png")

# 2x2 手算的三步：每步的 (k, v, q, S, z, y)
STEPS = [
    ("(1, 0)", "(2, 0)", "[[2, 0],\n[0, 0]]", "(1, 0)", "(2.0, 0)"),
    ("(1, 0)", "(0, 3)", "[[2, 3],\n[0, 0]]", "(2, 0)", "(1.0, 1.5)"),
    ("(0, 1)", "(5, 5)", "[[2, 3],\n[5, 5]]", "(2, 1)", "(1.0, 1.5)"),
]


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(11.8, 8.2))

    # ---------- 上半幅：两种结合顺序 ----------
    label(ax, 0.0, 15.3, "同一个式子，两种相乘顺序", ha="left", fontsize=FS_TITLE, bold=True)

    top = 13.6
    bw, bh, gap = 3.6, 1.3, 1.15

    # 左路：先 QK^T
    label(ax, 0.0, top + 0.55, "先算 Q K$^T$：softmax 注意力只能走这条", ha="left",
          fontsize=FS_TEXT, bold=True, color=MUTED)
    x = 0.0
    c1 = box(ax, x, top - bh, bw, bh, "Q[n, 128]", "data", fontsize=FS_NAME)
    label(ax, x + bw + gap / 2, top - bh / 2, "×", fontsize=FS_TITLE, color=MUTED)
    x += bw + gap
    c2 = box(ax, x, top - bh, bw, bh, "K$^T$[128, n]", "data", fontsize=FS_NAME)
    x += bw + gap
    label(ax, x - gap / 2, top - bh / 2, "→", fontsize=FS_TITLE, color=MUTED)
    box(ax, x, top - bh, bw, bh, "分数[n, n]", "new", fontsize=FS_NAME)
    label(ax, x + bw / 2, top - bh - 0.55, "随 n 平方增长", fontsize=FS_SMALL, color=ACCENT)
    x += bw + gap
    label(ax, x - gap / 2, top - bh / 2, "×", fontsize=FS_TITLE, color=MUTED)
    box(ax, x, top - bh, bw, bh, "V[n, 128]", "data", fontsize=FS_NAME)
    label(ax, x + bw + 0.5, top - bh / 2, "→ 输出[n, 128]", ha="left", fontsize=FS_NAME)
    del c1, c2

    # 右路：先 K^T V。分两行画，让"先乘哪两块"在图上可见。
    top2 = top - 3.5
    bw2, x0 = 4.2, 0.8
    label(ax, 0.0, top2 + 0.55, "先算 φ(K)$^T$ V：去掉 softmax 之后才允许", ha="left",
          fontsize=FS_TEXT, bold=True, color=MUTED)
    x = x0
    label(ax, x - 0.5, top2 - bh / 2, "①", fontsize=FS_TITLE, color=ACCENT)
    box(ax, x, top2 - bh, bw2, bh, "φ(K)$^T$[128, n]", "data", fontsize=FS_NAME)
    x += bw2 + gap
    label(ax, x - gap / 2, top2 - bh / 2, "×", fontsize=FS_TITLE, color=MUTED)
    box(ax, x, top2 - bh, bw2, bh, "V[n, 128]", "data", fontsize=FS_NAME)
    x += bw2 + gap
    label(ax, x - gap / 2, top2 - bh / 2, "→", fontsize=FS_TITLE, color=MUTED)
    box(ax, x, top2 - bh, bw2, bh, "S[128, 128]", "new", fontsize=FS_NAME)
    label(ax, x + bw2 / 2, top2 - bh - 0.55, "与 n 无关", fontsize=FS_SMALL, color=ACCENT)

    top3 = top2 - 2.45
    x = x0
    label(ax, x - 0.5, top3 - bh / 2, "②", fontsize=FS_TITLE, color=ACCENT)
    box(ax, x, top3 - bh, bw2, bh, "φ(Q)[n, 128]", "data", fontsize=FS_NAME)
    x += bw2 + gap
    label(ax, x - gap / 2, top3 - bh / 2, "×", fontsize=FS_TITLE, color=MUTED)
    box(ax, x, top3 - bh, bw2, bh, "S[128, 128]", "new", fontsize=FS_NAME)
    x += bw2 + gap
    label(ax, x - gap / 2, top3 - bh / 2, "→", fontsize=FS_TITLE, color=MUTED)
    box(ax, x, top3 - bh, bw2, bh, "输出[n, 128]", "data", fontsize=FS_NAME)

    label(ax, 0.0, top3 - bh - 1.0,
          "两条路的结果相同；括号打在哪里，决定中间那块是 [n, n] 还是 [128, 128]。",
          ha="left", fontsize=FS_TEXT, color=ACCENT)

    # ---------- 下半幅：递推 ----------
    ytop = top3 - bh - 2.35
    label(ax, 0.0, ytop, "因果情形：同一个 S 写成逐步递推，就是一个 RNN",
          ha="left", fontsize=FS_TITLE, bold=True)

    cw, kh, sh, zh = 5.4, 0.9, 2.0, 0.9
    y = ytop - 1.5 - (kh + sh + zh)
    for t, (k, v, s, z, yv) in enumerate(STEPS, 1):
        x = (t - 1) * (cw + 1.6)
        label(ax, x + cw / 2, y + kh + sh + zh + 0.45, f"第 {t} 步", fontsize=FS_TEXT,
              bold=True)
        box(ax, x, y + sh + zh, cw, kh, f"k = {k}    v = {v}", "data", fontsize=FS_SMALL)
        box(ax, x, y + zh, cw, sh, f"S = {s}", "new", fontsize=FS_NAME, linespacing=1.25)
        box(ax, x, y, cw, zh, f"z = {z}", "neutral", fontsize=FS_SMALL)
        label(ax, x + cw / 2, y - 0.62, f"q = (1, 0) 读出 y = {yv}", fontsize=FS_SMALL,
              color=ACCENT)
        if t < len(STEPS):
            arrow(ax, (x + cw + 0.25, y + zh + sh / 2), (x + cw + 1.35, y + zh + sh / 2),
                  color=MUTED)
            label(ax, x + cw + 0.8, y + zh + sh / 2 + 0.45, "+k v$^T$", fontsize=FS_SMALL,
                  color=MUTED)

    label(ax, 0.0, y - 1.75,
          "第 1、2 步的键相同：第二个值没有覆盖第一个，而是被加了上去，读出的是两者的均值。\n"
          "这正是 delta 规则要改掉的地方。",
          ha="left", fontsize=FS_TEXT, color=INK)

    finish(fig, ax, OUTPUT, xlim=(-0.4, 21.0), ylim=(y - 2.8, 15.9))


if __name__ == "__main__":
    main()
