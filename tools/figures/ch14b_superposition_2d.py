"""生成 14.8 节插图：二维空间里叠加 5 个特征，以及 ReLU 读出如何压住干扰。

正文位置：14_future_trends/14.8_interpretability.md
输出：14_future_trends/_images/ch14b_superposition_2d.png

左：5 个单位方向两两相隔 72 度，正好排成正五边形。任两个方向的余弦只有两种取值，
cos 72 度 = 0.309 与 cos 144 度 = -0.809，均由脚本算出。
右：只有特征 1 取值 1 时，各方向的读出分数就是这两个余弦；减去偏置 0.31 再过 ReLU，
只剩特征 1 自己。负向干扰本就被 ReLU 压掉，真正要压的只有 +0.309 这一项。

蓝 = 当前激活的方向与它的读出，橙 = 其余方向，紫 = 强调。
"""

from __future__ import annotations

from math import cos, pi, sin

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Rectangle

from _diagram import (ACCENT, DATA_EDGE, DATA_FACE, INK, MUTED, WEIGHT_EDGE,
                      WEIGHT_FACE, arrow, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("14_future_trends", "ch14b_superposition_2d.png")

N = 5
R = 1.7
FS, FS_S = 10.0, 9.0
BIAS = 0.31
ANG = [pi / 2 + 2 * pi * k / N for k in range(N)]


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(9.6, 4.8))

    # ---- 左：五个方向 ----
    ax.add_patch(Circle((0, 0), R, fill=False, edgecolor=MUTED, lw=1.0,
                        linestyle=(0, (4, 3)), zorder=1))
    for k, a in enumerate(ANG):
        edge = DATA_EDGE if k == 0 else WEIGHT_EDGE
        arrow(ax, (0, 0), (R * cos(a), R * sin(a)), color=edge, lw=1.8, zorder=3)
        label(ax, (R + 0.42) * cos(a), (R + 0.42) * sin(a), f"特征 {k + 1}",
              fontsize=FS_S, color=edge)
    ax.add_patch(Arc((0, 0), 1.3, 1.3, theta1=18, theta2=90, color=ACCENT, lw=1.4,
                     zorder=4))
    label(ax, 1.12, 0.92, "72°", fontsize=FS_S, color=ACCENT)
    label(ax, 0, -R - 1.05, "5 个方向挤进 2 维：\n相邻夹角 72°，余弦 0.309",
          fontsize=FS_S, color=INK)

    # ---- 右：只有特征 1 激活时，各方向读出多少 ----
    x0, bw, gap, scale = 4.5, 0.62, 0.42, 1.55
    raw = [cos(ANG[0] - a) for a in ANG]
    out = [max(0.0, v - BIAS) for v in raw]
    ax.plot([x0 - 0.3, x0 + N * (bw + gap)], [0, 0], color=INK, lw=1.2, zorder=3)
    for k, (v, o) in enumerate(zip(raw, out)):
        cx = x0 + k * (bw + gap)
        face, edge = (DATA_FACE, DATA_EDGE) if k == 0 else (WEIGHT_FACE, WEIGHT_EDGE)
        ax.add_patch(Rectangle((cx, min(0, v * scale)), bw, abs(v) * scale,
                               facecolor=face, edgecolor=edge, lw=1.3, zorder=2))
        label(ax, cx + bw / 2, v * scale + (0.22 if v > 0 else -0.24), f"{v:+.3f}",
              fontsize=FS_S, color=edge)
        label(ax, cx + bw / 2, -2.35, f"{k + 1}", fontsize=FS_S, color=MUTED)
        label(ax, cx + bw / 2, -2.95, f"{o:.2f}", fontsize=FS_S,
              color=ACCENT if o > 0 else MUTED)
    ax.plot([x0 - 0.3, x0 + N * (bw + gap)], [BIAS * scale] * 2,
            color=ACCENT, lw=1.3, linestyle=(0, (5, 3)), zorder=4)
    label(ax, x0 + N * (bw + gap) + 0.15, BIAS * scale, f"偏置 {BIAS}",
          fontsize=FS_S, color=ACCENT, ha="left")
    label(ax, x0 + N * (bw + gap) / 2 - 0.2, 2.15,
          "只有特征 1 取 1 时，各方向的读出分数", fontsize=FS, bold=True)
    label(ax, x0 - 0.45, -2.35, "方向", fontsize=FS_S, color=MUTED, ha="right")
    label(ax, x0 - 0.45, -2.95, "ReLU 后", fontsize=FS_S, color=MUTED, ha="right")
    label(ax, x0 + N * (bw + gap) / 2 - 0.2, -3.7,
          "负的干扰本来就被 ReLU 压掉，要挡住的只有 +0.309 这一项",
          fontsize=FS_S, color=INK)

    finish(fig, ax, OUTPUT, xlim=(-3.2, 11.0), ylim=(-4.1, 2.6))


if __name__ == "__main__":
    main()
