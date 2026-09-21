"""生成 10.6 节插图：投机解码的一轮，草稿 4 个、第 3 个被拒。

正文位置：10_inference_optimization/10.6_speculative_decoding.md
输出：10_inference_optimization/_images/ch10_spec_round.png

蓝 = 数据（草稿词元、目标分布），橙 = 模型（权重），青绿 = 本轮确认产出的词元，
灰 = 作废，紫 = 强调。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import ACCENT, FS_TEXT, MUTED, arrow, box, finish, label
from _style import image_path, use_cjk_font

OUTPUT = image_path("10_inference_optimization", "ch10_spec_round.png")
FS = 9.5


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    left, cw, gap = 5.6, 3.3, 0.35          # 五列的起点、列宽、间距
    xs = [left + k * (cw + gap) for k in range(5)]

    # 第 1 行：草稿模型
    label(ax, 0.2, 11.9, "① 草稿模型\n串行 4 次小前向", fontsize=FS, ha="left")
    box(ax, xs[0], 11.2, cw, 1.4, "上一轮末尾\n词元 x", "data", fontsize=FS)
    for k in range(1, 5):
        box(ax, xs[k], 11.2, cw, 1.4, f"草稿 $d_{k}$\n采自 $q_{k}$", "data", fontsize=FS)
        arrow(ax, (xs[k - 1] + cw, 11.9), (xs[k], 11.9))

    # 第 2 行：目标模型一次前向
    label(ax, 0.2, 8.95, "② 目标模型\n1 次前向，5 个位置", fontsize=FS, ha="left")
    box(ax, xs[0], 8.2, 5 * cw + 4 * gap, 1.5,
        "输入 [x, $d_1$, $d_2$, $d_3$, $d_4$]，因果掩码；一次读权重，得到 5 个位置的分布", "weight",
        fontsize=FS)
    for k in range(5):
        arrow(ax, (xs[k] + cw / 2, 11.2), (xs[k] + cw / 2, 9.7), color=MUTED, lw=1.0)

    # 第 3 行：逐位验证
    label(ax, 0.2, 5.75, "③ 逐位验证\n接受概率 min(1, p/q)", fontsize=FS, ha="left")
    cells = [
        ("$p_1$ 验 $d_1$\n接受", "new"),
        ("$p_2$ 验 $d_2$\n接受", "new"),
        ("$p_3$ 验 $d_3$\n拒绝", "plain"),
        ("$p_4$ 验 $d_4$\n作废，不再看", "neutral"),
        ("$p_5$\n全部接受时才用", "neutral"),
    ]
    for k, (text, kind) in enumerate(cells):
        box(ax, xs[k], 5.0, cw, 1.5, text, kind, fontsize=FS)
        arrow(ax, (xs[k] + cw / 2, 8.2), (xs[k] + cw / 2, 6.5), color=MUTED, lw=1.0)

    # 第 4 行：结果
    label(ax, 0.2, 2.55, "④ 本轮产出\n3 个词元", fontsize=FS, ha="left")
    box(ax, xs[0], 1.8, cw, 1.5, "$d_1$", "new", fontsize=FS_TEXT)
    box(ax, xs[1], 1.8, cw, 1.5, "$d_2$", "new", fontsize=FS_TEXT)
    box(ax, xs[2], 1.8, cw, 1.5, "$d_3'$\n由残差分布重采样", "new", fontsize=FS)
    for k in range(3):
        arrow(ax, (xs[k] + cw / 2, 5.0), (xs[k] + cw / 2, 3.3))
    label(ax, xs[3] + cw + gap / 2, 2.55, "残差分布：norm(max(0, $p_3 - q_3$))\n$d_3$、$d_4$ 位置的 KV 回滚，\n下一轮从 $d_3'$ 接着起草",
          fontsize=FS, color=ACCENT)

    label(ax, 11.8, 0.6, "目标模型读一遍权重，确认了 3 个词元；普通解码同样的读取只确认 1 个",
          fontsize=FS, color=MUTED)
    finish(fig, ax, OUTPUT, xlim=(0, 24.0), ylim=(0.0, 13.0))


if __name__ == "__main__":
    main()
