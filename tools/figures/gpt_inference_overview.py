"""图 3-10：整个模型 → 一层 → 一个头，展示一次 Prefill 的结构。

普通多头、预归一化、加性位置方案。实线表示数据流，虚线只表示展开；
跨轮的 KV 缓存与停止条件由相邻时间线和 Decode 图说明。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from _diagram import ACCENT, INK, MUTED, arrow, box, finish, label
from _style import image_path, use_cjk_font

OUTPUT = image_path("03_components", "gpt_inference_overview.png")


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(13, 14))

    def block(cx, y, h, text, kind="neutral", w=6.2):
        box(ax, cx - w / 2, y, w, h, text, kind,
            fontsize=13, linespacing=1.15)

    def down(cx, y1, y2):
        arrow(ax, (cx, y1), (cx, y2), lw=1.6)

    def panel(x, w, title):
        ax.add_patch(FancyBboxPatch(
            (x, 1.4), w, 27.2, boxstyle="round,pad=0,rounding_size=0.3",
            facecolor="#faf9f6", edgecolor="#c8c6c0", lw=1, zorder=0))
        label(ax, x + w / 2, 2.1, title, fontsize=17, bold=True)

    label(ax, 15, -0.9, "GPT 的一次计算：从整体逐级展开", fontsize=22, bold=True)
    label(ax, 15, 0.2, "以 Prefill 为例 · 预归一化 · 普通多头注意力", fontsize=15, color=MUTED)
    panel(0.2, 8.6, "A  整个模型")
    panel(10, 9, "B  展开一层")
    panel(20.2, 9.6, "C  展开一个头")

    # A: 全部位置逐层传递，末层之后才选择最后一行。
    a = 4.5
    block(a, 3, 1.6, "Prompt 词元 ID\nT 个词元", "data")
    down(a, 4.6, 5.2)
    block(a, 5.2, 2.0, "查词嵌入 + 位置向量\n[T, d_model]", "weight")
    down(a, 7.2, 8.0)
    block(a, 8, 1.8, "第 1 层 Transformer", "weight")
    down(a, 9.8, 10.7)
    block(a, 10.7, 2.0, "第 2 层 → … → 第 L 层\n各层参数不同", "weight")
    down(a, 12.7, 14.0)
    label(ax, a, 13.35, "全部位置逐层传递", fontsize=14, color=ACCENT)
    block(a, 14, 1.7, "最终归一化\n[T, d_model]", "weight")
    down(a, 15.7, 16.4)
    block(a, 16.4, 1.9, "取最后一行\n[1, d_model]", "data")
    down(a, 18.3, 19.0)
    block(a, 19, 2.3, "LM head：乘词表矩阵\n[1, d_model]\n→ [1, 词表大小]", "weight")
    down(a, 21.3, 22.0)
    block(a, 22, 2.0, "从词表选择下一词元\n本例取最高分", "data")
    label(ax, a, 25.0, "输出：一个新词元 ID\n不是从输入词元中挑选", fontsize=15)
    label(ax, a, 27.1, "继续生成：见下一张时间线图", fontsize=13, color=MUTED)

    # B: 旁路从各子层归一化前的主干分叉，分别回到残差相加。
    b = 14.5
    label(ax, b, 3.4, "层输入 [T, d_model]", fontsize=15, color=ACCENT)
    down(b, 3.8, 5.0)
    block(b, 5, 1.5, "归一化", "weight")
    down(b, 6.5, 7.8)
    for cx, name in ((12.1, "头 1"), (14.5, "头 2"), (16.9, "头 H")):
        arrow(ax, (b, 7.8), (cx, 8.4))
        box(ax, cx - 1, 8.4, 2, 2.0, name + "\n注意力", "data", fontsize=14)
        arrow(ax, (cx, 10.4), (b, 11.4))
    block(b, 11.4, 1.8, "拼接各头 → 乘 W_O\n[T, d_model]", "weight")
    down(b, 13.2, 13.4)
    block(b, 13.4, 1.8, "＋ 残差相加", "neutral")
    down(b, 15.2, 16.4)
    block(b, 16.4, 1.5, "归一化", "weight")
    down(b, 17.9, 18.6)
    block(b, 18.6, 2.5, "MLP：逐位置加工\n扩维 → 激活 → 降维\n输出仍为 [T, d_model]", "weight")
    down(b, 21.1, 22.1)
    block(b, 22.1, 1.8, "＋ 残差相加", "neutral")
    down(b, 23.9, 25.0)
    label(ax, b, 25.5, "层输出 [T, d_model]", fontsize=15, color=ACCENT)
    label(ax, b, 27.1, "头并行计算；层依次执行", fontsize=14, color=MUTED)
    label(ax, 15.7, 9.4, "…", fontsize=10)
    for fork_y, add_y in ((4.3, 14.3), (15.8, 23.0)):
        ax.plot([b, 10.6, 10.6], [fork_y, fork_y, add_y], color=INK, lw=1.5)
        ax.plot(b, fork_y, "o", color=INK, markersize=3)
        arrow(ax, (10.6, add_y), (11.4, add_y))

    # C: 不再嵌入 Decode 缓存，统一采用 Prefill 的 T×T 形状。
    c = 25
    block(c, 3, 2.1, "归一化后的层输入 " + r"$\widetilde{X}$" + "\n[T, d_model]", "data", w=8.2)
    down(c, 5.1, 5.8)
    block(c, 5.8, 3.1,
          r"$Q=\widetilde{X}W_Q$" + "\n"
          + r"$K=\widetilde{X}W_K,\quad V=\widetilde{X}W_V$"
          + "\n每个投影：[d_model, d_h]\nQ、K、V：[T, d_h]", "weight", w=8.2)
    down(c, 8.9, 9.6)
    block(c, 9.6, 2.4, r"$QK^\mathsf{T}/\sqrt{d_h}$"
          + "\n[T, d_h] × [d_h, T]\n→ 匹配分数 [T, T]", "data", w=8.2)
    down(c, 12, 12.7)
    block(c, 12.7, 2.0, "加因果掩码 M\n未来位置的分数置为 " + r"$-\infty$", "neutral", w=8.2)
    down(c, 14.7, 15.4)
    block(c, 15.4, 2.0, "逐行 Softmax\n→ 注意力权重 [T, T]", "data", w=8.2)
    down(c, 17.4, 18.1)
    block(c, 18.1, 2.3, "注意力权重 × V\n[T, T] × [T, d_h]\n→ 上下文向量 [T, d_h]", "data", w=8.2)
    # V 绕过打分和 Softmax，独立进入加权求和。
    ax.plot([29.1, 29.55, 29.55], [7.35, 7.35, 19.25], color=INK, lw=1.4)
    arrow(ax, (29.55, 19.25), (29.1, 19.25))
    label(ax, c, 21.6, "每一行：该头为一个位置\n从可见上下文读出的向量", fontsize=15)
    label(ax, c, 24.1, "单头完整公式", fontsize=14, color=MUTED)
    label(ax, c, 25.4, r"$\mathrm{softmax}\left(\frac{QK^\mathsf{T}}{\sqrt{d_h}}+M\right)V$",
          fontsize=18)
    label(ax, c, 27.1, "各头共用输入，投影参数不同", fontsize=14, color=MUTED)

    # 展开关系不是计算连线：无箭头虚线连接外框，避免伪造数据流。
    for x1, y1, x2, y2 in ((7.6, 8.0, 10, 2.7), (17.9, 8.4, 20.2, 2.7)):
        ax.plot([x1, x2], [y1, y2], "--", color=ACCENT, lw=1.4, zorder=5)
    label(ax, 15, 29.6, "实线箭头：数据流，均自上而下；紫色虚线：将左侧模块展开", fontsize=15)
    label(ax, 15, 30.8, "T：输入词元数　L：层数　H：头数（正文记作 n_h）", fontsize=15, color=MUTED)
    label(ax, 15, 31.9, "d_model：模型表示宽度　d_h：每头宽度；本图 d_model = H × d_h", fontsize=15, color=MUTED)
    finish(fig, ax, OUTPUT, xlim=(-0.2, 30.2), ylim=(32.8, -1.8))


if __name__ == "__main__":
    main()
