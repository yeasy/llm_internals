"""生成 11.8 节的图：一层 Transformer 在张量并行下的数据流（Llama 3 70B，TP = 2）。

正文位置：11_serving/11.8_multi_gpu_inference.md
输出：11_serving/_images/ch11_8_tp_layer.png

上半幅是注意力子层：Q/K/V 投影按列切（等于按头分给各卡），各卡只对本卡的头、本卡的
KV 缓存做注意力，输出投影 W_O 按行切，得到部分和，all-reduce 相加后两卡都拿到完整输出。
下半幅是 MLP 子层：W_gate、W_up 按列切，SiLU 与逐格相乘在本卡完成，W_down 按行切，
再做一次 all-reduce。形状按 Llama 3 70B（d_model = 8192，64 个 Q 头，8 个 K/V 头，
每头 128 维，MLP 中间层 28672）除以 TP = 2 算出。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from _diagram import (ACCENT, FS_NAME, FS_SMALL, FS_TITLE, INK, MUTED,
                      arrow, box, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("11_serving", "ch11_8_tp_layer.png")

D, N_H, N_KV, D_H, D_FF, TP = 8192, 64, 8, 128, 28672, 2
Q_W, KV_W, FF_W = N_H * D_H // TP, N_KV * D_H // TP, D_FF // TP

# 各列的左边界与宽度
X0, W_X = 1.5, 2.5
X1, W_1 = 4.6, 5.3
X2, W_2 = 10.5, 4.9
X3, W_3 = 16.0, 4.0
X4, W_4 = 20.6, 2.5
X5, W_5 = 23.7, 1.9
X6, W_6 = 26.2, 2.5
BH = 2.9      # 方框高度
ROW_GAP = 0.7  # 两张卡之间的间距


def panel(ax, top, title, col1, col2, col3):
    """画一个子层：两行（两张卡）七列。top 是标题的 y 坐标。"""
    label(ax, 0.2, top, title, ha="left", fontsize=FS_TITLE, bold=True)
    y_hi = top - 1.2 - BH
    y_lo = y_hi - ROW_GAP - BH
    for k, y in enumerate((y_hi, y_lo)):
        mid = y + BH / 2
        label(ax, 0.2, mid, f"卡 {k}", ha="left", fontsize=FS_NAME, bold=True)
        box(ax, X0, y, W_X, BH, f"X\n[T, {D}]\n完整一份", "data", fontsize=FS_SMALL)
        box(ax, X1, y, W_1, BH, col1[k], "weight", fontsize=FS_SMALL)
        box(ax, X2, y, W_2, BH, col2[k], "data", fontsize=FS_SMALL)
        box(ax, X3, y, W_3, BH, col3[k], "weight", fontsize=FS_SMALL)
        box(ax, X4, y, W_4, BH, f"部分和\n[T, {D}]", "data", fontsize=FS_SMALL)
        box(ax, X6, y, W_6, BH, f"输出\n[T, {D}]\n两卡相同", "data", fontsize=FS_SMALL)
        for a, b in ((X0 + W_X, X1), (X1 + W_1, X2), (X2 + W_2, X3), (X3 + W_3, X4),
                     (X4 + W_4, X5), (X5 + W_5, X6)):
            arrow(ax, (a + 0.08, mid), (b - 0.08, mid))
    # all-reduce：跨两张卡的一根竖条
    ax.add_patch(FancyBboxPatch((X5, y_lo), W_5, 2 * BH + ROW_GAP,
                                boxstyle="round,pad=0,rounding_size=0.18",
                                facecolor="white", edgecolor=ACCENT, linewidth=2.0, zorder=3))
    label(ax, X5 + W_5 / 2, y_lo + BH + ROW_GAP / 2, "all-\nreduce\n\n逐格\n相加",
          fontsize=FS_SMALL, color=ACCENT, bold=True)
    # 列标题
    heads = [(X1 + W_1 / 2, "列切：每卡一半的列"), (X2 + W_2 / 2, "本卡内计算，不通信"),
             (X3 + W_3 / 2, "行切：每卡一半的行"), (X5 + W_5 / 2, "通信")]
    for cx, text in heads:
        label(ax, cx, y_hi + BH + 0.45, text, fontsize=FS_SMALL, color=MUTED)
    return y_lo


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(12.4, 9.0))

    attn1 = [f"$W_Q$ [{D}, {Q_W}]\n$W_K$ [{D}, {KV_W}]\n$W_V$ [{D}, {KV_W}]" for _ in range(TP)]
    attn2 = [f"本卡 {N_H // TP} 个 Q 头\n本卡 {N_KV // TP} 个 K/V 头\n只读写本卡 KV 缓存\n拼接后 [T, {Q_W}]"
             for _ in range(TP)]
    attn3 = [f"$W_O$\n[{Q_W}, {D}]" for _ in range(TP)]
    bottom = panel(ax, 17.6, "注意力子层：按头分卡", attn1, attn2, attn3)

    ax.plot([0.2, 28.7], [bottom - 0.9, bottom - 0.9], color="#c9c8c2", lw=1.0)

    mlp1 = [f"$W_{{gate}}$ [{D}, {FF_W}]\n$W_{{up}}$ [{D}, {FF_W}]" for _ in range(TP)]
    mlp2 = [f"SiLU 与逐格相乘\n都是逐格运算\n结果 [T, {FF_W}]" for _ in range(TP)]
    mlp3 = [f"$W_{{down}}$\n[{FF_W}, {D}]" for _ in range(TP)]
    bottom = panel(ax, bottom - 1.9, "MLP 子层：按中间层宽度分卡", mlp1, mlp2, mlp3)

    label(ax, 0.2, bottom - 1.0,
          f"蓝色是数据，橙色是权重。每层 2 次 all-reduce，每次的消息都是 [T, {D}]；"
          "TP 度加大，消息大小不变。",
          ha="left", fontsize=FS_SMALL, color=INK)
    label(ax, 0.2, bottom - 1.85,
          "Decode 时 T 换成本步的词元总数 B；归一化与残差相加在每张卡上对完整的 [T, 8192] 各做一遍。",
          ha="left", fontsize=FS_SMALL, color=MUTED)
    finish(fig, ax, OUTPUT, xlim=(-0.2, 29.1), ylim=(bottom - 2.5, 18.4))


if __name__ == "__main__":
    main()
