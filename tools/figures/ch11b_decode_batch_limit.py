"""图：批处理能把 Decode 推多远（11.12.2）。

左：H200 上 Llama 3 70B（FP8 权重、4K 上下文、FP16 KV）的总吞吐与每流速度随批大小的变化，
    每步耗时 = (权重 + B × 每序列 KV) / 显存带宽；竖线是显存容量允许的最大批。
右：算术强度 I(B) = B(2N + 4sdL) / (W + B·s·kv) 随批大小饱和，极限为 2N/(s·kv) + 2d/(n_kv·d_h·b)，b 为每个 KV 数的字节数（FP16 KV 下第二项为 8）；
    横线是 H200 的 FP8 拐点。上下文一长，曲线在拐点以下就到顶：加批也用不满算力。

口径与正文一致：只数权重和 KV 的读取，是上界估算。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _diagram import ACCENT, DATA_EDGE, INK, MUTED, NEW_EDGE, WEIGHT_EDGE
from _style import image_path, use_cjk_font

OUTPUT = image_path("11_serving", "ch11b_decode_batch_limit.png")

N = 70e9                      # 参数量
W = 70e9                      # FP8 权重字节
L, D, N_KV, D_H = 80, 8192, 8, 128
KV_TOKEN = 2 * L * N_KV * D_H * 2   # FP16，327,680 字节
BW = 4.8e12                   # H200 显存带宽
RIDGE_FP8 = 3958e12 / 2 / BW  # 稠密 FP8 峰值 / 带宽
B_MAX = (141e9 * 0.9 - W - 5e9) / (4096 * KV_TOKEN)


def main() -> None:
    use_cjk_font()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.1))

    # 左：吞吐与每流速度
    b = np.arange(1, 65)
    step = (W + b * 4096 * KV_TOKEN) / BW
    ok = b <= B_MAX
    ax1.plot(b[ok], (b / step)[ok], color=DATA_EDGE, lw=2.2)
    ax1.plot(b[~ok], (b / step)[~ok], color=DATA_EDGE, lw=2.2, alpha=0.3)
    ax1.set_xlabel("批大小 B（4K 上下文的序列数）", fontsize=11)
    ax1.set_ylabel("总吞吐（词元/秒）", color=DATA_EDGE, fontsize=11)
    ax1.tick_params(axis="y", colors=DATA_EDGE)
    ax1.set_xlim(0, 64)
    ax1.set_ylim(0, 2400)
    ax1b = ax1.twinx()
    ax1b.plot(b[ok], (1 / step)[ok], color=WEIGHT_EDGE, lw=2.2, ls="--")
    ax1b.plot(b[~ok], (1 / step)[~ok], color=WEIGHT_EDGE, lw=2.2, ls="--", alpha=0.3)
    ax1b.set_ylabel("每流速度（词元/秒）", color=WEIGHT_EDGE, fontsize=11)
    ax1b.tick_params(axis="y", colors=WEIGHT_EDGE)
    ax1b.set_ylim(0, 80)
    ax1.axvline(B_MAX, color=ACCENT, lw=1.3, ls=":")
    ax1.text(B_MAX + 1.2, 1050, f"显存只放得下\n约 {int(B_MAX)} 条，\n浅色段到不了", color=ACCENT, fontsize=10.5, va="top")
    for bb in (1, 32):
        s = (W + bb * 4096 * KV_TOKEN) / BW
        ax1.plot([bb], [bb / s], "o", color=DATA_EDGE, ms=5)
        ax1.annotate(f"B={bb}：{bb / s:.0f} 词元/秒\n每流 {1 / s:.0f}", xy=(bb, bb / s),
                     xytext=((14, 130) if bb == 1 else (9, 2000)), fontsize=10, color=INK,
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax1.set_title("H200，70B FP8：吞吐涨、每流变慢", fontsize=12)
    ax1.grid(alpha=0.25)

    # 右：算术强度饱和
    bb = np.logspace(0, 3.3, 200)
    for ctx, color, name in ((1024, NEW_EDGE, "1K"), (4096, DATA_EDGE, "4K"), (16384, WEIGHT_EDGE, "16K")):
        f_tok = 2 * N + 4 * ctx * D * L
        inten = bb * f_tok / (W + bb * ctx * KV_TOKEN)
        lim = f_tok / (ctx * KV_TOKEN)
        ax2.plot(bb, inten, color=color, lw=2.2)
        ax2.text(2300, lim, f"上下文 {name}\n极限 {lim:.0f}", color=color, fontsize=10.5, va="center")
    ax2.axhline(RIDGE_FP8, color=ACCENT, lw=1.4, ls="--")
    ax2.text(1.15, RIDGE_FP8 * 1.18, f"H200 的 FP8 拐点：{RIDGE_FP8:.0f}", color=ACCENT, fontsize=10.5)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(1, 2000)
    ax2.set_ylim(1, 1500)
    ax2.set_xlabel("批大小 B", fontsize=11)
    ax2.set_ylabel("算术强度（FLOPs / 读 1 字节）", fontsize=11)
    ax2.set_title("算术强度随批大小饱和", fontsize=12)
    ax2.grid(alpha=0.25, which="both")

    fig.tight_layout(w_pad=2.0)
    fig.subplots_adjust(right=0.89)
    fig.savefig(OUTPUT, dpi=150, facecolor="white")
    print(f"已写入 {OUTPUT}")


if __name__ == "__main__":
    main()
