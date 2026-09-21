"""生成 11.3 节插图：批大小、到达率与 TPOT 的关系。

正文位置：11_serving/11.3_scheduler_loop.md
输出：11_serving/_images/ch11_3_batch_latency.png

口径与正文 11.3.6 一致：Llama 3 8B，BF16 权重与 KV；H100 显存带宽 3.35 TB/s；
Decode 步长取访存下界 a + b·B，a 是读一遍权重的时间，b 是读一个请求 KV 的时间；
输入 1,000、输出 500 个词元，Decode 期间平均上下文 1,250；KV 池 400,000 个词元。
左：步长（即 TPOT）随批大小线性上升，吞吐按 B/(a + b·B) 趋于饱和。
右：由 Little 定律 B = λ·N·(a + b·B) 解出 TPOT = a/(1 - λ·N·b)，到达率逼近容量时陡升；
KV 池装满后 B 封顶，TPOT 停在 a + b·B_max，多出的请求进入等待队列。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _diagram import ACCENT, DATA_EDGE, FS_SMALL, FS_TEXT, INK, MUTED, WEIGHT_EDGE
from _style import image_path, use_cjk_font

OUTPUT = image_path("11_serving", "ch11_3_batch_latency.png")

L, D, NH, DH, NKV, DFF, NVOCAB = 32, 4096, 32, 128, 8, 14336, 128256
PARAMS = L * (2 * D * NH * DH + 2 * D * NKV * DH + 3 * D * DFF + 2 * D) + 2 * NVOCAB * D + D
BW = 3.35e12
KV_TOK = 2 * L * NKV * DH * 2
N_OUT, CTX, POOL = 500, 1250, 400_000
A = PARAMS * 2 / BW
B_ = CTX * KV_TOK / BW
B_MAX = POOL // CTX
B_HALF = A / B_
LAM_CAP = B_MAX / (N_OUT * (A + B_ * B_MAX))
LAM_INF = 1 / (N_OUT * B_)


def style(ax):
    for side in ("top",):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=FS_SMALL, colors=MUTED)
    ax.grid(True, color="#e4e4e1", lw=0.8)


def main() -> None:
    use_cjk_font()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.3))

    # 左：批大小 -> TPOT 与吞吐
    b = np.linspace(1, 420, 400)
    tpot = (A + B_ * b) * 1e3
    thr = b / (A + B_ * b)
    ax1.plot(b, tpot, color=DATA_EDGE, lw=2.2)
    ax1.set_xlabel("同批 Decode 的请求数 B", fontsize=FS_TEXT, color=INK)
    ax1.set_ylabel("步长，即 TPOT（毫秒）", fontsize=FS_TEXT, color=DATA_EDGE)
    ax1.set_xlim(0, 420)
    ax1.set_ylim(0, 28)
    style(ax1)
    ax1r = ax1.twinx()
    ax1r.plot(b, thr, color=WEIGHT_EDGE, lw=2.2)
    ax1r.axhline(1 / B_, color=WEIGHT_EDGE, lw=1.0, ls="--")
    # 标签放在 B = 98 与 B = 320 两条竖线之间，避免被竖线穿过
    ax1r.text(B_HALF + 8, 1 / B_ + 350, f"吞吐渐近线\n1/b = {1 / B_:,.0f} 词元/秒",
              fontsize=FS_SMALL, color=WEIGHT_EDGE, va="bottom", linespacing=1.3)
    ax1r.set_ylim(0, 24000)
    ax1r.set_ylabel("吞吐（词元/秒）", fontsize=FS_TEXT, color=WEIGHT_EDGE)
    ax1r.tick_params(labelsize=FS_SMALL, colors=MUTED)
    ax1r.spines["top"].set_visible(False)
    for x, text in ((B_HALF, f"B = a/b = {B_HALF:.0f}\n吞吐到渐近线的一半"),
                    (B_MAX, f"B = {B_MAX}\nKV 池装满")):
        ax1.axvline(x, color=ACCENT, lw=1.2, ls=":")
        ax1.text(x + 6, 1.2, text, fontsize=FS_SMALL, color=ACCENT, va="bottom",
                 linespacing=1.3)

    # 右：到达率 -> TPOT
    lam = np.linspace(0.1, LAM_CAP, 300)
    ax2.plot(lam, A / (1 - lam * N_OUT * B_) * 1e3, color=DATA_EDGE, lw=2.2)
    # 容量墙之后 B 封顶 B_MAX，步长停在 a + b·B_MAX，多出的请求进入等待队列
    tpot_cap = (A + B_ * B_MAX) * 1e3
    ax2.plot([LAM_CAP, 47], [tpot_cap, tpot_cap], color=DATA_EDGE, lw=1.4, ls="--")
    ax2.axvspan(LAM_CAP, 47, color="#ece9f7", zorder=0)
    ax2.axvline(LAM_CAP, color=ACCENT, lw=1.2, ls=":")
    ax2.axvline(LAM_INF, color=MUTED, lw=1.0, ls="--")
    ax2.text(LAM_CAP - 0.8, 29, f"KV 池装满\nλ = {LAM_CAP:.1f}", fontsize=FS_SMALL, color=ACCENT,
             ha="right", va="top", linespacing=1.3)
    ax2.text(LAM_CAP - 0.8, tpot_cap + 1.2, f"B 封顶 {B_MAX}，TPOT 停在 {tpot_cap:.1f} ms",
             fontsize=FS_SMALL, color=DATA_EDGE, ha="right", va="bottom")
    ax2.text(LAM_CAP + 0.5, 16.5, "等待队列\n无界增长，\nTTFT 随\n时间上升", fontsize=FS_SMALL,
             color=ACCENT, ha="left", va="top", linespacing=1.3)
    ax2.text(LAM_INF + 0.4, 29, f"带宽极限\n1/(Nb)\n= {LAM_INF:.1f}", fontsize=FS_SMALL,
             color=MUTED, ha="left", va="top", linespacing=1.3)
    for x in (5, 10, 20, 30):
        y = A / (1 - x * N_OUT * B_) * 1e3
        ax2.scatter([x], [y], s=42, color=DATA_EDGE, zorder=4)
        ax2.text(x - 0.4, y + 1.6, f"{y:.1f}", fontsize=FS_SMALL, color=INK, ha="right")
    ax2.set_xlabel("到达率 λ（请求/秒）", fontsize=FS_TEXT, color=INK)
    ax2.set_ylabel("TPOT（毫秒）", fontsize=FS_TEXT, color=DATA_EDGE)
    ax2.set_xlim(0, 47)
    ax2.set_ylim(0, 30)
    style(ax2)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout(w_pad=2.5)
    fig.savefig(OUTPUT, dpi=148, facecolor="white")
    print(f"已写入 {OUTPUT}  a={A * 1e3:.3f}ms b={B_ * 1e3:.4f}ms B_half={B_HALF:.0f} "
          f"B_max={B_MAX} lam_cap={LAM_CAP:.1f} lam_inf={LAM_INF:.1f}")


if __name__ == "__main__":
    main()
