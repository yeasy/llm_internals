"""生成图 4-3：RoPE 的波长谱、临界维度，以及三种缩放各自改动了哪些频率。

正文位置：04_position_encoding/4.3_rope.md
输出：04_position_encoding/_images/ch04_rope_wavelength.png

左图按 d_h = 128 画出 64 对频率的波长 lambda_i = 2*pi*b^(2i/d_h)，并用一条水平线
标出训练长度；线以上的那些对在整个训练长度内转不满一圈，承担的是绝对位置。
右图画每种缩放把第 i 对的波长拉长了多少倍，可直接读出它改动了哪一段频率。

数值与 scratchpad 里的 ch04_numbers.py 同源，可互相复核。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _diagram import ACCENT, DATA_EDGE, INK, MUTED, NEW_EDGE, WEIGHT_EDGE
from _style import image_path, use_cjk_font

OUTPUT = image_path("04_position_encoding", "ch04_rope_wavelength.png")

D_H = 128          # 每个注意力头的宽度，Llama 系为 128
N_PAIR = D_H // 2  # 64 对二维子空间
L_TRAIN = 8192     # Llama 3 预训练上下文长度
FACTOR = 8.0       # 扩展倍数 s


def wavelengths(base: float) -> np.ndarray:
    i = np.arange(N_PAIR)
    return 2 * np.pi * base ** (2 * i / D_H)


def llama3_wavelengths(base: float, factor: float = FACTOR,
                       low_freq_factor: float = 1.0,
                       high_freq_factor: float = 4.0,
                       old_context: int = 8192) -> np.ndarray:
    """复刻 meta-llama/llama-models 里 apply_scaling() 的分段规则。"""
    wl = wavelengths(base)
    freq = 1.0 / wl
    low_wl, high_wl = old_context / low_freq_factor, old_context / high_freq_factor
    new_freq = np.where(wl > low_wl, freq / factor, freq)
    smooth = (old_context / wl - low_freq_factor) / (high_freq_factor - low_freq_factor)
    mid = (wl >= high_wl) & (wl <= low_wl)
    new_freq = np.where(mid, (1 - smooth) * new_freq / factor + smooth * new_freq, new_freq)
    return 1.0 / new_freq


def main() -> None:
    use_cjk_font()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    idx = np.arange(N_PAIR)

    # ---- 左图：两个 base 的波长谱与临界维度 ----
    ax = axes[0]
    wl_small, wl_big = wavelengths(1e4), wavelengths(5e5)
    top = wl_big[-1] * 40
    ax.axhspan(L_TRAIN, top, color=ACCENT, alpha=0.07)
    ax.semilogy(idx, wl_small, color=DATA_EDGE, lw=2, label="base = 10,000")
    ax.semilogy(idx, wl_big, color=WEIGHT_EDGE, lw=2, label="base = 500,000")
    ax.axhline(L_TRAIN, color=ACCENT, ls="--", lw=1.5)

    n_small = int((wl_small > L_TRAIN).sum())
    n_big = int((wl_big > L_TRAIN).sum())
    ax.text(1.5, top / 3,
            "线以上：整个训练长度内转不满一圈，\n这些对实际承担绝对位置\n"
            f"base = 10,000 有 {n_small}/64 对\n"
            f"base = 500,000 有 {n_big}/64 对",
            color=ACCENT, fontsize=9.5, linespacing=1.5, va="top")
    ax.text(62, L_TRAIN / 2.6, "训练长度 8,192", color=ACCENT, fontsize=10,
            ha="right", va="top")

    ax.set_xlabel("频率对编号 i（共 64 对，$d_h$ = 128）")
    ax.set_ylabel("波长 $\\lambda_i$（词元）")
    ax.set_title("波长谱与临界维度", fontsize=12, color=INK)
    ax.set_ylim(3, top)
    ax.set_xlim(-1.5, N_PAIR + 0.5)
    ax.legend(fontsize=9.5, loc="lower right")
    ax.grid(True, alpha=0.25)

    # ---- 右图：s = 8 的三种缩放各把哪一段频率拉长了 ----
    ax = axes[1]
    base = 5e5
    wl = wavelengths(base)
    b_ntk = base * FACTOR ** (D_H / (D_H - 2))
    ratio_ntk = wavelengths(b_ntk) / wl
    ratio_l3 = llama3_wavelengths(base) / wl

    ax.axhline(1.0, color=MUTED, lw=1.2, ls=":")
    ax.plot(idx, np.full(N_PAIR, FACTOR), color=DATA_EDGE, lw=2,
            label="PI：所有对一律乘 8")
    ax.plot(idx, ratio_ntk, color=WEIGHT_EDGE, lw=2,
            label=f"NTK 感知：base 改为 {b_ntk / 1e6:.2f}e6")
    ax.plot(idx, ratio_l3, color=NEW_EDGE, lw=2, label="Llama 3.1 分段")

    keep = int((ratio_l3 < 1.001).sum())
    ax.axvspan(-1.5, keep - 0.5, color=NEW_EDGE, alpha=0.08)
    ax.text(0.5, 7.4, f"第 0–{keep - 1} 对原样保留\n（波长 < 2,048）",
            color=NEW_EDGE, fontsize=9.5, linespacing=1.5, va="top")
    ax.text(63, 1.35, "倍数 = 1 即该对完全不动", color=MUTED, fontsize=9.5, ha="right")

    ax.set_xlabel("频率对编号 i")
    ax.set_ylabel("波长被拉长的倍数")
    ax.set_title("s = 8 时三种缩放改动了哪一段频率", fontsize=12, color=INK)
    ax.set_ylim(0, 9.3)
    ax.set_xlim(-1.5, N_PAIR + 0.5)
    ax.legend(fontsize=9, loc="center left")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150, facecolor="white")
    print(f"已写入 {OUTPUT}")
    print(f"base 1e4 越线 {n_small}/64；base 5e5 越线 {n_big}/64")
    print(f"NTK 感知 b' = {b_ntk:,.0f}；最高频对倍数 {ratio_ntk[0]:.3f}，最低频对倍数 {ratio_ntk[-1]:.3f}")
    print(f"Llama 3.1 分段：完全不动的对数 {keep}/64，倍数为 8 的对数 {int((ratio_l3 > 7.999).sum())}/64")


if __name__ == "__main__":
    main()
