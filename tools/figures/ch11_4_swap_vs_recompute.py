"""生成 11.4 节的“换出再换入”与“丢弃后重算”耗时对比图。

正文位置：11_serving/11.4_kv_memory_management.md
输出：11_serving/_images/ch11_4_swap_vs_recompute.png

口径与正文算例一致：
- 重算 = 一次 Prefill，计算量 2·P·T + 2·T²·d·L + LM head（3.8.7 节的数法，P 为各层矩阵参数）；
  算力取 10.1 节的 H100 BF16 稠密峰值约 990 TFLOPs/s，利用率假设 0.5。
- 往返换出 = 2 × 序列的 KV 字节数 ÷ 单向带宽；PCIe Gen5 x16 取 64 GB/s，Gen4 x16 取 32 GB/s，
  都是标称值，没有计入小块传输的额外开销。
左：Llama 3 8B（GQA，n_kv = 8）。右：GPT-3 6.7B（MHA，n_kv = 32），上下文上限 2,048。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _diagram import ACCENT, DATA_EDGE, FS_SMALL, FS_TEXT, INK, MUTED, NEW_EDGE, WEIGHT_EDGE
from _style import image_path, use_cjk_font

OUTPUT = image_path("11_serving", "ch11_4_swap_vs_recompute.png")

FLOPS = 9.9e14 * 0.5
GEN5, GEN4 = 64e9, 32e9

LLAMA3_8B = dict(title="Llama 3 8B（GQA，每词元 KV 128 KiB）", L=32, d=4096, n_kv=8, d_h=128,
                 p_mat=32 * (2 * 4096 * 4096 + 2 * 4096 * 1024 + 3 * 4096 * 14336),
                 vocab=128256, t_max=32000)
GPT3_6B7 = dict(title="GPT-3 6.7B（MHA，每词元 KV 512 KiB）", L=32, d=4096, n_kv=32, d_h=128,
                p_mat=32 * 12 * 4096 * 4096, vocab=50257, t_max=2048)


def kv_bytes(cfg) -> int:
    return 2 * cfg["n_kv"] * cfg["d_h"] * cfg["L"] * 2


def recompute_ms(cfg, t):
    flops = 2 * cfg["p_mat"] * t + 2 * t * t * cfg["d"] * cfg["L"] + 2 * cfg["d"] * cfg["vocab"]
    return flops / FLOPS * 1e3


def swap_ms(cfg, t, bandwidth):
    return 2 * t * kv_bytes(cfg) / bandwidth * 1e3


def draw(ax, cfg) -> None:
    t = np.linspace(0, cfg["t_max"], 200)
    ax.plot(t, recompute_ms(cfg, t), color=WEIGHT_EDGE, lw=2.4, label="丢弃后重算（一次 Prefill）")
    ax.plot(t, swap_ms(cfg, t, GEN4), color=DATA_EDGE, lw=2.4, ls="--",
            label="换出再换入，PCIe Gen4 x16")
    ax.plot(t, swap_ms(cfg, t, GEN5), color=NEW_EDGE, lw=2.4,
            label="换出再换入，PCIe Gen5 x16")
    ax.set_title(cfg["title"], fontsize=FS_TEXT, color=INK)
    ax.set_xlabel("被抢占序列的长度（词元）", fontsize=FS_SMALL, color=INK)
    ax.set_xlim(0, cfg["t_max"])
    ax.set_ylim(0, None)
    ax.tick_params(labelsize=FS_SMALL, colors=MUTED)
    ax.grid(True, color="#e4e4e1", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    breakeven = 2 * kv_bytes(cfg) / (2 * cfg["p_mat"] / FLOPS) / 1e9
    ax.text(0.04, 0.62, f"持平带宽约 {breakeven:.0f} GB/s", transform=ax.transAxes,
            fontsize=FS_SMALL, color=ACCENT, ha="left")


def main() -> None:
    use_cjk_font()
    fig, axes = plt.subplots(1, 2, figsize=(9.9, 4.4))
    draw(axes[0], LLAMA3_8B)
    draw(axes[1], GPT3_6B7)
    axes[0].set_ylabel("恢复这条序列的 KV 所需时间（毫秒）", fontsize=FS_SMALL, color=INK)
    axes[0].legend(fontsize=FS_SMALL, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150, facecolor="white")
    print(f"已写入 {OUTPUT}")


if __name__ == "__main__":
    main()
