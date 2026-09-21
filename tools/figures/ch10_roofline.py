"""生成 10.1 节插图：H100 的 Roofline 与 Llama 3 8B 的四个工作点。

正文位置：10_inference_optimization/10.1_bottleneck.md
输出：10_inference_optimization/_images/ch10_roofline.png

口径与正文 10.1.2 一致：Llama 3 8B，BF16 权重与 KV；H100 峰值算力 990 TFLOPs/s、
显存带宽 3.35 TB/s，拐点约 295 FLOPs/Byte。Decode 的上下文取 1,000 个词元，
算术强度 = B × (2N + 4tdL) ÷ (权重字节 + B × t × 每词元 KV 字节)；
空心点是“只数权重”的强度（等于 B），用来显示 KV 读取把工作点往左拖了多少。
Prefill 取 T = 512，只数权重，强度等于 T。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _diagram import ACCENT, DATA_EDGE, FS_SMALL, FS_TEXT, INK, MUTED, NEW_EDGE, WEIGHT_EDGE
from _style import image_path, use_cjk_font

OUTPUT = image_path("10_inference_optimization", "ch10_roofline.png")

L, D, NH, DH, NKV, DFF, NVOCAB = 32, 4096, 32, 128, 8, 14336, 128256
PARAMS = L * (2 * D * NH * DH + 2 * D * NKV * DH + 3 * D * DFF + 2 * D) + 2 * NVOCAB * D + D
WEIGHT_BYTES = PARAMS * 2
KV_TOK = 2 * L * NKV * DH * 2
PEAK_FLOPS, PEAK_BW = 990e12, 3.35e12
RIDGE = PEAK_FLOPS / PEAK_BW
CTX = 1000


def decode_intensity(batch: int) -> float:
    flops = batch * (2 * PARAMS + 4 * CTX * D * L)
    return flops / (WEIGHT_BYTES + batch * CTX * KV_TOK)


def roof(intensity):
    return np.minimum(intensity * PEAK_BW, PEAK_FLOPS) / 1e12


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    lo, hi = 0.5, 4000
    x = np.logspace(np.log10(lo), np.log10(hi), 400)
    ax.plot(x, roof(x), color=INK, lw=2.4, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(1, 2500)

    ax.axvline(RIDGE, color=MUTED, lw=1.0, ls="--", zorder=1)
    ax.text(RIDGE * 1.08, 1.35, f"拐点 约 {int(RIDGE)}", fontsize=FS_SMALL, color=MUTED,
            ha="left", va="bottom")
    ax.text(4.2, 40, "斜线段：带宽受限\n上界 = 强度 × 3.35 TB/s", fontsize=FS_TEXT, color=INK,
            ha="center", va="center", rotation=32, linespacing=1.4)
    ax.text(1300, 45, "水平段：算力受限\n上界 = 990 TFLOPs/s", fontsize=FS_TEXT, color=INK,
            ha="center", va="center", linespacing=1.4)

    points = [
        (1, "Decode B = 1", (34, -14)),
        (32, "Decode B = 32", (36, -22)),
        (256, "Decode B = 256", (-6, -24)),
    ]
    for batch, name, offset in points:
        real = decode_intensity(batch)
        if batch > 1:
            ax.scatter([batch], [roof(batch)], s=90, facecolor="white", edgecolor=DATA_EDGE,
                       linewidth=1.8, zorder=4)
            ax.annotate("", xy=(real * 1.12, roof(real) * 1.12), xytext=(batch * 0.9, roof(batch) * 0.9),
                        arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.4), zorder=4)
        ax.scatter([real], [roof(real)], s=110, color=DATA_EDGE, edgecolor="white", linewidth=1.6,
                   zorder=5)
        ax.annotate(f"{name}\n强度 {real:.0f}" if real >= 10 else f"{name}\n强度 {real:.1f}",
                    xy=(real, roof(real)), xytext=offset, textcoords="offset points",
                    fontsize=FS_SMALL, color=INK, ha="left", va="top", linespacing=1.35,
                    bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none"))

    pre = 512.0
    ax.scatter([pre], [roof(pre)], s=110, color=WEIGHT_EDGE, edgecolor="white", linewidth=1.6, zorder=5)
    ax.annotate("Prefill T = 512\n强度 512", xy=(pre, roof(pre)), xytext=(10, -12),
                textcoords="offset points", fontsize=FS_SMALL, color=INK, ha="left", va="top",
                linespacing=1.35)

    ax.scatter([0.9], [900], s=90, facecolor="white", edgecolor=DATA_EDGE, linewidth=1.8, zorder=4)
    ax.text(1.15, 900, "只数权重（强度 = B）", fontsize=FS_SMALL, color=INK, va="center")
    ax.scatter([0.9], [480], s=110, color=DATA_EDGE, edgecolor="white", linewidth=1.6, zorder=4)
    ax.text(1.15, 480, "计入 KV 读取（上下文 1,000）", fontsize=FS_SMALL, color=INK, va="center")
    ax.annotate("", xy=(0.75, 250), xytext=(1.08, 250),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.4))
    ax.text(1.15, 250, "KV 读取把工作点拖向左下", fontsize=FS_SMALL, color=ACCENT, va="center")

    ax.set_xlabel("算术强度：每从显存读 1 字节做多少次浮点运算（FLOPs/Byte）", fontsize=FS_TEXT, color=INK)
    ax.set_ylabel("可达到的算力（TFLOPs/s）", fontsize=FS_TEXT, color=INK)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1,000"])
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels(["1", "10", "100", "1,000"])
    ax.tick_params(labelsize=FS_SMALL, colors=MUTED)
    ax.grid(True, which="major", color="#e4e4e1", lw=0.8)
    ax.minorticks_off()
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"已写入 {OUTPUT}  ridge={RIDGE:.1f} "
          f"decode={[round(decode_intensity(b), 2) for b in (1, 32, 256)]}")
    _ = NEW_EDGE


if __name__ == "__main__":
    main()
