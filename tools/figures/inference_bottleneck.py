"""生成图 3-13：Prefill 与 Decode 各自落在哪一种瓶颈里。

正文位置：03_components/3.8_gpt_inference_flow.md
输出：03_components/_images/inference_bottleneck.png

横轴是“每从显存读 1 字节，做多少次浮点运算”（对数刻度）。数值按正文表 3-14 的
口径算出：GPT-3 Small，FP16 权重，上下文 1,000 个词元；访存只数权重和 KV 缓存。
分界线取 10.1 节给出的 H100 拐点（约 295 次/字节）。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import (ACCENT, DATA_EDGE, FS_NAME, FS_SMALL, FS_TEXT, INK, MUTED,
                      NEW_EDGE, WEIGHT_EDGE)
from _style import image_path, use_cjk_font

OUTPUT = image_path("03_components", "inference_bottleneck.png")

D, LAYERS, VOCAB, CTX = 768, 12, 50257, 1000
PARAMS = 12 * D * D * LAYERS + VOCAB * D + 2048 * D
WEIGHT_BYTES = PARAMS * 2
KV_BYTES_PER_TOKEN = 2 * 12 * 64 * LAYERS * 2
LM_HEAD = 2 * D * VOCAB
PREFILL_FLOPS = 24 * CTX * D * D * LAYERS + 2 * CTX * CTX * D * LAYERS + LM_HEAD
DECODE_FLOPS = 24 * D * D * LAYERS + 4 * CTX * D * LAYERS + LM_HEAD
RIDGE = 295


def intensity_decode(batch: int) -> float:
    return batch * DECODE_FLOPS / (WEIGHT_BYTES + batch * KV_BYTES_PER_TOKEN * CTX)


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    lo, hi = 0.4, 3000
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1)

    # 两个区域
    ax.axvspan(lo, RIDGE, color="#f4f3ef", zorder=0)
    ax.axvspan(RIDGE, hi, color="#e6e5df", zorder=0)
    ax.axvline(RIDGE, color=INK, lw=1.4, zorder=2)
    ax.text(RIDGE, 0.97, f"分界线：约 {RIDGE} 次/字节（H100）", ha="center", va="top",
            fontsize=FS_SMALL, color=INK, bbox=dict(facecolor="white", edgecolor="none", pad=2.5))
    ax.text(11, 0.84, "显存带宽受限：运算单元在等数据", ha="center", va="center",
            fontsize=FS_NAME, color=INK, fontweight="bold")
    ax.text(930, 0.84, "算力受限", ha="center", va="center", fontsize=FS_NAME, color=INK,
            fontweight="bold")

    # 基线与各个工作点
    y = 0.42
    ax.plot([lo, hi], [y, y], color="#b9b8b2", lw=1.0, zorder=1)
    d1, d32 = intensity_decode(1), intensity_decode(32)
    pre = PREFILL_FLOPS / WEIGHT_BYTES
    points = [
        (d1, DATA_EDGE, f"Decode，单个请求\n约 {d1:.0f} 次/字节", "center"),
        (d32, DATA_EDGE, f"Decode，32 个请求合批\n约 {d32:.0f} 次/字节", "center"),
        (pre, DATA_EDGE, f"Prefill，1,000 个词元\n约 {round(pre, -1):.0f} 次/字节（只数权重）", "center"),
    ]
    for x, color, text, ha in points:
        ax.scatter([x], [y], s=170, color=color, edgecolor="white", linewidth=2, zorder=4)
        ax.text(x, y - 0.11, text, ha=ha, va="top", fontsize=FS_TEXT, color=INK,
                linespacing=1.4)

    # 优化把 Decode 往右推
    ax.annotate("", xy=(d32 * 0.82, y + 0.13), xytext=(d1 * 1.25, y + 0.13),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.6))
    ax.text((d1 * d32) ** 0.5, y + 0.17, "批处理：读一次权重，服务多个请求",
            ha="center", va="bottom", fontsize=FS_SMALL, color=ACCENT)
    ax.annotate("", xy=(RIDGE * 0.8, y + 0.13), xytext=(d32 * 1.3, y + 0.13),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.6, linestyle="--"))
    ax.text((d32 * RIDGE) ** 0.5, y + 0.17, "再往右要靠：量化、GQA、投机解码",
            ha="center", va="bottom", fontsize=FS_SMALL, color=ACCENT)

    ax.set_yticks([])
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1,000"], fontsize=FS_SMALL, color=MUTED)
    ax.set_xlabel("每从显存读 1 字节，做多少次浮点运算（对数刻度）", fontsize=FS_TEXT, color=INK,
                  labelpad=8)
    ax.tick_params(axis="x", which="both", length=0)
    ax.minorticks_off()
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#b9b8b2")
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"已写入 {OUTPUT}  decode1={d1:.2f} decode32={d32:.2f} prefill={pre:.0f}")


if __name__ == "__main__":
    main()
