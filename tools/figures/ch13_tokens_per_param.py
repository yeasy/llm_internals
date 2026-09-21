"""生成 13.2 节插图：各代模型的参数量与训练词元数，以及它们的 D/N。

正文位置：13_decoder_models/13.2_llama.md
输出：13_decoder_models/_images/ch13_tokens_per_param.png

横轴是参数量 N，纵轴是训练词元数 D，两轴都取对数。斜的虚线是 D/N 等值线：
落在同一条线上的模型，每个参数分到的词元数相同。Chinchilla 的约 20 词元/参数画成
加粗的那条，用来读出各代模型离“算力最优”有多远。点线是等算力线 6ND = C。

数值来源：
- GPT-3    : GPT-3 论文表 D.1（174,600M 参数、300B 词元、3.14E+23 FLOPs）
- LLaMA 1  : LLaMA 论文表 2（6.7B/1.0T、65.2B/1.4T）
- Llama 2  : Llama 2 论文正文“We trained on 2 trillion tokens”
- Llama 3  : Llama 3 论文（405B 参数、15.6T 词元、3.8e25 FLOPs；语料约 15T）
- DeepSeek : DeepSeek-V3 技术报告（37B 激活 / 671B 总量、14.8T 词元）
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _diagram import (ACCENT, DATA_EDGE, FS_NAME, FS_SMALL, FS_TEXT, INK,
                      MUTED, NEW_EDGE, WEIGHT_EDGE)
from _style import image_path, use_cjk_font

OUTPUT = image_path("13_decoder_models", "ch13_tokens_per_param.png")

# 标签压在等值线上会被线划穿，统一垫一层白底。
LABEL_BBOX = dict(boxstyle="round,pad=0.18", facecolor="white",
                  edgecolor="none", alpha=0.88)

# (名字, N, D, 颜色, 文字相对点的像素偏移, 水平对齐, 垂直对齐)
MODELS = [
    ("GPT-3 175B", 174.6e9, 300e9, WEIGHT_EDGE, (-12, 6), "right", "bottom"),
    ("LLaMA 1 7B", 6.7e9, 1.0e12, DATA_EDGE, (-12, 0), "right", "center"),
    ("LLaMA 1 65B", 65.2e9, 1.4e12, DATA_EDGE, (-14, -12), "right", "top"),
    ("Llama 2 70B", 70e9, 2.0e12, DATA_EDGE, (14, 4), "left", "bottom"),
    ("Llama 3 8B", 8.03e9, 15e12, NEW_EDGE, (-12, 0), "right", "center"),
    ("Llama 3.1 405B", 405.85e9, 15.6e12, NEW_EDGE, (0, 16), "center", "bottom"),
    ("DeepSeek-V3（激活 37B）", 37e9, 14.8e12, ACCENT, (0, 16), "center", "bottom"),
]


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(10.0, 6.4))

    xlo, xhi = 2e9, 3.0e12
    ylo, yhi = 2e11, 1.1e14
    n = np.logspace(np.log10(xlo), np.log10(xhi), 200)

    # D/N 等值线；标签放在线与右边界的交点附近，保证落在画面内。
    for ratio, bold, x_at in ((20, True, 1.0e10), (200, False, 2.4e11), (2000, False, 2.0e10)):
        ax.plot(n, ratio * n, ls="--", lw=2.2 if bold else 1.0,
                color=ACCENT if bold else MUTED, alpha=1.0 if bold else 0.6, zorder=1)
        text = "D/N = 20（Chinchilla）" if bold else f"D/N = {ratio:,}"
        ax.text(x_at, ratio * x_at * 1.06, text, fontsize=FS_SMALL,
                color=ACCENT if bold else MUTED, rotation=30,
                rotation_mode="anchor", ha="left", va="bottom", zorder=4,
                bbox=LABEL_BBOX)

    # 等算力线 6ND = C
    for c, tag, x_at in ((3.14e23, "6ND = 3.1e23（GPT-3）", 8.0e9),
                         (3.8e25, "6ND = 3.8e25（Llama 3.1 405B）", 6.0e11)):
        ax.plot(n, c / (6 * n), ls=":", lw=1.3, color=INK, alpha=0.45, zorder=1)
        ax.text(x_at, c / (6 * x_at) * 1.05, tag, fontsize=FS_SMALL,
                color=INK, alpha=0.75, rotation=-30,
                rotation_mode="anchor", ha="left", va="bottom", zorder=4,
                bbox=LABEL_BBOX)

    for name, N, D, color, off, ha, va in MODELS:
        ax.scatter([N], [D], s=110, color=color, edgecolor="white",
                   linewidth=1.5, zorder=5)
        ax.annotate(f"{name}\nD/N = {D / N:,.0f}" if D / N >= 10
                    else f"{name}\nD/N = {D / N:.1f}",
                    xy=(N, D), xytext=off, textcoords="offset points",
                    fontsize=FS_SMALL, color=INK, ha=ha, va=va,
                    linespacing=1.35, zorder=6, bbox=LABEL_BBOX)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("参数量 N（对数坐标）", fontsize=FS_TEXT)
    ax.set_ylabel("训练词元数 D（对数坐标）", fontsize=FS_TEXT)
    ax.set_title("每个参数分到多少词元：GPT-3 是 1.7，Llama 3 8B 是 1,868",
                 fontsize=FS_NAME, pad=14)
    ax.grid(True, which="major", ls="-", lw=0.4, color="#dddddc", zorder=0)
    ax.tick_params(labelsize=FS_SMALL)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=125, facecolor="white")
    print(f"已写入 {OUTPUT}")


if __name__ == "__main__":
    main()
