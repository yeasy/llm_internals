"""生成图 4-1：正弦位置编码的频率分解可视化。

正文位置：04_position_encoding/4.1_sinusoidal.md
输出：04_position_encoding/_images/sinusoidal_pe_visualization.png

PE 矩阵的构造与正文保留的那段代码一致，此处一并带上，使本脚本可独立运行。
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import torch

from _style import image_path, use_cjk_font

OUTPUT = image_path("04_position_encoding", "sinusoidal_pe_visualization.png")

D_MODEL = 64   # 编码维度
MAX_POS = 100  # 位置数量


def sinusoidal_pe(d_model: int = D_MODEL, max_pos: int = MAX_POS) -> torch.Tensor:
    """构造正弦位置编码矩阵（与 4.1 节正文同一段代码）。"""
    pe = torch.zeros(max_pos, d_model)
    position = torch.arange(0, max_pos).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                         -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
    pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
    return pe


def main() -> None:
    use_cjk_font()
    pe = sinusoidal_pe()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：位置编码热力图
    im = axes[0].imshow(pe.numpy().T, aspect="auto", cmap="RdBu_r",
                        origin="lower")
    axes[0].set_xlabel("位置 (pos)")
    axes[0].set_ylabel("维度 (i)")
    axes[0].set_title("正弦位置编码热力图（PE 矩阵）")
    plt.colorbar(im, ax=axes[0])

    # 右图：选取 4 个不同频率通道的波形
    channels = [0, 10, 20, 30]  # 从高频到低频
    for ch in channels:
        freq = 1.0 / (10000 ** (ch / D_MODEL))
        axes[1].plot(pe[:, ch].numpy(),
                     label=f"维度 {ch}（频率 {freq:.4f}）")
    axes[1].set_xlabel("位置 (pos)")
    axes[1].set_ylabel("编码值")
    axes[1].set_title("不同频率通道的波形对比")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=150)
    print(f"已写入 {OUTPUT}")


if __name__ == "__main__":
    main()
