"""生成图 2-1：缩放点积注意力权重热力图。

正文位置：02_attention/2.2_scaled_dot_product.md
输出：02_attention/_images/attention_heatmap.png

Q/K/V 与注意力权重的计算与正文那段可运行示例完全一致，此处一并带上，
使本脚本可独立运行。
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from _style import image_path, use_cjk_font

OUTPUT = image_path("02_attention", "attention_heatmap.png")


def compute_attention_weights() -> torch.Tensor:
    """复现 2.2 节示例中的注意力权重矩阵。"""
    Q = torch.tensor([[1, 0, 1, 0],
                      [0, 1, 0, 1],
                      [1, 1, 0, 0]], dtype=torch.float32)

    K = torch.tensor([[1, 1, 0, 0],
                      [0, 0, 1, 1],
                      [1, 0, 1, 0]], dtype=torch.float32)

    d_k = Q.size(-1)  # 4
    scores = torch.matmul(Q, K.transpose(-2, -1))
    scaled_scores = scores / math.sqrt(d_k)
    return F.softmax(scaled_scores, dim=-1)


def main() -> None:
    use_cjk_font()
    attn_weights = compute_attention_weights()

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(attn_weights.detach().numpy(), cmap="Reds", vmin=0, vmax=1)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(["键 0", "键 1", "键 2"])
    ax.set_yticklabels(["查询 0", "查询 1", "查询 2"])
    ax.set_xlabel("键位置")
    ax.set_ylabel("查询位置")
    ax.set_title("注意力权重热力图")

    # 在每个单元格中标注数值
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{attn_weights[i, j]:.3f}",
                    ha="center", va="center", fontsize=11)

    plt.colorbar(im, ax=ax, label="权重")
    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=150)
    print(f"已写入 {OUTPUT}")


if __name__ == "__main__":
    main()
