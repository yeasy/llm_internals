"""生成图 2-4：因果掩码前后的注意力权重热力图对比。

正文位置：02_attention/2.4_self_cross_causal.md
输出：02_attention/_images/causal_mask_heatmap.png

分数与掩码的计算与正文那段可运行示例完全一致（含 `torch.manual_seed(42)`，
故每次运行结果可复现），此处一并带上，使本脚本可独立运行。
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from _style import image_path, use_cjk_font

OUTPUT = image_path("02_attention", "causal_mask_heatmap.png")

SEQ_LEN, D_K = 4, 8


def compute_weights() -> tuple[torch.Tensor, torch.Tensor]:
    """返回（无掩码权重, 因果掩码后权重）。"""
    torch.manual_seed(42)
    Q = torch.randn(SEQ_LEN, D_K)
    K = torch.randn(SEQ_LEN, D_K)
    torch.randn(SEQ_LEN, D_K)  # V：与正文同序消耗随机数，保证分数一致

    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D_K)

    mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1).bool()
    scores_masked = scores.masked_fill(mask, float("-inf"))

    return F.softmax(scores, dim=-1), F.softmax(scores_masked, dim=-1)


def _draw(ax, weights: torch.Tensor, title: str, labels: list[str]):
    im = ax.imshow(weights.detach().numpy(), cmap="Reds", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks(range(SEQ_LEN))
    ax.set_yticks(range(SEQ_LEN))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(SEQ_LEN):
        for j in range(SEQ_LEN):
            ax.text(j, i, f"{weights[i, j]:.2f}",
                    ha="center", va="center", fontsize=8)
    return im


def main() -> None:
    use_cjk_font()
    attn_no_mask, attn_weights = compute_weights()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = [f"位置 {i}" for i in range(SEQ_LEN)]

    _draw(axes[0], attn_no_mask, "（a）无掩码的注意力权重", labels)
    im1 = _draw(axes[1], attn_weights, "（b）因果掩码后的注意力权重", labels)

    plt.colorbar(im1, ax=axes.ravel().tolist(), label="权重", shrink=0.8)
    plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    print(f"已写入 {OUTPUT}")


if __name__ == "__main__":
    main()
