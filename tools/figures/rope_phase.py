"""生成图 4-2：RoPE 注意力分数随相对距离的相位变化示例。

正文位置：04_position_encoding/4.3_rope.md
输出：04_position_encoding/_images/rope_decay.png

`apply_rope()` 与分数序列的计算与正文保留的那段代码一致（含
`torch.manual_seed(42)`，故曲线可复现），此处一并带上，使本脚本可独立运行。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from _style import image_path, use_cjk_font

OUTPUT = image_path("04_position_encoding", "rope_decay.png")

D = 64            # 向量维度
MAX_DIST = 512    # 最大相对距离
THETA_BASE = 10000.0

# 频率参数
freqs = 1.0 / (THETA_BASE ** (torch.arange(0, D, 2).float() / D))


def apply_rope(x: torch.Tensor, pos: float) -> torch.Tensor:
    """对向量施加 RoPE 旋转。"""
    x_pairs = x.view(-1, 2)
    angles = pos * freqs
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    x_rot = torch.stack([
        x_pairs[:, 0] * cos_a - x_pairs[:, 1] * sin_a,
        x_pairs[:, 0] * sin_a + x_pairs[:, 1] * cos_a
    ], dim=-1)
    return x_rot.flatten()


def scores_by_distance() -> tuple[torch.Tensor, list[float]]:
    """固定查询在位置 0，扫描键所在的相对距离，返回（距离, 点积分数）。"""
    torch.manual_seed(42)
    q = torch.randn(D)
    k = torch.randn(D)

    distances = torch.arange(0, MAX_DIST)
    q_pos0 = apply_rope(q, 0)  # 查询固定在位置 0
    scores = [torch.dot(q_pos0, apply_rope(k, float(dist))).item()
              for dist in distances]
    return distances, scores


def main() -> None:
    use_cjk_font()
    distances, scores = scores_by_distance()

    plt.figure(figsize=(10, 4))
    plt.plot(distances.numpy(), scores, linewidth=0.8, alpha=0.8)
    plt.xlabel("相对距离")
    plt.ylabel("注意力分数（点积）")
    plt.title("RoPE 注意力分数随相对距离的相位变化示例")
    plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=150)
    print(f"已写入 {OUTPUT}")


if __name__ == "__main__":
    main()
