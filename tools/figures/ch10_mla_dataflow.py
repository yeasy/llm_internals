"""生成 10.2 节插图：MLA 一个词元的数据流，缓存里到底存了什么。

正文位置：10_inference_optimization/10.2_kv_cache.md
输出：10_inference_optimization/_images/ch10_mla_dataflow.png

形状取 DeepSeek-V3 的公开配置：hidden_size 7168，kv_lora_rank 512，
qk_rope_head_dim 64，qk_nope_head_dim 128，v_head_dim 128，num_attention_heads 128，
q_lora_rank 1536。蓝 = 数据，橙 = 权重，青绿 = 写入缓存的两个向量，紫 = 强调。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import ACCENT, FS_TEXT, INK, MUTED, arrow, box, finish, label
from _style import image_path, use_cjk_font

OUTPUT = image_path("10_inference_optimization", "ch10_mla_dataflow.png")
FS = 9.5


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(10.0, 4.6))

    # 输入
    box(ax, 0.2, 5.0, 3.3, 1.5, "新词元的表示\n$x_t$  [1, 7168]", "data", fontsize=FS)

    # 上路：压缩 KV
    box(ax, 4.7, 8.0, 4.0, 1.5, "$W^{DKV}$\n[7168, 512]", "weight", fontsize=FS)
    box(ax, 9.7, 8.0, 3.8, 1.5, "$c_t$  [1, 512]\n不带位置信息", "new", fontsize=FS)
    arrow(ax, (1.85, 6.5), (4.7, 8.75), rad=-0.15)
    arrow(ax, (8.7, 8.75), (9.7, 8.75))

    # 中路：解耦的 RoPE 键
    box(ax, 4.7, 5.0, 4.0, 1.5, "$W^{KR}$，再做 RoPE\n[7168, 64]", "weight", fontsize=FS)
    box(ax, 9.7, 5.0, 3.8, 1.5, "$k^R_t$  [1, 64]\n各头共用，带位置", "new", fontsize=FS)
    arrow(ax, (3.5, 5.75), (4.7, 5.75))
    arrow(ax, (8.7, 5.75), (9.7, 5.75))

    # 缓存
    box(ax, 14.6, 4.7, 4.9, 5.0, "", "neutral")
    label(ax, 17.05, 9.15, "本层 KV 缓存", fontsize=FS_TEXT, bold=True)
    label(ax, 17.05, 7.85, "每个词元只存\n$c_j$ 与 $k^R_j$", fontsize=FS)
    label(ax, 17.05, 6.3, "512 + 64 = 576 个数\nBF16 下 1,152 字节", fontsize=FS, color=ACCENT)
    label(ax, 17.05, 5.15, "形状 [t, 576]", fontsize=FS, color=MUTED)
    arrow(ax, (13.5, 8.75), (14.6, 8.2))
    arrow(ax, (13.5, 5.75), (14.6, 6.1))

    # 对照
    box(ax, 20.1, 4.7, 4.6, 5.0, "", "plain")
    label(ax, 22.4, 9.15, "对照：存完整 K、V", fontsize=FS, bold=True)
    label(ax, 22.4, 7.6, "2 × 128 头 × 128\n= 32,768 个数", fontsize=FS)
    label(ax, 22.4, 5.8, "576 ÷ 32,768\n约 1/57", fontsize=FS, color=ACCENT)

    # 下路：查询
    box(ax, 4.7, 1.3, 4.0, 2.2, "$W^{DQ}$、$W^{UQ}$、$W^{QR}$\n7168 → 1536\n→ 每头 128 + 64", "weight",
        fontsize=FS)
    box(ax, 9.7, 1.5, 3.8, 1.8, "$q^C_t$ [128 头, 128]\n$q^R_t$ [128 头, 64]", "data", fontsize=FS)
    arrow(ax, (1.85, 5.0), (4.7, 2.4), rad=0.15)
    arrow(ax, (8.7, 2.4), (9.7, 2.4))

    # 打分
    box(ax, 14.6, 0.7, 10.1, 3.3,
        "每个头对缓存里的位置 j 打分：\n"
        r"$q^C_t\,(W^{UK})^{\top} c_j^{\top} \;+\; q^R_t\,(k^R_j)^{\top}$" "\n"
        r"$q^C_t (W^{UK})^{\top}$ 先算成 [1, 512]，" "\n" r"$c_j$ 不必展开成 128 个头的 K",
        "plain", fontsize=FS)
    arrow(ax, (13.5, 2.4), (14.6, 2.4))
    arrow(ax, (17.05, 4.7), (17.05, 4.0))

    label(ax, 12.4, 10.6, "MLA：每层为一个词元写入缓存的只有两个向量（形状取 DeepSeek-V3）",
          fontsize=FS_TEXT, bold=True)
    label(ax, 12.4, 0.2, "橙色是权重，蓝色是数据，青绿色是写入缓存的量；V 的上投影 $W^{UV}$ 同理并入输出投影 $W^O$",
          fontsize=FS, color=MUTED)
    _ = INK
    finish(fig, ax, OUTPUT, xlim=(0, 24.9), ylim=(-0.2, 11.1))


if __name__ == "__main__":
    main()
