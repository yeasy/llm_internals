"""生成 13.2 节插图：GPT-2 块与 Llama 块的逐项差异。

正文位置：13_decoder_models/13.2_llama.md
输出：13_decoder_models/_images/ch13_block_diff.png

左边是 GPT-2 式的块（取值按 GPT-2 论文与 openai-community/gpt2-xl 的 config.json；
GPT-2 已是 Pre-LN，归一化放在子层之前），右边是 Llama 3 8B 的块（Llama 3 论文表 3
与模型 config.json）。

箭头链只串真正的执行顺序：输入 -> 归一化 -> 自注意力 -> 归一化 -> MLP（两侧同为
Pre-Norm，归一化在子层之前）。位置编码与偏置项是两条属性而非流程步骤，画在链条
下方作并列注记。
两边的方框一一对应，颜色沿用 _diagram.py 的约定：蓝=数据、橙=权重、青绿=本轮改掉的部件。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import (ACCENT, FS_NAME, FS_SMALL, FS_TEXT, INK, MUTED, arrow,
                      box, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("13_decoder_models", "ch13_block_diff.png")

W, H, GAP = 5.4, 0.92, 0.42          # 方框宽、高、竖直间距
LEFT_X, RIGHT_X = 0.4, 7.6


def stack(ax, x, title, subtitle, flow_rows, attr_rows):
    """画一列方框：flow_rows 用箭头串成执行顺序，attr_rows 是块的属性，不进箭头链。

    rows 元素是 (文字, kind)。返回两段的方框中心。
    """
    label(ax, x + W / 2, 12.55, title, fontsize=FS_NAME, bold=True)
    label(ax, x + W / 2, 11.95, subtitle, fontsize=FS_SMALL, color=MUTED)
    flow = []
    y = 11.2
    for text, kind in flow_rows:
        cx, cy = box(ax, x, y - H, W, H, text, kind, fontsize=FS_TEXT)
        flow.append((cx, cy))
        y -= H + GAP
    for (x0, y0), (x1, y1) in zip(flow, flow[1:]):
        arrow(ax, (x0, y0 - H / 2), (x1, y1 + H / 2), color=MUTED, lw=1.2)

    y -= 0.34                                   # 与箭头链拉开，表示不属于流程
    attrs = []
    for text, kind in attr_rows:
        cx, cy = box(ax, x, y - H, W, H, text, kind, fontsize=FS_TEXT)
        attrs.append((cx, cy))
        y -= H + GAP
    return flow, attrs


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(13.0, 7.6))

    gpt_flow = [
        ("输入 x[T, 1600]", "data"),
        ("LayerNorm", "new"),
        ("自注意力\nMHA：25 个头各有自己的 K/V", "new"),
        ("LayerNorm", "new"),
        ("MLP：两个矩阵，GELU\nd_ff = 4d = 6,400", "new"),
    ]
    gpt_attrs = [
        ("位置：可学习位置嵌入表\n1,024 x 1,600，进块之前加在输入上", "new"),
        ("所有线性层带偏置项", "weight"),
    ]
    llama_flow = [
        ("输入 x[T, 4096]", "data"),
        ("RMSNorm", "new"),
        ("自注意力\nGQA：32 个 Q 头共用 8 组 K/V", "new"),
        ("RMSNorm", "new"),
        ("MLP：三个矩阵，SwiGLU\nd_ff = 14,336 = 3.5d", "new"),
    ]
    llama_attrs = [
        ("位置：RoPE，逐层作用在 Q/K 上\n不占参数，theta = 500,000", "new"),
        ("线性层一律不带偏置项", "weight"),
    ]

    left, left_attrs = stack(ax, LEFT_X, "GPT-2 块",
                             "48 层，d_model = 1,600，25 个头",
                             gpt_flow, gpt_attrs)
    right, right_attrs = stack(ax, RIGHT_X, "Llama 3 8B 块",
                               "32 层，d_model = 4,096，32 个头",
                               llama_flow, llama_attrs)

    flow_notes = [
        "",
        "算子换了，位置没换：两侧都是 Pre-Norm。\n少算一次均值与减均值，每个归一化少一组偏置",
        "省的是 KV 缓存，不是参数。\n基准换成 Llama 2 7B -> Llama 3 8B：\n每层 K/V 投影 33.6M -> 8.4M（见 13.2.3）",
        "",
        "同宽度下多一个门控矩阵。\n同一基准：每层 FFN 135.3M -> 176.2M",
    ]
    attr_notes = [
        "相对位置性质；直接外推仍会掉点",
        "每层省 3d + d + d_ff + d 个数",
    ]
    for (x0, y0), note in list(zip(right, flow_notes)) + list(zip(right_attrs, attr_notes)):
        if not note:
            continue
        label(ax, RIGHT_X + W + 0.35, y0, note, fontsize=FS_SMALL,
              color=ACCENT, ha="left")

    pairs = list(zip(left[1:], right[1:])) + list(zip(left_attrs, right_attrs))
    for (x0, y0), (x1, y1) in pairs:
        arrow(ax, (x0 + W / 2 + 0.08, y0), (x1 - 0.08, y1), color=ACCENT,
              lw=1.1, style="-|>")

    label(ax, (LEFT_X + W + RIGHT_X) / 2, 1.15,
          "箭头链是块内的执行顺序；下方两项是块的属性，不在这条顺序里",
          fontsize=FS_SMALL, color=INK)

    ax.set_xlim(0, 18.2)
    ax.set_ylim(0.75, 13.0)
    ax.axis("off")
    fig.savefig(OUTPUT, dpi=115, bbox_inches="tight", facecolor="white")
    print(f"已写入 {OUTPUT}")


if __name__ == "__main__":
    main()
