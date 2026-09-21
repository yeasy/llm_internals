"""生成 14.6 节插图：DeepSeek-R1 的四阶段训练流程，以及每一阶段的输入与产出。

正文位置：14_future_trends/14.6_test_time_scaling.md
输出：14_future_trends/_images/ch14b_r1_pipeline.png

四段流程、样本量与奖励构成取自 DeepSeek-R1 论文（arXiv:2501.12948）的奖励设计与
数据配方两节。冷启动为 "thousands of cold-start data"；第三段为 "about 600k
reasoning related training samples" 加 "approximately 200k training samples that
are unrelated to reasoning"，合计 "about 800k samples"；该文 v2（Nature 版）的
超参数一节写 "we fine-tune DeepSeek-V3-Base for 2-3 epochs using the curated
dataset"（v1 此处写的是 two epochs）。R1-Zero 的奖励为 accuracy rewards 与
format rewards 两项，且不使用神经奖励模型。

蓝 = 数据，橙 = 模型权重，青绿 = 本阶段新产生的东西，紫 = 强调。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import ACCENT, INK, MUTED, arrow, box, finish, label
from _style import image_path, use_cjk_font

OUTPUT = image_path("14_future_trends", "ch14b_r1_pipeline.png")
FS, FS_S = 10.0, 9.2

XT, WT = 0.3, 5.2          # 左列：阶段名
XD, WD = 6.2, 9.6          # 右列：这一阶段做了什么
ROW_H, GAP = 1.5, 0.95


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(9.2, 7.6))

    top = 10.6
    box(ax, XT, top, WT, 1.0, "DeepSeek-V3-Base", "weight", fontsize=FS)

    stages = [
        ("阶段 1\n冷启动 SFT",
         "输入数千条长思维链数据，微调基座模型\n产出：可读的格式与起手能力"),
        ("阶段 2\n面向推理的 RL",
         "奖励 = 答案是否正确 + 格式是否合规\n收敛前再加一项语言一致性奖励"),
        ("阶段 3\n拒绝采样 + SFT",
         "从阶段 2 的检查点采样筛出 60 万条推理样本\n"
         "加 20 万条非推理样本，合计约 80 万条"),
        ("阶段 4\n全场景 RL",
         "推理题继续用规则奖励\n其余任务改用偏好奖励，对齐有用与无害"),
    ]
    ys = []
    for k, (title, body) in enumerate(stages):
        y = top - (k + 1) * (ROW_H + GAP)
        ys.append(y)
        box(ax, XT, y, WT, ROW_H, title, "new", fontsize=FS, bold=True)
        box(ax, XD, y, WD, ROW_H, body, "data", fontsize=FS_S)
        arrow(ax, (XT + WT / 2, y + ROW_H + GAP), (XT + WT / 2, y + ROW_H),
              color=INK, lw=1.6)

    y_end = ys[-1] - (ROW_H + GAP) + 0.5
    box(ax, XT, y_end, WT, 1.0, "DeepSeek-R1", "weight", fontsize=FS)
    arrow(ax, (XT + WT / 2, ys[-1]), (XT + WT / 2, y_end + 1.0), color=INK, lw=1.6)

    # 阶段 3 的那条回退线：重新从 Base 起训，而不是在阶段 2 的检查点上续训
    x_back = XD + WD + 0.55
    ax.plot([XT + WT + 0.35, x_back, x_back],
            [top + 0.5, top + 0.5, ys[2] + ROW_H / 2], color=ACCENT, lw=1.6, zorder=1)
    arrow(ax, (x_back, ys[2] + ROW_H / 2), (XD + WD, ys[2] + ROW_H / 2),
          color=ACCENT, lw=1.6)
    label(ax, (XT + WT + 0.35 + x_back) / 2, top + 1.25,
          "阶段 3 不在阶段 2 的检查点上续训\n而是用这 80 万条重新微调 Base，训练 2-3 个 epoch",
          fontsize=FS_S, color=ACCENT)

    # 对照支线：R1-Zero 跳过阶段 1、3、4
    x_zero = XT - 0.5
    ax.plot([x_zero, x_zero], [top + 0.5, y_end + 0.5], color=MUTED, lw=1.3, zorder=1)
    arrow(ax, (x_zero, y_end + 0.5), (XT, y_end + 0.5), color=MUTED, lw=1.3)
    ax.text(x_zero - 0.3, (top + y_end) / 2 + 0.5,
            "R1-Zero：跳过阶段 1、3、4，直接在 Base 上做规则奖励的 RL",
            fontsize=FS_S, color=MUTED, ha="center", va="center", rotation=90, zorder=6)

    finish(fig, ax, OUTPUT, xlim=(-2.2, 17.0), ylim=(y_end - 0.4, top + 2.0))


if __name__ == "__main__":
    main()
