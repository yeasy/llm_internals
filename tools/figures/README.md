# 插图生成脚本

本目录存放正文各章插图的生成脚本。**正文只保留渲染好的图与讲解，不再内联 matplotlib
绘图代码**——那些代码的信息量几乎全是坐标轴与图例的装饰调用，与所在章节的主题无关，
而图本身已经渲染在正文里了。需要复现或改画时，来这里。

每个脚本都是自包含的：把正文里必要的计算一并带上，可直接运行，无需先执行正文的其他代码块。

## 用法

从**仓库根目录**运行，脚本会把图写回对应章节的 `_images/` 目录：

```bash
pip install torch matplotlib
python tools/figures/attention_heatmap.py
```

中文标签需要系统装有中文字体（macOS 的 PingFang SC、Windows 的 Microsoft YaHei、
Linux 的 Noto Sans CJK 任一即可）；各脚本已按此顺序自动挑选可用字体，找不到时会退回
默认字体并在图中把中文显示成方框，这不影响曲线本身。

## 清单

| 脚本 | 生成 | 正文位置 |
|---|---|---|
| `attention_heatmap.py` | 图 2-1 缩放点积注意力权重热力图 | [2.2](../../02_attention/2.2_scaled_dot_product.md) |
| `causal_mask_heatmap.py` | 图 2-4 因果掩码前后的注意力权重对比 | [2.4](../../02_attention/2.4_self_cross_causal.md) |
| `sinusoidal_pe.py` | 图 4-1 正弦位置编码的频率分解 | [4.1](../../04_position_encoding/4.1_sinusoidal.md) |
| `rope_phase.py` | 图 4-2 RoPE 注意力分数随相对距离的相位变化 | [4.3](../../04_position_encoding/4.3_rope.md) |
| `lr_schedule_comparison.py` | 图 6-1 三种学习率调度策略对比 | [6.2](../../06_training_techniques/6.2_lr_schedule.md) |

`tests/test_figure_scripts.py` 保证这张表与正文引用、磁盘文件三者不脱节：正文引用的脚本
必须存在，本目录的每个脚本必须被正文引用，且每个脚本声明的输出图必须已提交。
