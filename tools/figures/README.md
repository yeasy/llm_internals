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
| `attention_heatmap.py` | 图 2-2 缩放点积注意力权重热力图 | [2.2](../../02_attention/2.2_scaled_dot_product.md) |
| `causal_mask_heatmap.py` | 图 2-4 因果掩码前后的注意力权重对比 | [2.4](../../02_attention/2.4_self_cross_causal.md) |
| `inference_timeline.py` | 图 3-10 回答“5 | [3.8](../../03_components/3.8_gpt_inference_flow.md) |
| `input_pipeline.py` | 图 3-11 一条消息怎样变成初始表示 | [3.8](../../03_components/3.8_gpt_inference_flow.md) |
| `attention_shape_flow.py` | 图 3-12 一层注意力计算中各矩阵的形状变化 | [3.8](../../03_components/3.8_gpt_inference_flow.md) |
| `decode_with_cache.py` | 图 3-13 一轮 Decode 怎样对着 KV 缓存只算一行 | [3.8](../../03_components/3.8_gpt_inference_flow.md) |
| `transformer_layer_blocks.py` | 图 3-14 真实 GPT 中一层 Transformer 的结构与形状 | [3.8](../../03_components/3.8_gpt_inference_flow.md) |
| `inference_bottleneck.py` | 图 3-15 Prefill 与 Decode 各自落在哪一种瓶颈里 | [3.8](../../03_components/3.8_gpt_inference_flow.md) |
| `sinusoidal_pe.py` | 图 4-1 正弦位置编码的频率分解 | [4.1](../../04_position_encoding/4.1_sinusoidal.md) |
| `rope_phase.py` | 图 4-2 RoPE 注意力分数随相对距离的相位变化 | [4.3](../../04_position_encoding/4.3_rope.md) |
| `lr_schedule_comparison.py` | 图 6-1 三种学习率调度策略对比 | [6.2](../../06_training_techniques/6.2_lr_schedule.md) |

| `ch07_ring_allreduce.py` | 图 7-1 4 张卡上的 Ring AllReduce | [7.1](../../07_distributed_training/7.1_data_parallel.md) |
| `ch07_zero3_timeline.py` | 图 7-2 ZeRO-3 的一步 | [7.2](../../07_distributed_training/7.2_zero.md) |
| `ch07_tp_sp_layer.py` | 图 7-3 一层 Transformer 内张量并行的通信位置 | [7.3](../../07_distributed_training/7.3_model_tensor_parallel.md) |
| `ch07_pipeline_schedules.py` | 图 7-4 4 级流水线、8 个微批量时 GPipe 与 1F1B 的时间线 | [7.4](../../07_distributed_training/7.4_pipeline_hybrid.md) |
| `ch07_precision_flow.py` | 图 7-6 混合精度训练的一步 | [7.6](../../07_distributed_training/7.6_mixed_precision.md) |
| `ch07_reshard.py` | 图 7-7 同一个逻辑张量从 8 片重切为 6 片 | [7.7](../../07_distributed_training/7.7_checkpoint.md) |
| `ch09_logits_pipeline.py` | 图 9-1 一轮 Decode 的后半段 | [9.1](../../09_decoding/9.1_autoregressive_decode.md) |
| `ch09_padding_sides.py` | 图 9-2 批量生成时的右填充与左填充 | [9.1](../../09_decoding/9.1_autoregressive_decode.md) |
| `ch09_beam_tree.py` | 图 9-3 两步搜索树 | [9.2](../../09_decoding/9.2_greedy_beam.md) |
| `ch09_temperature_truncation.py` | 图 9-4 表 9-7 的图示 | [9.3](../../09_decoding/9.3_sampling.md) |
| `ch09_token_vs_char.py` | 图 9-6 正则 `[0-9]+\.[0-9]{2}` 的字符级自动机 | [9.4](../../09_decoding/9.4_constrained.md) |
| `ch09_block_diffusion.py` | 图 9-7 (a) 表 9-14 的去噪过程 | [9.6](../../09_decoding/9.6_diffusion_lm.md) |
| `ch10_roofline.py` | 图 10-1 H100 的 Roofline 与 Llama 3 8B 的四个工作点 | [10.1](../../10_inference_optimization/10.1_bottleneck.md) |
| `ch10_mla_dataflow.py` | 图 10-4 MLA 中一个词元的数据流 | [10.2](../../10_inference_optimization/10.2_kv_cache.md) |
| `ch10_flash_tiling.py` | 图 10-5 FlashAttention 分块前向的循环结构与各变量的驻留位置 | [10.3](../../10_inference_optimization/10.3_flash_attention.md) |
| `ch10_quant_grid.py` | 图 10-9 同一行权重在三种量化方案下落到的格点 | [10.4](../../10_inference_optimization/10.4_quantization.md) |
| `ch10_spec_round.py` | 图 10-10 投机解码的一轮 | [10.6](../../10_inference_optimization/10.6_speculative_decoding.md) |
| `ch11a_engine_components.py` | 图 11-1 推理引擎的组件与一个请求的路径 | [11.1](../../11_serving/11.1_engines_overview.md) |
| `ch11a_batching_gantt.py` | 图 11-2 同样 8 个请求 | [11.2](../../11_serving/11.2_continuous_batching.md) |
| `ch11a_mixed_batch_shapes.py` | 图 11-3 一轮迭代里的混合批 | [11.2](../../11_serving/11.2_continuous_batching.md) |
| `ch11a_block_table_mapping.py` | 图 11-4 两个请求的块表指向同一个物理块池 | [11.2](../../11_serving/11.2_continuous_batching.md) |
| `ch11a_chunked_prefill_timeline.py` | 图 11-5 整段 Prefill 与分块 Prefill 的时间线 | [11.2](../../11_serving/11.2_continuous_batching.md) |
| `ch11_3_request_states.py` | 图 11-6 一个请求在调度器里的状态机 | [11.3](../../11_serving/11.3_scheduler_loop.md) |
| `ch11_3_schedule_trace.py` | 图 11-7 玩具调度器的 11 步轨迹 | [11.3](../../11_serving/11.3_scheduler_loop.md) |
| `ch11_3_batch_latency.py` | 图 11-8 批大小、到达率与 TPOT 的关系 | [11.3](../../11_serving/11.3_scheduler_loop.md) |
| `ch11_4_block_table_cow.py` | 图 11-9 块表寻址与写时复制 | [11.4](../../11_serving/11.4_kv_memory_management.md) |
| `ch11_4_swap_vs_recompute.py` | 图 11-10 恢复一条被抢占序列的 K/V 要多久 | [11.4](../../11_serving/11.4_kv_memory_management.md) |
| `ch11_5_block_hash_chain.py` | 图 11-11 按块链式哈希怎样匹配前缀 | [11.5](../../11_serving/11.5_prefix_reuse.md) |
| `ch11_5_radix_tree_ops.py` | 图 11-12 前缀树的匹配、拆分节点与从叶子淘汰 | [11.5](../../11_serving/11.5_prefix_reuse.md) |
| `ch11_6_segmented_lora_batch.py` | 图 11-13 同一批 8 个词元、3 个适配器的一次线性投影 | [11.6](../../11_serving/11.6_multi_lora_serving.md) |
| `ch11_6_adapter_memory.py` | 图 11-14 适配器权重在显存里的两种放法 | [11.6](../../11_serving/11.6_multi_lora_serving.md) |
| `ch11_7_token_index.py` | 图 11-15 正则 `\{"age": (0|[1-9][0-9]*)\}` 的字符级 DFA | [11.7](../../11_serving/11.7_constrained_decoding.md) |
| `ch11_7_mask_overlap.py` | 图 11-16 掩码计算与前向的时序 | [11.7](../../11_serving/11.7_constrained_decoding.md) |
| `ch11_8_tp_layer.py` | 图 11-17 一层 Transformer 在张量并行下的数据流 | [11.8](../../11_serving/11.8_multi_gpu_inference.md) |
| `ch11_8_pp_timeline.py` | 图 11-18 4 级流水线做 Decode 时 | [11.8](../../11_serving/11.8_multi_gpu_inference.md) |
| `ch11_8_ep_all_to_all.py` | 图 11-19 专家并行中一次 dispatch 的收发矩阵与各卡负载 | [11.8](../../11_serving/11.8_multi_gpu_inference.md) |
| `ch11b_disagg_timeline.py` | 图 11-20 分离式架构下一个请求的时间线 | [11.9](../../11_serving/11.9_disaggregated_serving.md) |
| `ch11_10_vllm_request_path.py` | 图 11-21 一个请求在 vLLM 三类进程之间的路径 | [11.10](../../11_serving/11.10_vllm_internals.md) |
| `ch11_10_vllm_free_queue.py` | 图 11-22 空闲块队列兼做分配器与前缀缓存的 LRU | [11.10](../../11_serving/11.10_vllm_internals.md) |
| `ch11_11_sglang_three_tables.py` | 图 11-23 三张表怎样接在一起 | [11.11](../../11_serving/11.11_sglang_internals.md) |
| `ch11_11_sglang_overlap_loop.py` | 图 11-24 调度主循环的两种排法 | [11.11](../../11_serving/11.11_sglang_internals.md) |
| `ch11b_decode_batch_limit.py` | 图 11-25 批处理对 Decode 的作用与极限 | [11.12](../../11_serving/11.12_hardware.md) |
| `ch11b_goodput_curve.py` | 图 11-26 到达率升高时 TTFT 达标率与 P90 的变化 | [11.13](../../11_serving/11.13_best_practices.md) |

| `ch02_qkv_roles.py` | 图 2-1 三路投影各自流向哪里 | [2.1](../../02_attention/2.1_qkv_intuition.md) |
| `ch02_multihead_reshape.py` | 图 2-3 多头在实现里是“一次大投影 | [2.3](../../02_attention/2.3_multi_head.md) |
| `ch02_mask_family.py` | 图 2-5 五种掩码都是同一张 `[T, T]` 矩阵 | [2.4](../../02_attention/2.4_self_cross_causal.md) |
| `ch02_flops_crossover.py` | 图 2-6 $$d = 4096$$ 时 | [2.5](../../02_attention/2.5_complexity_limits.md) |
| `ch03_bpe_pipeline.py` | 图 3-1 一段文本怎样经预分词、字节化与合并变成词元 ID | [3.1](../../03_components/3.1_tokenization.md) |
| `ch03_ffn_shapes.py` | 图 3-3 两矩阵 FFN 与三矩阵 SwiGLU 的形状对照 | [3.4](../../03_components/3.4_feedforward.md) |
| `ch03_residual_stream.py` | 图 3-4 残差流——各子层从同一条通路读、向同一条通路写 | [3.5](../../03_components/3.5_residual.md) |
| `ch03_norm_axes.py` | 图 3-5 同一张 `[B, T, d]` 上 | [3.6](../../03_components/3.6_layer_norm.md) |
| `ch03_norm_placement.py` | 图 3-6 归一化放在哪里——Post-Norm、Pre-Norm 与前后双归一化 | [3.6](../../03_components/3.6_layer_norm.md) |
| `ch03_gqa_heads.py` | 图 3-9 8 个 Query 头共用几组 K/V | [3.7](../../03_components/3.7_full_architecture.md) |
| `ch04_rope_wavelength.py` | 图 4-3 左图是 64 对频率的波长谱 | [4.3](../../04_position_encoding/4.3_rope.md) |
| `ch04_alibi_decay.py` | 图 4-4 8 个头的 ALiBi 偏置换算成 Softmax 权重乘子后的衰减曲线 | [4.4](../../04_position_encoding/4.4_alibi_others.md) |
| `ch05_shift_labels.py` | 图 5-1 一条序列怎样错一位变成 T−1 个训练样本 | [5.1](../../05_pretraining/5.1_autoregressive.md) |
| `ch05_mlm_masking.py` | 图 5-2 一条句子经 80/10/10 改写后 | [5.2](../../05_pretraining/5.2_masked_lm.md) |
| `ch05_mask_shapes.py` | 图 5-3 全可见、因果、前缀三种注意力掩码形状 | [5.3](../../05_pretraining/5.3_encoder_decoder.md) |
| `ch05_scaling_curves.py` | 图 5-4 左为同一条幂律的两种画法 | [5.4](../../05_pretraining/5.4_data_scaling.md) |
| `ch05_data_pipeline.py` | 图 5-5 FineWeb 管线的逐步保留量 | [5.5](../../05_pretraining/5.5_data_pipeline.md) |
| `ch05_minhash_scurve.py` | 图 5-6 三组 $$(b, r)$$ 下的候选概率曲线 | [5.5](../../05_pretraining/5.5_data_pipeline.md) |
| `ch06_logit_guards.py` | 图 6-2 四类约束落在三个互不相同的位置 | [6.3](../../06_training_techniques/6.3_regularization.md) |
| `ch06_batch_tradeoff.py` | 图 6-3 批量在「步数」与「词元数」之间的取舍 | [6.4](../../06_training_techniques/6.4_batch_sequence.md) |
| `ch06_memory_stack.py` | 图 6-4 GPT-3 6.7B | [6.4](../../06_training_techniques/6.4_batch_sequence.md) |
| `ch08_sft_loss_mask.py` | 图 8-1 一条 SFT 样本怎样变成带掩码的损失 | [8.1](../../08_alignment/8.1_sft.md) |
| `ch08_ppo_iteration.py` | 图 8-2 一轮 PPO 迭代里四个模型谁读谁写 | [8.2](../../08_alignment/8.2_rlhf.md) |
| `ch08_reward_overopt.py` | 图 8-3 优化得越远 | [8.2](../../08_alignment/8.2_rlhf.md) |
| `ch08_lora_bypass.py` | 图 8-5 一个线性层加上 LoRA 旁路后的形状与显存去向 | [8.4](../../08_alignment/8.4_peft.md) |
| `ch08_tradeoff_curves.py` | 图 8-6 同一次微调里两条曲线与每轮的交换比 | [8.5](../../08_alignment/8.5_practice.md) |

`tests/test_figure_scripts.py` 保证这张表与正文引用、磁盘文件三者不脱节：正文引用的脚本
必须存在，本目录的每个脚本必须被正文引用，且每个脚本声明的输出图必须已提交。

`_style.py` 提供中文字体与输出路径；`_diagram.py` 提供 3.8 节各示意图共用的配色与绘图原语（方框、箭头、带数值的矩阵、省略号）。以下划线开头的文件是公共模块，不单独生成插图。
