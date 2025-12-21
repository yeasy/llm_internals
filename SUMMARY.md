# Summary

* [前言](README.md)

## 第一部分：基础篇

* [第一章：从序列建模到 Transformer](01_introduction/README.md)
  * [1.1 序列建模的根本挑战](01_introduction/1.1_seq_challenge.md)
  * [1.2 RNN 与 CNN：成就与瓶颈](01_introduction/1.2_rnn_cnn_limits.md)
  * [1.3 注意力的诞生：让模型学会"看哪里"](01_introduction/1.3_attention_birth.md)
  * [1.4 Transformer 的提出与核心思想](01_introduction/1.4_transformer_idea.md)
  * [1.5 里程碑时刻：从学术论文到产业变革](01_introduction/1.5_milestones.md)
  * [本章小结](01_introduction/summary.md)

* [第二章：注意力机制：为什么它是核心](02_attention/README.md)
  * [2.1 查询-键-值：一种信息检索的直觉](02_attention/2.1_qkv_intuition.md)
  * [2.2 缩放点积注意力：为什么要除以根号 d](02_attention/2.2_scaled_dot_product.md)
  * [2.3 多头注意力：为什么多个子空间更好](02_attention/2.3_multi_head.md)
  * [2.4 自注意力、交叉注意力与因果掩码](02_attention/2.4_self_cross_causal.md)
  * [2.5 注意力的代价：复杂度与局限](02_attention/2.5_complexity_limits.md)
  * [本章小结](02_attention/summary.md)

* [第三章：Transformer 核心组件解析](03_components/README.md)
  * [3.1 词嵌入：从离散符号到连续向量](03_components/3.1_embedding.md)
  * [3.2 位置编码：为什么顺序信息必须显式注入](03_components/3.2_position_encoding.md)
  * [3.3 前馈网络：Transformer 的"记忆层"](03_components/3.3_feedforward.md)
  * [3.4 残差连接：梯度为什么能流过百层网络](03_components/3.4_residual.md)
  * [3.5 层归一化：为什么选择 LayerNorm 而非 BatchNorm](03_components/3.5_layer_norm.md)
  * [3.6 编码器-解码器：完整架构如何协同工作](03_components/3.6_full_architecture.md)
  * [本章小结](03_components/summary.md)

* [第四章：位置编码的设计哲学](04_position_encoding/README.md)
  * [4.1 正弦位置编码：频率与外推的直觉](04_position_encoding/4.1_sinusoidal.md)
  * [4.2 可学习位置编码：灵活性与局限](04_position_encoding/4.2_learnable.md)
  * [4.3 旋转位置编码：为什么旋转能编码相对位置](04_position_encoding/4.3_rope.md)
  * [4.4 ALiBi 与其他相对位置方案](04_position_encoding/4.4_alibi_others.md)
  * [本章小结](04_position_encoding/summary.md)

## 第二部分：训练篇

* [第五章：预训练：为什么"预测下一个词"能学到知识](05_pretraining/README.md)
  * [5.1 自回归语言模型：从左到右的世界观](05_pretraining/5.1_autoregressive.md)
  * [5.2 掩码语言模型：完形填空的智慧](05_pretraining/5.2_masked_lm.md)
  * [5.3 编码器-解码器预训练：两种范式的统一](05_pretraining/5.3_encoder_decoder.md)
  * [5.4 预训练数据：规模定律与数据质量的博弈](05_pretraining/5.4_data_scaling.md)
  * [本章小结](05_pretraining/summary.md)

* [第六章：训练技术的底层逻辑](06_training_techniques/README.md)
  * [6.1 损失函数与优化器：为什么选择 Adam](06_training_techniques/6.1_loss_optimizer.md)
  * [6.2 学习率调度：为什么需要先预热再衰减](06_training_techniques/6.2_lr_schedule.md)
  * [6.3 正则化策略：防止过拟合的多重手段](06_training_techniques/6.3_regularization.md)
  * [6.4 批次与序列长度：效率与质量的平衡](06_training_techniques/6.4_batch_sequence.md)
  * [本章小结](06_training_techniques/summary.md)

* [第七章：大规模分布式训练](07_distributed_training/README.md)
  * [7.1 数据并行：为什么简单复制就能加速](07_distributed_training/7.1_data_parallel.md)
  * [7.2 ZeRO 优化：如何突破单卡显存限制](07_distributed_training/7.2_zero.md)
  * [7.3 模型并行与张量并行：拆分权重的艺术](07_distributed_training/7.3_model_tensor_parallel.md)
  * [7.4 流水线并行与混合并行策略](07_distributed_training/7.4_pipeline_hybrid.md)
  * [7.5 混合精度训练：精度与速度的权衡](07_distributed_training/7.5_mixed_precision.md)
  * [7.6 检查点管理与容错](07_distributed_training/7.6_checkpoint.md)
  * [本章小结](07_distributed_training/summary.md)

* [第八章：从预训练到对齐：让模型有用且安全](08_alignment/README.md)
  * [8.1 监督微调：教模型"怎么回答"](08_alignment/8.1_sft.md)
  * [8.2 RLHF：为什么需要人类反馈参与训练](08_alignment/8.2_rlhf.md)
  * [8.3 DPO 与新型对齐：从复杂到简洁的演化](08_alignment/8.3_dpo.md)
  * [8.4 参数高效微调：为什么不必更新所有参数](08_alignment/8.4_peft.md)
  * [本章小结](08_alignment/summary.md)

## 第三部分：推理与部署篇

* [第九章：解码策略：模型如何生成文本](09_decoding/README.md)
  * [9.1 自回归解码：逐词生成的机制](09_decoding/9.1_autoregressive_decode.md)
  * [9.2 贪心搜索与束搜索：确定性与全局最优](09_decoding/9.2_greedy_beam.md)
  * [9.3 采样策略：温度、Top-k 与 Top-p 的设计直觉](09_decoding/9.3_sampling.md)
  * [9.4 结构化输出与约束解码](09_decoding/9.4_constrained.md)
  * [9.5 推理时计算扩展：让模型学会深度思考](09_decoding/9.5_test_time_scaling.md)
  * [本章小结](09_decoding/summary.md)

* [第十章：推理优化：第一性原理的分析](10_inference_optimization/README.md)
  * [10.1 推理瓶颈分析：计算密集还是访存密集](10_inference_optimization/10.1_bottleneck.md)
  * [10.2 KV 缓存：为什么能避免重复计算](10_inference_optimization/10.2_kv_cache.md)
  * [10.3 Flash Attention：IO 感知的算法设计](10_inference_optimization/10.3_flash_attention.md)
  * [10.4 模型量化：用更少的位数表示权重](10_inference_optimization/10.4_quantization.md)
  * [10.5 剪枝与知识蒸馏：模型瘦身的两条路](10_inference_optimization/10.5_pruning_distillation.md)
  * [10.6 投机解码：为什么"先猜后验"能加速](10_inference_optimization/10.6_speculative_decoding.md)
  * [本章小结](10_inference_optimization/summary.md)

* [第十一章：推理引擎与生产部署](11_serving/README.md)
  * [11.1 推理引擎架构概览](11_serving/11.1_engines_overview.md)
  * [11.2 连续批处理与 PagedAttention](11_serving/11.2_continuous_batching.md)
  * [11.3 硬件选型：GPU、TPU 与 CPU 的适用场景](11_serving/11.3_hardware.md)
  * [11.4 生产部署最佳实践](11_serving/11.4_best_practices.md)
  * [本章小结](11_serving/summary.md)

## 第四部分：前沿与实践篇

* [第十二章：编码器系列模型](12_encoder_models/README.md)
  * [12.1 BERT：双向理解的突破](12_encoder_models/12.1_bert.md)
  * [12.2 RoBERTa、ALBERT 与 ELECTRA：BERT 的改进之路](12_encoder_models/12.2_roberta_albert.md)
  * [12.3 长文本编码器：Longformer 与 BigBird](12_encoder_models/12.3_longformer_bigbird.md)
  * [本章小结](12_encoder_models/summary.md)

* [第十三章：解码器系列与主流 LLM](13_decoder_models/README.md)
  * [13.1 GPT 系列：从语言模型到通用智能的扩展之路](13_decoder_models/13.1_gpt_series.md)
  * [13.2 Llama 家族：开源如何改变 LLM 格局](13_decoder_models/13.2_llama.md)
  * [13.3 DeepSeek、Gemini 与其他前沿模型](13_decoder_models/13.3_deepseek_gemini.md)
  * [13.4 编码器-解码器模型：T5 与 BART 的设计选择](13_decoder_models/13.4_t5_bart.md)
  * [本章小结](13_decoder_models/summary.md)

* [第十四章：架构创新与未来趋势](14_future_trends/README.md)
  * [14.1 高效注意力：突破二次复杂度的瓶颈](14_future_trends/14.1_efficient_attention.md)
  * [14.2 混合专家模型：为什么不必激活所有参数](14_future_trends/14.2_moe.md)
  * [14.3 状态空间模型与混合架构：注意力的挑战者](14_future_trends/14.3_ssm_hybrid.md)
  * [14.4 多模态 Transformer：统一不同模态的表示](14_future_trends/14.4_multimodal.md)
  * [14.5 AI Agent 与工具调用：让模型从"说"到"做"](14_future_trends/14.5_agent_tool_use.md)
  * [14.6 长上下文技术：从理论到工程实践](14_future_trends/14.6_long_context.md)
  * [14.7 未来展望](14_future_trends/14.7_outlook.md)
  * [本章小结](14_future_trends/summary.md)

* [附录](appendix/README.md)
  * [A.1 数学基础速查](appendix/a1_math_basics.md)
  * [A.2 PyTorch 实现示例](appendix/a2_pytorch_examples.md)
  * [A.3 主流模型参数速查表](appendix/a3_model_reference.md)
  * [A.4 推荐阅读与参考文献](appendix/a4_references.md)
