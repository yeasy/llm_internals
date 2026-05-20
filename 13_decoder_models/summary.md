## 本章小结

**GPT 系列**展示了规模扩展的力量——从 1.17 亿到未公开规模的 GPT-4，涌现能力的发现推动了行业向更大模型的竞赛。GPT-4o mini 改变了 API 定价格局，而 **o1 系列**开辟了推理时计算扩展的新维度，证明投入更多推理计算同样能大幅提升模型能力。

**Llama** 通过 RoPE、RMSNorm、SwiGLU、GQA 等架构改进确立了现代 LLM 的标准配方。Llama 3.1 的 405B 模型首次让开放权重模型追近闭源前沿，3.2 将 LLM 带到手机端侧，3.3 则以 70B 实现了 405B 级别的对齐效果，展现了开放生态的蓬勃发展。

**DeepSeek** 以创新的 MoE 架构和 FP8 训练大幅降低训练运行成本并达到前沿水平。特别是 **DeepSeek-R1-Zero** 展示了纯强化学习激发推理行为的可能性，正式 **DeepSeek-R1** 则采用 cold-start 数据和多阶段训练，并将推理模式蒸馏到小模型中，为开放推理模型开辟了新路径。**Gemini** 代表了原生多模态、百万级上下文和智能体化工作流的发展方向。**Claude** 以 Artifacts 和 Computer Use 推动了 AI 交互方式的创新，Opus 4.7（2026 年 4 月）进一步强调软件工程、视觉理解和复杂多步任务能力；具体基准应以官方模型页和系统卡为准。**Qwen 2.5** 标志着中国开源 LLM 在国际基准上的全面崛起。

**T5/BART** 等编码器-解码器模型在 Seq2Seq 任务上有架构优势，但在大规模实践中让位于更简洁的纯解码器架构。

下一章将讨论 Transformer 架构的前沿创新与未来趋势。

---

> 📝 **发现错误或有改进建议？** 欢迎提交 [Issue](https://github.com/yeasy/llm_internals/issues) 或 [PR](https://github.com/yeasy/llm_internals/pulls)。
