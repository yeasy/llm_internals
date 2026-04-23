## 本章小结

**推理引擎架构**（vLLM、TensorRT-LLM、SGLang）整合了优化技术栈，为多用户并发场景提供高效服务。每个引擎的优化重点不同：vLLM 追求通用性和易用性（PagedAttention + 连续批处理），TensorRT-LLM 追求极致单卡性能（算子融合 + 硬件感知），SGLang 追求编程灵活性（RadixAttention 支持分支推理）。

**连续批处理**通过在每个解码步动态添加和移除请求，消除了静态批处理中短请求等待长请求的浪费，将吞吐量提升 2-10 倍。结合 BatchScaling 等自适应策略可进一步优化。

**PagedAttention** 借鉴操作系统的虚拟内存管理思想，将 KV 缓存划分为固定大小的页（16-64 tokens）并按需分配，消除了内存碎片化和预分配浪费，将显存利用率从 60-80% 提升至 90-95%。结合前缀缓存（Prefix Caching），可在高并发场景下将显存占用减少 50% 以上。

**分离式架构**（Disaggregated Prefill-Decode）通过物理切分专门用于 Prefill 和 Decode 的计算集群，解决了计算密集型与访存密集型任务的资源错配，在高速网络（InfiniBand）支持下实现了 1.5-2 倍吞吐提升和 40-60% 的 TPOT 降低。条件分离策略可进一步适配混合流量模式。异构张量并行（Prefill 低 TP，Decode 高 TP）进一步优化了硬件利用率。

**超长上下文推理**需要 Ring Attention 等跨节点张量并行技术。在分离式架构支持下，结合 KV 缓存压缩和分级存储，可以在可接受延迟内处理 1M+ Token 上下文。

**硬件选择**的决策框架：GPU（NVIDIA H200/B200 用于推理）是首选，需根据模型大小、吞吐-延迟权衡、输入输出分布选择适当的显存和带宽；TPU（Google v6e）适合 Google Cloud 上的大规模部署；AMD GPU（MI350）提供了成本敏感的替代方案；FP4/FP8 精度支持成为选型的新维度。

**生产部署最佳实践**的完整技术栈包括：
- 性能指标：TTFT < 500ms、TPOT < 50ms、P99 延迟可控
- 架构：AI 网关（统一路由、降级、计费）+ 分离式推理集群 + 流式输出
- 调度：Token-aware 路由 + 条件分离策略 + 前缀缓存感知
- 可靠性：多层安全护栏（规则 + 分类器 + 重排序）+ 速率限制 + 隐私脱敏
- 可观测性：延迟拆解、KV Cache 监控、漂移检测

至此，第三部分”推理与部署篇”结束。下一部分将梳理主流 Transformer 变体模型和前沿架构创新。

---

> 📝 **发现错误或有改进建议？** 欢迎提交 [Issue](https://github.com/yeasy/llm_internals/issues) 或 [PR](https://github.com/yeasy/llm_internals/pulls)。
