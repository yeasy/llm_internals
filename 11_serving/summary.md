## 本章小结

**推理引擎架构**（vLLM、TensorRT-LLM、SGLang）整合了优化技术栈，为多用户并发场景提供高效服务。每个引擎的优化重点不同：vLLM 追求通用性和易用性（PagedAttention + 连续批处理），TensorRT-LLM 追求极致单卡性能（算子融合 + 硬件感知），SGLang 追求编程灵活性（RadixAttention 支持分支推理）。

**连续批处理**通过在每个解码步动态添加和移除请求，消除了静态批处理中短请求等待长请求的浪费，将吞吐量提升 2-10 倍。**分块 Prefill** 进一步防止长 Prompt 对 Decode 的头部阻塞，保证 TPOT 平稳性；**缓存感知路由**通过哈希亲和性最大化前缀缓存命中。

**PagedAttention** 借鉴操作系统的虚拟内存管理思想，将 KV 缓存划分为固定大小的页（16-64 tokens）并按需分配，消除了内存碎片化和预分配浪费，将显存利用率从 60-80% 提升至 90-95%。结合前缀缓存（Prefix Caching），可在高并发场景下将显存占用减少 50% 以上。

**分离式架构**（Disaggregated Prefill-Decode）通过物理切分专门用于 Prefill 和 Decode 的计算集群，解决了计算密集型与访存密集型任务的资源错配，在高速网络（InfiniBand/RoCEv2）支持下实现了 1.5-2 倍吞吐提升和 40-60% 的 TPOT 降低。**条件分离**策略可进一步适配混合流量模式；**异构张量并行**（Prefill 低 TP，Decode 高 TP）进一步优化。对于 MoE 模型，**专家并行**（Expert Parallelism）通过 Token 级 All-to-All 通信实现高效的分布式推理。

**超长上下文推理**需要 Ring Attention 等跨节点张量并行技术。在分离式架构支持下，结合 KV 缓存压缩和分级存储，可以在可接受延迟内处理 1M+ Token 上下文。

**硬件选择**的决策框架：GPU（NVIDIA H200 +45% 推理吞吐相比 H100、B200 极低精度 FP4/FP6 支持）是首选，需根据模型大小、吞吐-延迟权衡、输入输出分布选择适当的显存和带宽；下一代 **Rubin 架构**采用硅光互连突破 All-Reduce 瓶颈；TPU（Google v5e/v6e 支持光电混合动态交换）适合大规模部署；AMD GPU（MI350）提供成本敏感方案；RoCEv2 高速以太网证明了万级 GPU 集群无需 InfiniBand 的可行性。

**生产部署最佳实践**的完整技术栈包括：
- 性能指标：TTFT < 500ms、TPOT < 50ms、**Goodput**（SLA 约束下有效吞吐）、单词元成本
- 业界案例：Mooncake 的 KV Cache 中心架构（+525% 吞吐）、xAI Colossus 的万级 RoCEv2 集群、Meta 的前缀缓存预热、Google 的光电混合动态交换
- 架构：AI 网关（统一路由、降级、计费）+ 分离式推理集群 + 流式输出
- 调度：Token-aware 路由 + 条件分离策略 + 前缀缓存感知 + 缓存亲和性调度
- 可靠性：多层安全护栏（规则 + 分类器 + 重排序）+ 速率限制 + 隐私脱敏
- 可观测性：延迟拆解、KV Cache 监控、漂移检测

至此，第三部分“推理与部署篇”结束。下一部分将梳理主流 Transformer 变体模型和前沿架构创新。

---

> 📝 **发现错误或有改进建议？** 欢迎提交 [Issue](https://github.com/yeasy/llm_internals/issues) 或 [PR](https://github.com/yeasy/llm_internals/pulls)。
