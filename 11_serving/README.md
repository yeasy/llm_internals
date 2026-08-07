# 第十一章：推理引擎与生产部署

将推理优化技术整合为完整的推理服务是从研究到产品的关键一步。本章介绍现代推理引擎的架构设计、连续批处理与 PagedAttention 的工作原理、分离式 Prefill-Decode 架构的前沿探索、硬件平台的选择考量，以及生产部署的最佳实践。

本章依次回答五个问题：

- 直接用 `model.generate()` 上生产会卡在哪里？（[11.1 节](11.1_engines_overview.md)）
- 同一批请求的生成长度差异巨大，怎么不让短请求陪着长请求空等？（[11.2 节](11.2_continuous_batching.md)）
- Prefill 计算密集、Decode 访存密集，挤在同一张 GPU 上错配在哪里？（[11.3 节](11.3_disaggregated_serving.md)）
- GPU、TPU 还是其他加速器，该按什么指标选型？（[11.4 节](11.4_hardware.md)）
- 一套上线的服务，靠哪些指标判断它够不够好？（[11.5 节](11.5_best_practices.md)）
