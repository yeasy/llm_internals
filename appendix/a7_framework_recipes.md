## A.7 开源框架工程实践

本附录把第 7、8、11 章的机制接到三条工作流：先用本地数据完成一次可检查的微调，再按瓶颈扩展训练，最后比较服务质量与成本。判断依据是相同任务、相同预算下的实验结果，不是框架名称。

**执行边界：以下是环境配方与工程决策指南，未在本机执行 CUDA 训练或服务压测。** 完整脚本与集成片段分别标明；语法检查不能证明依赖兼容、模型适配或性能达标。参数以链接的官方接口为依据；读者实际部署时，应保存依赖版本与启动命令，并按自己的工作负载验收。

### A.7.1 先冻结实验契约

准备一个已安装兼容依赖的 Linux/CUDA 环境。训练环境需要 PyTorch、Transformers、Datasets、PEFT、TRL；两个服务端分别使用各自受支持的环境，压测客户端固定为第三个环境。不要把多个引擎的依赖强行装进同一环境。模型必须已经在本地，具有权重、配置、分词器；本例不启用远程模型代码。

在实验主终端执行以下初始化，将示例绝对路径替换成实际目录。`HF_HUB_OFFLINE` 使缺少缓存成为明确错误；`local_files_only=True` 进一步约束下面的 Transformers 加载调用，见[离线运行说明](https://huggingface.co/docs/transformers/installation#offline-mode)。

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export MODEL_DIR=/absolute/path/to/local-small-causal-lm
export DATA_DIR=/absolute/path/to/local-data
export RUN_DIR=/absolute/path/to/new-experiment
test -d "$MODEL_DIR" && test -d "$DATA_DIR" || exit 1
mkdir "$RUN_DIR" || exit 1  # 要求父目录已存在，拒绝复用已有实验目录
python -m pip freeze > "$RUN_DIR/environment.txt"
nvidia-smi > "$RUN_DIR/gpu.txt"
```

同一次实验的其他终端只需重复上述 `export`，指向已创建的目录，不再执行 `mkdir`。不同训练配置使用新的实验目录；服务端与客户端环境清单分别保存，例如 `vllm-environment.txt`、`sglang-environment.txt` 和 `client-environment.txt`，不要覆盖主终端记录。

这里选能在一张卡容纳的小型、非量化因果语言模型；BF16 路线要求硬件支持 BF16。模型结构还必须在所用服务引擎支持范围内。目录名称不代表参数量，也不构成显存保证。

每次实验留下四组内容：模型权重与分词器的校验值，数据版本及划分，全部非默认配置，质量和资源测量结果。训练集、调参验证集、最终测试集按来源或任务实体隔离；逐行随机切分不能保证近重复或同源样本隔离，样本少时更需检查。

预先填写验收表，阈值来自业务要求，不从测试结果反推：

| 门禁 | 最小对照与证据 | 不合格时的动作 |
| --- | --- | --- |
| 正确性 | 少量样本的标签、掩码、输出；保存后重载一致 | 暂停扩展训练，先修数据流 |
| 质量 | 原模型、SFT、候选方法在同一保留集的逐题结果 | 查退化任务与数据分布，不只追 loss |
| 资源 | 有效训练词元/秒、峰值显存、GPU 小时 | 对照 profiler 找瓶颈，再改一个因素 |
| 服务 | 错误率、TTFT/TPOT 分位数、输出吞吐、质量 | 在满足质量与延迟约束下比较成本 |
| 恢复 | 中断重启后的步数、优化器与采样位置 | 未通过恢复演练不扩大训练预算 |

“能跑完十步”只通过冒烟检查。它不能证明收敛、泛化、偏好提升或可用容量。

### A.7.2 本地 SFT 与 LoRA：先验证学了什么

**目标与输入。** 用单卡验证完成词元上的监督微调；同一脚本可在全参数与 LoRA 间做消融。数据是本地 `train.jsonl`、`valid.jsonl`，每行包含两个非空字符串。下例是数据格式样例，不是足够的训练集：

```json
{"prompt":"把下面术语解释成一句中文。\n术语：梯度累积\n解释：", "completion":"梯度累积是在多次反向传播后统一更新参数。"}
```

这里有意使用普通文本的 prompt/completion 数据；对话模型应改为模型匹配的会话格式并检查 chat template，不能把任意拼接当作对话训练。TRL 的[数据格式与完成词元损失](https://huggingface.co/docs/trl/sft_trainer#expected-dataset-type-and-format)规定了两类输入的处理方式。

下面是完整的 `sft_local.py`。依赖已准备好；默认只运行十步。它检查长度、明确 loss 的监督范围、评估并保存模型；不负责数据采集。

```python
import os
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

set_seed(42)
model_dir = Path(os.environ["MODEL_DIR"]).resolve(strict=True)
data_dir = Path(os.environ["DATA_DIR"]).resolve(strict=True)
run_dir = Path(os.environ["RUN_DIR"])
if (run_dir / "sft").exists() or (run_dir / "sft-final").exists():
    raise FileExistsError("请使用新的实验目录，避免覆盖或混合结果")
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
if tok.eos_token is None:
    raise ValueError("本例要求分词器定义 eos_token")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"
data = load_dataset("json", data_files={
    "train": str(data_dir / "train.jsonl"),
    "validation": str(data_dir / "valid.jsonl"),
})
for split in data:
    if len(data[split]) == 0:
        raise ValueError(f"{split}: 数据为空")
    for row in data[split]:
        if any(not isinstance(row[k], str) or not row[k].strip()
               for k in ("prompt", "completion")):
            raise ValueError("prompt/completion 必须为非空字符串")
        # 预留 EOS 位置；正式数据处理还要检查实际 trainer 输出。
        ids = tok(row["prompt"] + row["completion"])["input_ids"]
        if len(ids) + 1 > 512:
            raise ValueError("样本超过本实验长度预算，请先处理数据")
model = AutoModelForCausalLM.from_pretrained(
    model_dir, local_files_only=True, dtype=torch.bfloat16,
)
model.config.use_cache = False
use_lora = os.environ.get("USE_LORA", "1") == "1"
lora = LoraConfig(
    task_type="CAUSAL_LM", r=8, lora_alpha=16,
    lora_dropout=0.0, target_modules="all-linear", bias="none",
) if use_lora else None
args = SFTConfig(
    output_dir=str(run_dir / "sft"), max_steps=10,
    per_device_train_batch_size=1, gradient_accumulation_steps=4,
    per_device_eval_batch_size=1, learning_rate=2e-5,
    max_length=512, packing=False, completion_only_loss=True,
    bf16=True, fp16=False, logging_steps=1,
    save_strategy="no", report_to="none", seed=42,
)
trainer = SFTTrainer(
    model=model, args=args, train_dataset=data["train"],
    eval_dataset=data["validation"], processing_class=tok,
    peft_config=lora,
)
batch = next(iter(trainer.get_train_dataloader()))
labels = batch["labels"]
assert (labels != -100).any(dim=1).all(), "存在没有监督词元的样本"
print("首个训练批的监督文本：")
print(tok.decode(labels[0][labels[0] != -100].tolist()))
print("可训练参数：", sum(p.numel() for p in trainer.model.parameters()
                           if p.requires_grad))
torch.cuda.reset_peak_memory_stats()
trainer.train()
print(trainer.evaluate())
print("peak_allocated_bytes:", torch.cuda.max_memory_allocated())
trainer.save_model(str(run_dir / "sft-final"))
if trainer.is_world_process_zero():
    tok.save_pretrained(run_dir / "sft-final")
```

保存脚本后，在同一终端运行。全参数对照使用新的输出目录，避免覆盖第一次结果：

```bash
python sft_local.py
USE_LORA=0 RUN_DIR=/absolute/path/to/full-tuning-control python sft_local.py
```

脚本中的学习率、秩与步数是冒烟配置，不是任务最优值。LoRA 只训练低秩增量，激活和基座前向成本仍在；`all-linear` 的支持与排除规则见 [PEFT LoRA 接口](https://huggingface.co/docs/peft/package_reference/lora)。正式比较应固定样本顺序、有效训练词元数和评测集，再分别给两种方法相同的调参预算，而不是要求它们共用一个最优学习率。

**验收。** 首批打印的监督文本应包含答案而不包含问题；再抽查边界样本的 EOS、特殊词元与实际截断结果。十步内 loss 与梯度须有限，验证集可评估，LoRA 可训练参数应少于总参数。随后选一个小训练子集检验能否过拟合；若做不到，先检查监督标签、优化器和数据，再扩大数据量。

LoRA 保存的是 adapter，不能当成完整权重目录直接交给任何服务引擎。以下是独立的重载片段，使用原始本地基座；对应 [PEFT 模型加载接口](https://huggingface.co/docs/peft/package_reference/peft_model)。对保存前后模型输入同一批词元，比较 logits 或确定性解码结果，再评估保留集。

```python
import os
from pathlib import Path
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    os.environ["MODEL_DIR"], local_files_only=True,
)
restored = PeftModel.from_pretrained(
    base, str(Path(os.environ["RUN_DIR"]) / "sft-final"),
    local_files_only=True,
).eval()
```

**排障。** loss 为零先查所有 labels 是否被屏蔽；OOM 先减长度与微批，再评估激活检查点；目标模块不匹配先打印模型模块名，不随意替换名字。若训练 loss 下降但保留集变差，回到 [8.5 节](../08_alignment/8.5_practice.md)检查遗忘、污染与风格模仿。本例关闭中间检查点，只适合短实验；正式长任务必须补齐并演练[训练状态恢复](../07_distributed_training/7.7_checkpoint.md)。

<a id="distributed-frameworks"></a>

### A.7.3 从 DDP 到分片：由内存账单决定

**目标与环境。** 先在单机多卡保持更新语义，再减少每卡模型状态。DDP 复制模型并同步梯度；FSDP2 分片参数、梯度与优化器状态。选择前先区分峰值来自状态、激活还是临时工作区，见 [7.2 节](../07_distributed_training/7.2_zero.md)。激活已经占主导时，分片并不直接解决主要问题。

若单卡容纳完整模型，可以把前述单卡脚本作为 DDP 起点，由 Trainer 管理分布式训练。两卡、每卡微批 1、累积 4 的名义全局批量是 8；与单卡比较时须相应调整累积步数，不能把扩大批量的效果误算成并行收益。

```bash
torchrun --standalone --nproc_per_node=2 sft_local.py
```

对自己编写的训练循环，下面是**集成片段，不是完整脚本**。调用者必须提供各 rank 相同初始化的模型与分片后的 DataLoader；`blocks` 必须由目标架构明确提供，不能猜测属性路径。调用点位于创建模型之后、优化器之前；二选一，不能叠加 DDP 与 FSDP2。接口与初始化顺序见 [DDP 教程](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)和 [FSDP2 教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)。

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import fully_shard

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")

def prepare(model, blocks, mode):
    if mode == "ddp":
        model = DDP(model.cuda(local_rank), device_ids=[local_rank])
    elif mode == "fsdp2":
        for block in blocks:
            fully_shard(block)  # 由内向外，形成逐层通信单元
        fully_shard(model)
    else:
        raise ValueError(mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    return model, optimizer

# 后续接原训练循环；所有 rank 完成后调用 dist.destroy_process_group()。
```

**输入契约。** 分布式采样器必须按 rank 分配数据，每轮更新 shuffle 的 epoch；每卡参与相同数量的 collective。梯度累积还要处理最后不足一组的微批。有效 token 数不同时，“先各卡取平均，再对卡取平均”不等于全局 token 平均，必须按 [7.1 节](../07_distributed_training/7.1_data_parallel.md)的分母统一损失权重。

**验收。** 关闭 dropout，在同一个很小的全局批上，对照单卡与多卡一步更新；比较全局 loss、参数更新范数和容差内差异。再量稳态 step 时间与每卡峰值显存。FSDP2 必须额外验证分片检查点能恢复；不要用普通 rank 0 保存逻辑直接假定拿到了完整权重，参见 [Distributed Checkpoint](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)。

**故障定位。** 启动即 OOM 查模型加载与首个 all-gather；反向 OOM 查激活、预取与分片单元大小；偶发卡死查首个报错 rank、数据迭代长度和 collective 次序；扩卡变慢查通信暴露时间与数据供给。只把整个模型包成一个 FSDP 单元，会失去逐层释放与重叠的主要收益。

#### Megatron-Core：把并行布局接到训练循环

[Megatron-Core](https://docs.nvidia.com/megatron-core/developer-guide/latest/)提供可组合的 Transformer 层、并行通信、流水线调度和分布式优化器；Megatron-LM 是使用这些构件的参考训练实现。它适合需要张量并行（TP）、流水线并行（PP）、上下文并行（CP）或专家并行（EP）的模型与集群，不能只给前面的 Trainer 加一个开关就完成迁移。并行机制见第 7 章，这里关注构件如何形成完整任务。

**输入与运行路径。** 先准备模型结构、分词器、数据划分、精度和并行配置；从头预训练可随机初始化，微调则需要映射到目标架构的预训练权重。Megatron-LM 的经典预训练数据路径将文本预处理为 `.bin`/`.idx`，不是直接把前述 SFT JSONL 当成训练输入，见[快速入门](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/quickstart.html)。启动器创建各 rank，初始化通信组；模型按 TP/PP 等布局实例化，数据加载器按数据并行（DP）分样本；流水线调度器组织微批的前向与反向，优化器完成同步后的更新，检查点系统保存各 rank 的状态。先沿官方最小训练循环接通这条路径，再接真实数据与完整模型。

**拓扑示例。** 对不启用 CP、EP 的稠密模型，16 卡可组成 `TP=4、PP=2、DP=2`：每个副本有两段流水线，每段四卡切张量，两个副本读取不同样本。这只是可检查的布局示意，不是通用最优配置。TP 优先放在高带宽互连域内；PP 要评估阶段负载与微批数量，长序列激活压力再考虑 CP，MoE 才评估 EP，依据[并行策略](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)与[性能指南](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-guide.html)。扩卡后先确认实际通信组与预期一致，再测暴露的通信时间、流水线空泡和有效词元吞吐。

**检查点与导出。** 分布式训练检查点用于恢复模型、优化器及训练进度；完整恢复还要保存并核对随机数与数据采样状态，见 [7.7 节](../07_distributed_training/7.7_checkpoint.md)。能否换拓扑续训，还受[优化器检查点格式](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/core/dist_checkpointing.html)与[数据加载状态](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/checkpointing.html)限制，不能从“支持分片加载”推断全部状态可任意重分片。交给 vLLM、SGLang 或 TensorRT-LLM 前，要另行导出目标格式；[Megatron Bridge](https://docs.nvidia.com/nemo/megatron-bridge/latest/bridge-tech-details.html)为受支持架构处理配置、参数名称、QKV 布局与并行分片的转换。导出后用相同输入词元比较 logits 或确定性输出，并在保留集复评；文件能加载不等于转换正确。

**验收与排障。** 先做小批的一步更新对照和同拓扑中断恢复，再扩大规模。loss 异常先查词元、掩码与损失分母；启动长时间等待先查[数据索引构建与共享缓存](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/data-loading.html)；计算卡死查首个报错 rank、通信组与 collective 次序；GPU 空等查流水线不均衡、微批不足与数据供给。多种并行同时打开会扩大排查空间，应逐项加入并留下对照。

#### DeepSpeed：按状态开销选择 ZeRO 与卸载

**输入与运行路径。** 保留已有 PyTorch 模型与数据处理，增加 DeepSpeed 配置，明确微批、梯度累积、精度、优化器和 `zero_optimization`。自写循环通过 `deepspeed.initialize` 得到 engine，前向后由 engine 的 `backward` 与 `step` 管理梯度同步和更新，见[训练入口](https://www.deepspeed.ai/getting-started/)。Trainer 集成则传入 `deepspeed` 配置，由 Trainer 驱动；两条路径不要混用。使用集成支持的 `"auto"` 对齐批量、学习率等共同字段，或确保显式值一致，并核对最终生效配置，见 [Transformers 集成指南](https://huggingface.co/docs/transformers/deepspeed)。

**选择示例。** 权重可常驻但训练状态超预算时，先评估 ZeRO-1/2；参数本身也成为瓶颈时再评估 ZeRO-3。[ZeRO](https://www.deepspeed.ai/tutorials/zero/)的三个阶段依次分片优化器状态、梯度和参数，ZeRO-3 在计算时按需收集参数。若优化器仍挤占显存，可试 `offload_optimizer` 的 CPU 卸载；参数卸载 `offload_param` 属于 ZeRO-3 路径，见[集成配置](https://huggingface.co/docs/transformers/deepspeed)。例如，ZeRO-2 已能放下模型时，先比较有无 CPU 优化器卸载的峰值显存与 step 时间；容量收益必须连同 CPU 内存、计算及数据传输成本一起衡量，见 [ZeRO-Offload](https://www.deepspeed.ai/tutorials/zero-offload/)。激活占主导时，仍回到长度、微批和重算策略，不能只升 ZeRO 阶段。

**保存与交付。** engine 的训练检查点包含分片状态，原生 `save_checkpoint` 要由所有进程参与，单独让 rank 0 调用会等待其他 rank。用于推理时，要另存常规模型权重；ZeRO-3 的 16 位权重保存需按集成配置收集分片，或用[官方权重恢复工具](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html)从 ZeRO 检查点恢复完整 `state_dict`，并预算 CPU 内存。再配齐模型配置与分词器，重载复评，不能把任意一个 rank 的文件当完整模型。

**验收与排障。** 对照相同有效批量的一步更新，验证检查点恢复和推理导出两条路径；同时量显存、主机内存与稳态 step 时间。初始化 OOM 查分片加载是否真正生效；卸载后变慢查 CPU 优化器、主机内存与传输；保存卡死查所有 rank 是否进入同一保存调用。单卡或 DDP 已满足预算时，保留较简单的路径；选择框架的依据是瓶颈与完整交付成本。

### A.7.4 DPO 与 GRPO：先检查学习信号

**选择依据。** 有可靠的成对偏好、但没有可信在线奖励时先做 DPO；能自动验证答案且愿意支付在线采样成本时再实验 GRPO。两者都必须与相同起点的 SFT 对照，不能拿不同数据、不同基座的最终分数归因于算法。

#### 离线偏好优化

输入 `preference.jsonl` 使用同一 prompt 下的 chosen/rejected 字符串，先核验偏好方向与标签一致性：

```json
{"prompt":"用一句话解释 KV cache：", "chosen":"KV cache 缓存已处理词元的键和值，供后续解码复用。", "rejected":"KV cache 会省去所有注意力计算。"}
```

以下为**替换 SFT 训练段的集成片段**；复用前面导入与加载逻辑，但 `MODEL_DIR` 要指向已通过评测的完整本地 SFT 权重。重新加载模型，不能复用刚训练过且带 adapter 的对象。此路线在 SFT 权重上新建 LoRA；参考策略是冻结的 SFT 起点，数据与配置见 [DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer)。

```python
from trl import DPOConfig, DPOTrainer

assert lora is not None, "本片段要求在完整 SFT 权重上新建 LoRA"
preference = load_dataset("json", data_files=str(
    data_dir / "preference.jsonl"), split="train")
dpo = DPOTrainer(
    model=model, processing_class=tok, peft_config=lora,
    train_dataset=preference,
    args=DPOConfig(
        output_dir=str(run_dir / "dpo"), max_steps=10,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=5e-6, beta=0.1, max_length=512,
        bf16=True, fp16=False, report_to="none", save_strategy="no",
    ),
)
dpo.train()
```

这里必须开启 LoRA，`lora` 不能为 `None`；若要全参数 DPO，应另外配置并预算冻结参考模型。原始 SFT 只有 adapter 时，先用原基座重载、合并并保存完整权重，做合并前后的输出一致性检查，再作为该片段的起点。继续训练同一个 SFT adapter 与“在完整 SFT 权重上加新 adapter”的参考策略处理不同，不能混用。

**验收与排障。** 在独立偏好集评估盲测胜率、长度分布及通用能力回退；胜率差异附样本量与不确定性。偏好 accuracy 上升而真实回答变差时，检查长度偏差和 chosen/rejected 质量。先统计截断后是否仍保留两个答案的区别；两个答案都截成相同文本时，扩大训练不能修复监督信号。

#### 在线可验证奖励

GRPO 的输入是 prompt 与可验证标签。下面是**替换训练段的集成片段**，只演示纯文本数值答案任务；在新的进程中重新加载本地模型、分词器和 LoRA 配置。`math_train.jsonl` 每行形如 `{"prompt":"只输出整数：17+25=", "answer":"42"}`。完整模型与在线生成所需内存必须事先可容纳，不能用 SFT 的显存值直接推断。

```python
from trl import GRPOConfig, GRPOTrainer

math_data = load_dataset("json", data_files=str(
    data_dir / "math_train.jsonl"), split="train")
def exact_reward(completions, answer, **kwargs):
    return [float(text.strip() == str(target).strip())
            for text, target in zip(completions, answer)]

grpo = GRPOTrainer(
    model=model, processing_class=tok, peft_config=lora,
    reward_funcs=exact_reward, train_dataset=math_data,
    args=GRPOConfig(
        output_dir=str(run_dir / "grpo"), max_steps=10,
        per_device_train_batch_size=4, num_generations=4,
        gradient_accumulation_steps=1, max_completion_length=32,
        learning_rate=1e-6, beta=0.04,
        bf16=True, fp16=False, report_to="none", save_strategy="no",
        use_vllm=False,
    ),
)
grpo.train()
```

自定义奖励接收同一批生成结果与额外数据列；会话格式的 completions 结构不同，不能照搬上述字符串处理。生成数与有效批量存在整除约束，`beta` 显式设为非零表示本实验包含 KL 项，不能依赖默认值；这些行为见 [GRPOTrainer 官方接口](https://huggingface.co/docs/trl/grpo_trainer)。

**验收与排障。** 训练前给奖励函数输入正确答案、错误答案、空答案、带解释答案，确认得分符合任务契约。记录组内奖励标准差、全对/全错组占比、截断比例、生成长度和保留题正确率。组内奖励全相同时，相对奖励没有区分信号；应查题目难度与采样多样性。奖励升高而测试正确率不升，优先查奖励漏洞与数据泄漏，不能称为推理能力提升。此精确字符串奖励只适合严格格式的玩具任务，不是通用数学评测器。

### A.7.5 用同一客户端比较两个服务引擎

**目标与环境。** 在同一张空闲 GPU 上依次启动 vLLM、SGLang，使用同一份完整本地权重、分词器、精度、上下文上限与请求序列。这里只绑定本机回环地址。先选两者共同支持的标准模型；量化、多模态、投机解码与 adapter 服务属于后续独立实验。

在 vLLM 服务环境运行下面命令；先从日志确认权重精度、KV 类型、最大长度与可用 KV 容量。参数依据 [vLLM serve](https://docs.vllm.ai/en/latest/cli/serve/)。

```bash
vllm serve "$MODEL_DIR" --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 2048 \
  --gpu-memory-utilization 0.8 --no-enable-prefix-caching
```

完成第一轮并停止该服务后，在 SGLang 环境运行第二个命令；参数依据 [SGLang 服务配置](https://docs.sglang.io/docs/advanced_features/server_arguments)。两种内存比例参数的管理范围不完全相同，数值相等不表示实际显存相等；同时记录峰值占用与 KV 容量。

```bash
python -m sglang.launch_server --model-path "$MODEL_DIR" \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 \
  --context-length 2048 --mem-fraction-static 0.8 --disable-radix-cache
```

两轮均用固定版本的 SGLang 压测客户端，通过同一个 OpenAI-compatible completions 协议测量。下面的 `--backend vllm` 选择该协议客户端，并不要求被测服务只能是 vLLM；接口定义见[压测指南](https://docs.sglang.io/docs/developer_guide/bench_serving)。随机词元负载在本地生成，不下载公共数据集。

```bash
export ENGINE=vllm  # 第二轮改为 sglang
for RATE in 1 2 4; do
  python -m sglang.bench_serving \
    --backend vllm --base-url http://127.0.0.1:8000 \
    --model "$MODEL_DIR" --tokenizer "$MODEL_DIR" \
    --dataset-name random-ids --random-input-len 256 --random-output-len 64 \
    --random-range-ratio 1 --num-prompts 200 --seed 42 \
    --request-rate "$RATE" --max-concurrency 64 --warmup-requests 10 \
    --extra-request-body '{"temperature":0}' \
    --output-file "$RUN_DIR/${ENGINE}-${RATE}.jsonl" --output-details
done
```

`random-ids` 使用本地随机整数词元；不要换成 `random`，后者在缺少本地数据时可能下载 ShareGPT，见[官方数据生成实现](https://github.com/sgl-project/sglang/blob/main/python/sglang/benchmark/datasets/random.py)。词元解码成文本再重新分词后，长度可能变化，应以服务端实际计数核对。

这是容量冒烟负载。固定长度、强制输出预算的随机请求只回答性能问题，不能回答真实质量或生产容量。正式比较再增加本地脱敏的请求轨迹，保留输入/输出长度分布、到达时间、共享前缀与取消行为；确认两端实际处理的 token 数一致。禁用前缀缓存用于第一轮消融，再启用缓存并重放相同真实前缀；冷缓存与热缓存分开报告。

固定并发上限会限制实际到达率。出现客户端等待时，不能把目标 `request-rate` 当成服务已承接的速率；同时记录实际发出率、完成率、失败数与客户端 CPU/网络负载。每个测点先预热，再重复运行，保留各轮结果与运行顺序，不能挑最快的一轮。

### A.7.6 把指标变成可诊断的结论

设请求 $i$ 的发送时间为 $t_{i,0}$，首个输出词元到达为 $t_{i,1}$，末个输出词元到达为 $t_{i,n_i}$，输出词元数为 $n_i$。客户端口径下：

$$
\mathrm{TTFT}_i=t_{i,1}-t_{i,0},\qquad
\mathrm{TPOT}_i=\frac{t_{i,n_i}-t_{i,1}}{n_i-1}\quad(n_i>1).
$$

单词元响应没有 TPOT，应单列而非填零。p95 TTFT 是请求 TTFT 样本的第 95 百分位；p95 TPOT 是请求内平均出词时间的第 95 百分位。它们都不是“95% 的词元间隔”，一次流式返回多个词元时更要保留分块计时口径，详见 [11.13 节](../11_serving/11.13_best_practices.md)。

若测量窗口长度为 $T$，成功完成请求集合为 $S$，输出吞吐定义为 $Q_{\mathrm{out}}=\sum_{i\in S}n_i/T$。同时报告失败率与在途未完成数，不能把失败请求直接丢掉制造高吞吐。“p95 吞吐”只有在定义了固定窗口的吞吐时间序列后才有意义；通常更有用的是指定到达率下的 p95 延迟和整体吞吐。

质量由冻结测试集的任务指标衡量；性能配置改变后仍要重测。显存同时记录引擎的 KV 池容量、设备进程占用及训练张量峰值，它们不是同一指标。成本用完成同一合格工作量所需 GPU 小时衡量；云端再乘实际资源单价，重试、失败和预热都计入预算。

| 观察 | 首先验证的原因 | 单因素对照 |
| --- | --- | --- |
| 低负载 TTFT 已高 | 长输入、分词、冷启动或 prefill 慢 | 相同输出长度，缩短输入 |
| 到达率升高后 p95 TTFT 激增 | 排队、KV 不足导致抢占 | 降到达率，观察队列与抢占 |
| TTFT 正常、TPOT 高 | decode 带宽、批内竞争或通信 | 固定输入，改变并发与输出长度 |
| 峰值吞吐高但错误率上升 | 超时、过载、客户端瓶颈 | 加入失败与取消，重算合格吞吐 |
| 显存下降但质量退化 | 量化误差、截断、模板不一致 | 恢复精度/长度/模板，逐项消融 |

一次可交付实验应能回答：为什么采用该方法，收益出现在哪类输入，代价转移到什么资源，哪些测试能推翻结论。只有质量、延迟与预算同时达标的配置，才进入下一轮规模实验。

### A.7.7 专家决策与排障清单

工程经验的价值在于缩小搜索空间。以下清单把前文机制转成排查顺序；它不承诺通用最优参数，也不以本文配方代替目标环境的验证。

#### 训练：先排除无效工作，再减少资源开销

| 何时采用 | 先看什么 | 常见误区 |
| --- | --- | --- |
| 扩大数据或训练步数 | 抽看高损失样本，核验来源重复、答案截断、监督词元占比；按任务分层看退化 | 把格式噪声造成的高 loss 当成知识不足 |
| 微批受显存限制，采用梯度累积 | 对齐有效词元分母、更新次数和学习率进度；检查尾部不足一组的批次 | 批量数字相同就认为更新等价；累积更大就一定更快 |
| 使用 AMP 与激活检查点 | 先找非有限值首次出现的位置；确认峰值是否来自激活，再决定重算范围 | 把全模型强制转低精度当成 AMP；把激活重算当作训练状态恢复 |

有梯度缩放时，一个累积窗口内保持缩放因子不变，累积结束后再反缩放、裁剪和更新，见 [PyTorch AMP 示例](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)。激活检查点以重算换内存；同名的持久化检查点解决中断恢复，两者分别见 [7.5 节](../07_distributed_training/7.5_activation_checkpointing.md)与 [7.7 节](../07_distributed_training/7.7_checkpoint.md)。

#### 后训练：按监督信号选方法

有正确示范而模型不会遵循任务时，先做 SFT；显存或多任务维护预算有限时，再评估 LoRA。LoRA 是参数更新方式，可与 SFT、DPO 组合，不是另一种监督信号。有可信偏好对时考虑 DPO；能构造可靠在线奖励并承担采样成本时考虑 GRPO，参见 [8.5 节](../08_alignment/8.5_practice.md)。

先看失败题是否真正缺能力，再看标签与奖励能否区分好坏。DPO 要检查被截断的偏好差异；GRPO 要同时看奖励方差、输出长度和独立正确率，参见 [DPO](https://huggingface.co/docs/trl/dpo_trainer)与 [GRPO](https://huggingface.co/docs/trl/grpo_trainer)接口。格式得分上涨、答案更长或组内全同分，都不能单独证明能力增长；为奖励设计反例，比继续调学习率更能发现失真。

#### 推理：让优化匹配请求分布

| 何时采用 | 先看什么 | 常见误区 |
| --- | --- | --- |
| 重复长前缀多，启用前缀缓存 | 可复用词元比例、命中率与冷/热 TTFT | 期望它加速新词元解码 |
| 权重或 KV 容量紧张，评估量化 | 两类内存分别占多少、硬件内核支持、长输入质量 | 位宽更低就必然更快 |
| 单卡容量不足或需降低延迟，评估 TP | 通信时间、互连与同卡数多副本对照 | GPU 数翻倍就应吞吐翻倍 |
| 长 prefill 干扰流式尾延迟，评估 PD 分离 | KV 传输、两侧队列及资源配比 | 默认它会提高吞吐 |
| 解码延迟是主要约束，评估投机 | 接受率、草稿与验证开销，低/高并发分别比较 | 只报接受率，不报端到端收益 |

这些边界对应 [vLLM 前缀缓存](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)、[量化支持](https://docs.vllm.ai/en/latest/features/quantization/)、[并行部署](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/)、[PD 分离](https://docs.vllm.ai/en/latest/features/disagg_prefill/)和[投机解码](https://docs.vllm.ai/en/latest/features/speculative_decoding/)。按 [11.13 节](../11_serving/11.13_best_practices.md)先固定服务目标，再逐项加入优化；若复杂度增加而合格请求成本未降，应保留较简单的配置。
