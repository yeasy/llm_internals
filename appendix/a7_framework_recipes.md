## A.7 开源框架工程实践

本附录把第 7、8、11 章的机制接到三条工作流：先用本地数据完成一次可检查的微调，再按瓶颈扩展训练，最后比较服务质量与成本。A.7.8 收录正文各章迁出的源码名、配置键与启动参数。判断依据是相同任务、相同预算下的实验结果，不是框架名称。

[工单转 JSON 实验](#ticket-json-lab)在 Apple Silicon 上跑过完整的训练和评测。CUDA 配方和服务压测按各自环境准备，完整脚本与接入片段分别标明。保存依赖版本和启动命令，按自己的任务验收。

### A.7.1 先冻结实验契约 <a href="#a7-experiment-setup" id="a7-experiment-setup"></a>

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

### A.7.2 本地 SFT 与 LoRA：先验证学了什么 <a href="#cuda-sft-recipe" id="cuda-sft-recipe"></a>

本例在一张 NVIDIA GPU 上微调已有的指令模型。先按 [A.7.1](#a7-experiment-setup) 设置目录和环境变量。把审核过的问答写入本地的 `train.jsonl` 与 `valid.jsonl`，每行用 `prompt` 存放用户消息，用 `completion` 存放助手的示范回答。下面是一条样本。

```json
{"prompt":[{"role":"user","content":"把梯度累积解释成一句中文。"}],"completion":[{"role":"assistant","content":"梯度累积把多个小批次的梯度加起来，再进行一次参数更新。"}]}
```

TRL 根据模型的聊天模板把这些消息变成词元，再只对 `completion` 计损失。这个脚本接受单轮问答，可用于有聊天模板的纯文本因果语言模型，例如 [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)。若用有思考模式的模型，可按模型卡说明在每条样本的 `chat_template_kwargs` 里设定模式，推理时也用同一模式。数据与模板接口分别见 [TRL 数据格式](https://huggingface.co/docs/trl/dataset_formats)和 [SFTTrainer v1.7.0 的实现](https://github.com/huggingface/trl/blob/v1.7.0/trl/trainer/sft_trainer.py)。

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
max_length = int(os.environ.get("MAX_LENGTH", "512"))
if max_length < 1:
    raise ValueError("MAX_LENGTH 必须为正整数")
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
    if not len(data[split]):
        raise ValueError(f"{split}: 数据为空")
    for index, row in enumerate(data[split], start=1):
        prompt, answer = row["prompt"], row["completion"]
        for messages, role in ((prompt, "user"), (answer, "assistant")):
            if (not isinstance(messages, list) or len(messages) != 1
                    or not isinstance(messages[0], dict)
                    or messages[0].get("role") != role
                    or not isinstance(messages[0].get("content"), str)
                    or not messages[0]["content"].strip()):
                raise ValueError(f"{split} 第 {index} 条须为单轮问答")
        template_options = row.get("chat_template_kwargs") or {}
        prompt_ids = tok.apply_chat_template(
            prompt, add_generation_prompt=True, tokenize=True,
            return_dict=True, **template_options,
        )["input_ids"]
        full_ids = tok.apply_chat_template(
            prompt + answer, tokenize=True, return_dict=True, **template_options,
        )["input_ids"]
        if len(full_ids) > max_length:
            raise ValueError(f"{split} 第 {index} 条有 {len(full_ids)} 个词元，超过 {max_length}")
        if full_ids[:len(prompt_ids)] != prompt_ids or len(full_ids) == len(prompt_ids):
            raise ValueError(f"{split} 第 {index} 条的聊天模板无法分出回答位置")
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
    max_length=max_length, packing=False, completion_only_loss=True,
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

保存脚本后，在同一终端运行。默认上限 512 词元用于短问答，换成论文摘要时用 `MAX_LENGTH` 设置能放下正文、模板和目标摘要的预算。全参数对照使用新的输出目录，避免覆盖第一次结果。

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

#### 合并 SFT 适配器 <a href="#sft-adapter-merge" id="sft-adapter-merge"></a>

如果下一阶段需要完整的 SFT 权重，确认重载结果后执行下面的代码。它适用于本节的非量化基座。合并后再用同一批提示核对输出。[PEFT 合并接口](https://huggingface.co/docs/peft/package_reference/peft_model#peft.PeftModel.merge_and_unload)

```python
from transformers import AutoTokenizer

run_dir = Path(os.environ["RUN_DIR"])
merged_dir = run_dir / "sft-merged"
merged_dir.mkdir()
restored.merge_and_unload(safe_merge=True).save_pretrained(merged_dir)
AutoTokenizer.from_pretrained(
    run_dir / "sft-final", local_files_only=True,
).save_pretrained(merged_dir)
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

### A.7.4 DPO、PPO 与 GRPO：先检查学习信号

**选择依据。** 有可靠的成对偏好、但没有可信在线奖励时先做 DPO；能自动验证答案且愿意支付在线采样成本时再实验 GRPO。两者都必须与相同起点的 SFT 对照，不能拿不同数据、不同基座的最终分数归因于算法。

#### 离线偏好优化 <a href="#dpo-recipe" id="dpo-recipe"></a>

把同一条用户消息和两份助手回答写入 `preference.jsonl`。`chosen` 是审核者更认可的回答，`rejected` 是较差的回答。

```json
{"prompt":[{"role":"user","content":"用一句话解释 KV cache。"}],"chosen":[{"role":"assistant","content":"KV cache 缓存已处理词元的键和值，供后续解码复用。"}],"rejected":[{"role":"assistant","content":"KV cache 会省去所有注意力计算。"}]}
```

在新脚本中复用单卡 SFT 示例的导入、目录、模型和 LoRA 设置，跳过 SFT 的数据检查和训练。让 `MODEL_DIR` 指向已通过评测的完整本地 SFT 权重，另设新的 `RUN_DIR`。这段代码在 SFT 权重上新建 LoRA，冻结的 SFT 起点是参考策略。数据与配置见 [DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer)。

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
dpo.save_model(str(run_dir / "dpo-final"))
```

训练结束后，`dpo-final` 保存新适配器。评测时把它加载到同一份 SFT 权重上。若原始 SFT 只保存了适配器，先按上面的[合并步骤](#sft-adapter-merge)保存并检查完整权重。

**验收与排障。** 在独立偏好集评估盲测胜率、长度分布及通用能力回退；胜率差异附样本量与不确定性。偏好 accuracy 上升而真实回答变差时，检查长度偏差和 chosen/rejected 质量。先统计截断后是否仍保留两个答案的区别；两个答案都截成相同文本时，扩大训练不能修复监督信号。

#### 在本机跑一轮 PPO <a href="#ppo-capital-recipe" id="ppo-capital-recipe"></a>

这个例子用一个小 Transformer 回答首都问题。`prompt` 是模型读到的题目，`answer` 是判分程序核对用的标准答案。小模型每次生成两个词元，每个词元是一个汉字，可取“巴、黎、里、昂、马、赛、柏、林”中的任一个。它可以生成“巴林”这样的组合，到判分时才知道答得对不对。

在本书源码仓库根目录，用已经安装 PyTorch 的 Python 执行下列命令。

```bash
WORK="$(mktemp -d)"
cat > "$WORK/capital.jsonl" <<'JSONL'
{"prompt":"法国的首都是什么？","answer":"巴黎"}
{"prompt":"德国的首都是什么？","answer":"柏林"}
JSONL
python examples/ppo_capitals.py "$WORK/capital.jsonl" --batches 30
```

例如第一次抽到“巴”时，就把“巴”接到题目后面，预测下一个字。如果又抽到“黎”，就保存两个位置的概率，判分程序看到“巴黎”后给 1 分。训练时，把“题目 + 巴”重新送进 Transformer，一次前向就能重算这两个位置的概率。题目最后一个位置预测“巴”，“巴”的位置预测“黎”。

##### 分数怎样进入训练损失 <a href="#ppo-reward-loss" id="ppo-reward-loss"></a>

Transformer 的输入是题目和已经生成的词元。整条回答的 1 分留给训练程序。假设价值模型在“巴”和“黎”两个生成位置都预测能拿 0.5 分，那么实际得分减去预测，两个位置就各有 +0.5 的优势。如果整条得 0 分，在相同预测下，两个优势都是 −0.5。

下面摘出语言模型的更新，并展开了脚本里的 `clipped_policy_loss`。 `actions` 是模型实际生成的两个词元， `old_values` 是采样时价值模型给两个位置的预测。

```python
advantages = (reward[:, None] - old_values).detach()
new_logp = response_policy(actor, prefixes, start).log_prob(actions)
ratio = (new_logp - old_logp.detach()).exp()
clipped = ratio.clamp(0.8, 1.2)
loss = -torch.minimum(ratio * advantages, clipped * advantages).mean()
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

反向传播是对损失求导。SFT 从标准答案里取出正确词元来算损失。这个 PPO 例子先用标准答案给整条回答打分，再取模型**实际生成的词元**来算损失。比如模型生成“巴林”时，损失用的是“巴”和“林”的概率，判分程序返回的 0 分则通过 `advantages` 乘进损失。

只看生成“巴”这个位置对一个参数的影响。假设输出层有个权重 $$w$$ 对应的输入特征是 0.5，所以它对“巴”的原始分数的贡献是 $$0.5w$$。第一次更新前，“巴”的当前概率和旧概率都是 0.20，比值是 1。旧概率是采样时记下来的常数，当前概率仍然会随 $$w$$ 改变。由 softmax 的导数和上面的损失，在比值为 1 时，这一项的梯度为

$$
\frac{\partial L}{\partial w} = -A(1-p)\times 0.5.
$$

代入 $$p=0.20$$，优势为 +0.5 时，梯度是 −0.20。用学习率 0.1 做梯度下降，就把 $$w$$ 加 0.02。优势为 −0.5 时，梯度变成 +0.20，这次就把 $$w$$ 减 0.02。整批训练时，程序会合并所有生成位置对同一个参数的梯度。

每批每题生成 96 份答案，用于三次更新，然后重新采样。完整脚本还根据实际得分计算价值模型的损失，并调用优化器修改参数。

一次 CPU 实跑中，第一道题前四次生成了“黎黎、赛昂、昂里、里赛”。第一条的两个字在采样时的概率分别为 8.08% 和 9.34%，整条得 0 分。第一批法国题答对 1 次。跑完 30 批后，生成“巴黎”的概率从 0.93% 升到 99.20%，生成“柏林”的概率从 0.65% 升到 99.42%。这里测的是两道训练题上连续生成正确两个字的概率。

要给已有语言模型跑 PPO，可以用 [verl 的单卡 PPO 示例](https://verl.readthedocs.io/en/latest/start/quickstart.html)。它从 Qwen2.5-0.5B-Instruct 开始，让模型自己生成数学题的答案，再由程序核对。先用官方脚本准备训练和验证的 Parquet 文件，再运行文档里的 `python3 -m verl.trainer.main_ppo` 配方。换自己的任务时，要替换数据和判分规则，并指定生成模型和价值模型的起始权重。GPU 示例按官方文档核对，本地实跑的是上面的 CPU 例子。

#### 用程序打分做一次 GRPO <a href="#grpo-capital-recipe" id="grpo-capital-recipe"></a>

先从“法国的首都是什么？”看数据怎么准备。把问题写进 `prompt`，把正确答案写进 `answer`。模型只看问题，`answer` 留给评分程序。模型每次答完，程序把答案两端的空白去掉，只在答案恰好是“巴黎”时给 1 分，其余给 0 分。

把多道题整理成 `capital_train.jsonl`，放进 `DATA_DIR`。下面是一行样例和它的判分结果。

```json
{"prompt":[{"role":"user","content":"只写中文城市名。法国的首都是什么？"}],"answer":"巴黎"}
```

模型若生成“巴黎、里昂、巴黎、马赛”这四条回答，评分就是 1、0、1、0。TRL 的 `GRPOTrainer` 负责生成答案、调用判分函数、比较同一道题的四个分数，再更新模型。下一批用更新后的模型生成。

##### 在 CPU 上算一次参数更新 <a href="#capital-reward-update" id="capital-reward-update"></a>

下面复现 [8.2 节的算例](../08_alignment/8.2_rlhf.md#score-to-update)。三座城市用 0、1、2 表示。把代码保存为 `capital_update.py`，在装有 PyTorch 的环境里运行 `python capital_update.py`。这里训练的参数就是三个城市的 logits。

```python
import torch

logits = torch.nn.Parameter(torch.zeros(3))
sampled = torch.tensor([0, 1, 0, 2])  # 巴黎、里昂、巴黎、马赛
rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
advantages = (rewards - rewards.mean()) / rewards.std(correction=0)

optimizer = torch.optim.SGD([logits], lr=0.4)
log_probs = logits.log_softmax(dim=0)[sampled]
loss = -(advantages * log_probs).mean()
loss.backward()
optimizer.step()

probs = logits.detach().softmax(dim=0)
print([round(100 * p, 2) for p in probs.tolist()])  # [40.3, 29.85, 29.85]
```

手算用总体标准差，四个分数 1、0、1、0 的标准差为 0.5，优势为 +1、−1、+1、−1。下例采用 [TRL v1.7.0 的组内缩放](https://github.com/huggingface/trl/blob/v1.7.0/trl/trainer/grpo_trainer.py)，分母是样本标准差加一个很小的数，同一组的优势约为 +0.866、−0.866、+0.866、−0.866。

**接到自己的模型上。** 下面的代码替换[单卡 SFT 示例](#cuda-sft-recipe)的训练部分。在新的脚本和进程中，用同样的环境变量读取已会按要求回答的本地模型，保留示例中读取目录、加载分词器和模型、设置 LoRA 的代码，跳过 SFT 的数据加载和训练。加上下列代码后执行 `python grpo_local.py`。先跑十步检查流程，完整模型和生成时的内存都要留出来。

```python
from trl import GRPOConfig, GRPOTrainer

capital_data = load_dataset("json", data_files=str(
    data_dir / "capital_train.jsonl"), split="train")


def exact_reward(completions, answer, **kwargs):
    return [float(messages[0]["content"].strip() == target.strip())
            for messages, target in zip(completions, answer)]


grpo = GRPOTrainer(
    model=model, processing_class=tok, peft_config=lora,
    reward_funcs=exact_reward, train_dataset=capital_data,
    args=GRPOConfig(
        output_dir=str(run_dir / "grpo"), max_steps=10,
        per_device_train_batch_size=4, num_generations=4,
        gradient_accumulation_steps=1, max_completion_length=32,
        learning_rate=1e-6, beta=0.04, scale_rewards="group", loss_type="grpo",
        bf16=True, fp16=False, report_to="none", save_strategy="no",
        use_vllm=False, remove_unused_columns=False,
    ),
)
grpo.train()
grpo.save_model(str(run_dir / "grpo-final"))
```

上例用聊天消息格式的 `prompt`，评分函数从生成的助手消息里取文本。有效批量要能被每道题的生成数整除。`loss_type="grpo"` 先按每份回答的有效词元取平均，对应 [8.2 节的原始 GRPO 公式](../08_alignment/8.2_rlhf.md#rl-verifiable-rewards)。TRL v1.7.0 默认用 `"dapo"`，把批内有效词元合在一起取平均。`beta=0.04` 用来约束模型偏离训练起点的程度。GPU 配方按 [TRL v1.7.0 接口](https://huggingface.co/docs/trl/v1.7.0/grpo_trainer)核对。

**先查判分，再试训练。** 把答对、答错、空白和带解释的答案交给评分函数，逐项核对。这个评分规则适用于要求只输出标准城市名的练习。若允许别名或完整句子，应先调整规则。模型每次都答对，或每次都答错，同一组的分数就无法区分好坏。先看题是不是太简单或太难，再看采样有没有产生不同答案。

**训练后用没见过的题复测。** 用相同的提示和生成设置比较原模型与训练后的模型。训练时记录全对组和全错组的比例、回答长度和截断比例。奖励变高而新题正确率没有提高时，先查评分规则和数据有没有泄漏。

#### 把报修消息转成 JSON <a href="#ticket-json-lab" id="ticket-json-lab"></a>

有人发来“办公室的 Wi-Fi 连不上”，负责处理的人需要知道是什么问题、要不要马上处理。这样的报修记录也叫工单。这个实验给每条报修消息配上正确答案，教模型用两个 JSON 字段回答。`category` 表示类别，取 `network`（网络）、`account`（账号）或 `hardware`（设备）。`priority` 表示优先级。故障影响多人，或者导致工作完全中断且无替代方案时，填 `high`，其余填 `normal`。方法选择见[第八章](../08_alignment/README.md)。

##### 准备数据

训练前先把工单分开，以便用模型没见过的写法检验结果。在本书源码仓库根目录运行下列命令，生成 108 条训练样本、6 条验证样本和 18 条测试样本。三组分别使用不同的故障描述和影响描述。工单都是专为这个小实验编写的。

```bash
WORK="$(mktemp -d)"
python3 examples/ticket_post_training.py prepare "$WORK/data"
```

`train.jsonl` 的每一行是一条训练样本。用户消息给出规则和工单，助手消息把标准 JSON 答案写在文本字段里。训练时模型学习直接生成这段文本。

```json
{"messages":[{"role":"user","content":"把工单转成 JSON 对象，只输出 category 和 priority 两个字段。网络连接问题归为 network，账号登录与权限问题归为 account，设备故障归为 hardware。影响多人或导致工作完全中断且没有替代方案时，priority 为 high，其余为 normal。\n工单：办公 Wi-Fi 连不上。整个部门都受影响。"},{"role":"assistant","content":"{\"category\":\"network\",\"priority\":\"high\"}"}]}
```

##### 准备在本机运行的模型

可以在搭载 Apple 芯片（Apple Silicon）的 Mac 上完成实验。[MLX LM](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md) 提供了在这种机器上运行和微调语言模型的工具。这里用 Python 3.11，下载通义千问指令模型 [Qwen2.5-0.5B-Instruct 的 MLX 4 位权重](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-4bit)。已有兼容环境时可从下载模型开始。

```bash
python3.11 -m venv "$WORK/venv"
. "$WORK/venv/bin/activate"
python -m pip install "mlx==0.32.1" "mlx-lm==0.31.3"
hf download mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --revision a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 \
  --include '*.json' --include '*.safetensors' --include 'merges.txt' --include '*.jinja' \
  --local-dir "$WORK/model"
```

##### 先测原模型

在动手训练前，先把原模型的表现记录下来，之后才知道微调带来了什么变化。这个实验的训练设置已经定好，所以可以先保存原模型在测试集上的回答。做自己的任务时，可以先用验证集找错，最终测试集留到模型选好以后。

```bash
python examples/ticket_post_training.py predict "$WORK/data/test.jsonl" \
  --model "$WORK/model" --output "$WORK/base.jsonl"
python examples/ticket_post_training.py score "$WORK/data/test.jsonl" "$WORK/base.jsonl"
```

##### 用 SFT 和 LoRA 做微调

用工单和标准答案做监督微调（SFT）。训练时只更新附加到模型上的少量参数，也就是 LoRA 适配器。在同一终端训练 80 步。`--mask-prompt` 让模型读到提示词，只在助手回答的位置计算损失。训练结束后，适配器保存在 `$WORK/adapter/adapters.safetensors`。

```bash
mlx_lm.lora --model "$WORK/model" --train --data "$WORK/data" \
  --fine-tune-type lora --mask-prompt --num-layers 8 --batch-size 2 \
  --iters 80 --learning-rate 1e-4 --max-seq-length 256 --seed 42 \
  --steps-per-eval 40 --val-batches -1 --adapter-path "$WORK/adapter"
```

##### 用同一批工单复测

从保存的文件加载适配器，再回答一遍同一份测试集。两次都采用贪心解码，即每一步选择概率最高的词元。预测文件按测试集顺序保存模型原文，每行是一个 JSON 字符串。

```bash
python examples/ticket_post_training.py predict "$WORK/data/test.jsonl" \
  --model "$WORK/model" --adapter "$WORK/adapter" --output "$WORK/tuned.jsonl"
python examples/ticket_post_training.py score "$WORK/data/test.jsonl" "$WORK/tuned.jsonl"
```

先检查模型原文能否直接解析成没有重复键的 JSON 对象，再看字段是否符合约定。类别和优先级分别计分，都除以测试工单总数。格式符合约定、两个字段也都正确，才算整条通过。下表是一次本地实测的结果。

| 模型 | JSON 对象 | 字段符合约定 | 类别正确 | 优先级正确 | 整条通过 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原模型 | 0/18 | 0/18 | 0/18 | 0/18 | 0/18 |
| 加载适配器 | 18/18 | 18/18 | 18/18 | 9/18 | 9/18 |

原模型的 18 份回答都用了 Markdown 代码块，原文因此无法直接解析。去掉代码块后，原模型答对了 11 条的类别和 10 条的优先级。微调后可以直接读取 JSON，但模型把所有工单都判成了 `normal`，漏掉了紧急工单。下一轮可在验证集上分开看 `high` 和 `normal`，再决定补哪些示范。若用这 18 条的表现来决定怎样改模型，就把它们留作回归集，并另备一批未见工单做最终验收。

##### 换成自己的模型和任务 <a href="#own-model-sft" id="own-model-sft"></a>

如果你要让现有模型从订单通知里提取订单号和状态，准备一批真实写法和审核过的答案。用 MLX LM 时，可以把每条示范写成一行 JSONL。每行先给出任务要求和订单消息，然后给出模型应该生成的文本。下面是一条格式样例。

```json
{"messages":[{"role":"user","content":"从订单消息提取订单号和状态，只返回包含 order_id 和 status 的 JSON。订单 A102 已签收。"},{"role":"assistant","content":"{\"order_id\":\"A102\",\"status\":\"已签收\"}"}]}
```

把训练样本放到 `train.jsonl`，用来挑选训练设置的样本放到 `valid.jsonl`，最终验收用的样本放到 `test.jsonl`。为同一种状态准备不同说法，分文件时把同一订单的近重复记录放在同一组。测试样本使用训练时没见过的订单。检查用的提示里保留任务要求和订单内容，让模型自己生成答案。

下面沿用上面准备好的 MLX LM 环境。把 `MODEL_DIR` 改成 MLX LM 支持的本地指令模型目录，目录应包含权重、配置和分词器文件，并且能用聊天模板处理上面的消息。把 `DATA_DIR` 改成放这三个文件的目录。先用验证集里的一条订单试答，保存原模型的输出。

```bash
MODEL_DIR=/path/to/your-mlx-model
DATA_DIR=/path/to/your-jsonl-data
RUN_DIR="$(mktemp -d)"
PROMPT='从订单消息提取订单号和状态，只返回包含 order_id 和 status 的 JSON。订单 B207 正在配送。'
mlx_lm.generate --model "$MODEL_DIR" --prompt "$PROMPT" --max-tokens 64 --temp 0
```

用训练文件做一次小规模试跑。以下配置沿用上述工单实验的模型和参数规模，先执行 10 步，确认能读入数据、算出损失和保存适配器。若更换模型或答案长度，检查它支持的层数、上下文长度和微调方式。训练、验证样本连同模板都要放进 `--max-seq-length`。

```bash
mlx_lm.lora --model "$MODEL_DIR" --train --data "$DATA_DIR" \
  --fine-tune-type lora --mask-prompt --num-layers 8 --batch-size 2 \
  --iters 10 --learning-rate 1e-4 --max-seq-length 256 --seed 42 \
  --steps-per-eval 5 --val-batches -1 --adapter-path "$RUN_DIR/adapter"
```

权重和适配器配置都保存在 `$RUN_DIR/adapter`。把这个目录传给 `--adapter-path`，就能让原模型带着适配器回答同一个提示。执行成功说明加载流程通了。要判断训练效果，用同一份验证集比较微调前后，选好方案后再测独立测试集。

```bash
mlx_lm.generate --model "$MODEL_DIR" --adapter-path "$RUN_DIR/adapter" \
  --prompt "$PROMPT" --max-tokens 64 --temp 0
```

单轮 `user` 加 `assistant` 的短答任务可以沿用上面的 `predict` 命令批量生成。把模型、适配器和 `test.jsonl` 的路径换成自己的。订单示例可以分别比较 JSON 格式、订单号和状态。工单的 `score` 命令读取 `category` 和 `priority`，换了字段要按新任务写评分规则。

##### 用 Qwen 写论文摘要 <a href="#qwen-paper-summary" id="qwen-paper-summary"></a>

这个例子输入论文正文，输出一段中文摘要。摘要要交代研究问题、方法、主要结果和原文说明的局限。先用合成的短正文走完数据准备、微调和验收，再换成自己的论文。实验用上文同一个 Qwen2.5-0.5B-Instruct 的 MLX 4 位权重。

**准备正文和摘要。** `examples/data/paper_summaries.jsonl` 每篇论文只占一行。给同一篇论文的预印本和发表版分配同一个 `paper_id`，选用其中一版。按节提取材料时，先把选出的段落并进同一条记录的 `body`。若目的是让模型从正文总结，就从输入移除论文自带的摘要。抽查提取后的正文，确认表格的数字仍能对应到单位、分组和比较对象。

```json
{"paper_id":"train-pdf","split":"train","body":"研究比较了两种 PDF 表格提取方法。研究者标注了 120 篇英文论文的表格，并比较两种方法提取正确的比例。方法 A 的准确率为 89%，方法 B 为 81%。实验未覆盖扫描件。","summary":"研究在 120 篇英文论文上比较了两种 PDF 表格提取方法。方法 A 的准确率为 89%，比方法 B 高 8 个百分点。实验未纳入扫描件，结果能否推广到扫描文档仍待验证。"}
```

下列命令在本书源码仓库根目录运行。若已有上面的 MLX LM 环境和模型目录，可以跳过安装与下载，把 `MODEL_DIR` 设为现有目录。

```bash
PAPER_WORK="$(mktemp -d)"
python3.11 -m venv "$PAPER_WORK/venv"
. "$PAPER_WORK/venv/bin/activate"
python -m pip install "mlx==0.32.1" "mlx-lm==0.31.3"
MODEL_DIR="$PAPER_WORK/model"
hf download mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --revision a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 \
  --include '*.json' --include '*.safetensors' --include 'merges.txt' --include '*.jinja' \
  --local-dir "$MODEL_DIR"

python examples/paper_summary.py prepare examples/data/paper_summaries.jsonl \
  "$PAPER_WORK/data" --model "$MODEL_DIR" --max-seq-length 512
```

脚本把正文放进用户消息，把审核过的摘要放进助手消息，分别写出 `train.jsonl`、`valid.jsonl` 和 `test.jsonl`。它按模型的聊天模板计算长度，超过预算的样本会报错。512 词元适合本例的短正文。处理真实论文时，先测正文和目标摘要加上模板的长度，再选训练预算。篇幅超过可用上下文时，可以按节先摘取证据，再用证据生成摘要。训练和推理应采用同一种输入整理方式。

**先测原模型，再训练。** 此例的训练配置已定好，因此可以先保存原模型在测试集上的输出。换自己的数据时，用验证集找错误、挑设置，最终测试集留到方案定下以后。生成只使用用户消息，参考摘要留给验收。两个模型用相同的提示和输出长度上限。换成真实论文时，先检查这个上限会不会截断摘要。

```bash
python examples/paper_summary.py predict "$PAPER_WORK/data/test.jsonl" \
  --model "$MODEL_DIR" --output "$PAPER_WORK/base.jsonl" \
  --max-input-tokens 384 --max-new-tokens 128

mlx_lm.lora --model "$MODEL_DIR" --train --data "$PAPER_WORK/data" \
  --fine-tune-type lora --mask-prompt --num-layers 8 --batch-size 2 \
  --iters 30 --learning-rate 1e-4 --max-seq-length 512 --seed 42 \
  --steps-per-eval 15 --val-batches -1 --adapter-path "$PAPER_WORK/adapter"

python examples/paper_summary.py predict "$PAPER_WORK/data/test.jsonl" \
  --model "$MODEL_DIR" --adapter "$PAPER_WORK/adapter" \
  --output "$PAPER_WORK/tuned.jsonl" --max-input-tokens 384 --max-new-tokens 128
```

**按原文验收。** 逐篇看两个模型的摘要。数一数有多少说法得不到原文支持，再数漏了几项必须覆盖的信息。已经提到但写错的信息只计在第一项。把人工核对结果按 `paper_id` 分别写进 `base-review.jsonl` 和 `tuned-review.jsonl`。下列两行是一次实跑后填的适配器评分。脚本按论文编号核对三份文件，汇总空输出、问题数和通过的摘要数。

```json
{"paper_id":"test-water","unsupported_claims":2,"missed_key_points":1}
{"paper_id":"test-museum","unsupported_claims":0,"missed_key_points":0}
```

```bash
python examples/paper_summary.py review "$PAPER_WORK/data/test.jsonl" \
  "$PAPER_WORK/base.jsonl" "$PAPER_WORK/base-review.jsonl"
python examples/paper_summary.py review "$PAPER_WORK/data/test.jsonl" \
  "$PAPER_WORK/tuned.jsonl" "$PAPER_WORK/tuned-review.jsonl"
```

这次实跑用了 8 篇训练正文、2 篇验证正文和 2 篇测试正文。验证损失在第 15 步从 2.215 降到 1.901，第 30 步回升到 2.122。漏水检测的摘要里，原模型把比较检出次数写成了记录失败次数。加载适配器后，模型又把“未测量不漏水时的误报率”写成“未测量漏水时的误报率”，并漏掉了各做 50 次模拟漏水的实验规模。两种模型都只通过了 1 篇。真实论文的摘要质量需要用更大的一批独立样本测量。

**有可靠的判分再加 GRPO。** 先在验证集上试评分器，让同一篇论文的几份候选摘要同时包含正确、漏写和误报的版本，核对判分能否把它们区分开。可以先做一个容易核验的任务，例如同时输出关键结论的数值、单位和证据段落编号，再用这些字段核验结果。自由文本中的推断仍要和原文核对。奖励若只看格式或文字重合，模型可能得高分却遗漏主要发现。

要在 NVIDIA 上比较 SFT 与 SFT 加 GRPO，可用[单卡 SFT 示例](#cuda-sft-recipe)训练同样的问答数据。把生成的每条 `messages` 中的用户消息放进 `prompt`，助手消息放进 `completion`，两个字段都用数组包住。随后从 SFT 权重启动 [GRPO 配方](#grpo-capital-recipe)，把正文留在 `prompt`，把训练用的证据标注放进额外字段，由评分函数读取并为每份生成结果打分。若 SFT 用 PEFT 保存为适配器，先按[合并步骤](#sft-adapter-merge)得到完整权重。验收时用同一批留出论文比较两个模型。实跑覆盖上面的 MLX 流程，GPU 配方按 TRL 官方文档核对。公开的长文训练结果见 [Highlight-Then-Summarize](https://arxiv.org/html/2609.31382)。

##### 通过 API 提交 Gemini 监督微调作业 <a href="#gemini-sft-job" id="gemini-sft-job"></a>

如果希望把训练交给云平台，可以在 Google Cloud 上提交 Gemini 监督微调作业。数据脚本会把同一批训练样本另存为 `gemini_train.jsonl`，供这个作业使用。Gemini 的[文本训练数据格式](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/tuning/text)用 `contents` 和 `parts` 表示消息和消息里的内容，助手角色写成 `model`。下面的用户指令已缩短，实际文件保留完整指令。

```json
{"contents":[{"role":"user","parts":[{"text":"把工单转成 JSON。工单：办公 Wi-Fi 连不上。整个部门都受影响。"}]},{"role":"model","parts":[{"text":"{\"category\":\"network\",\"priority\":\"high\"}"}]}]}
```

先准备一个已开通计费和相应 API 的 Google Cloud 项目，以及存放训练数据的 Cloud Storage 存储桶。运行脚本的身份需要有提交作业和调用推理端点的权限，Google 的客户端通过[应用默认凭据](https://docs.cloud.google.com/docs/authentication/set-up-adc-local-dev-environment)识别这个身份。下面把合成训练集上传到存储桶。评测时，只把工单和指令发给推理 API，标准答案留在本机。

```bash
export GOOGLE_CLOUD_PROJECT=your-project
export GOOGLE_CLOUD_LOCATION=us
export GOOGLE_GENAI_USE_ENTERPRISE=True
TRAIN_URI=gs://your-bucket/ticket-lab/gemini_train.jsonl
python -m pip install --upgrade google-genai
gcloud storage cp "$WORK/data/gemini_train.jsonl" "$TRAIN_URI"
```

将下列代码保存为 `ticket_gemini.py`。脚本通过 Google 的 Python SDK 选择支持监督微调的 Gemini 3.5 Flash。`submit` 提交训练并打印作业名，`check` 查询作业状态。作业成功后，平台会返回一个可调用的地址，也就是调优模型的推理端点。此时可以让原模型和调优模型分别用同样的推理设置回答同一批工单。接口见 [Gemini 监督微调文档](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/tuning/supervised-tuning/use)。

```python
import argparse
import json
from pathlib import Path

from google import genai
from google.genai import types

base_model = "gemini-3.5-flash"
parser = argparse.ArgumentParser()
parser.add_argument("action", choices=("submit", "check"))
parser.add_argument("resource", help="训练数据 URI 或作业名")
parser.add_argument("--test", type=Path)
parser.add_argument("--output-dir", type=Path)
args = parser.parse_args()
if (args.test is None) != (args.output_dir is None):
    parser.error("--test 和 --output-dir 必须一起提供")
if args.action == "submit" and args.test is not None:
    parser.error("评测参数只用于 check")

if args.action == "submit":
    with genai.Client(http_options=types.HttpOptions(api_version="v1beta1")) as client:
        job = client.tunings.tune(
            base_model=base_model,
            training_dataset=types.TuningDataset(gcs_uri=args.resource),
            config=types.CreateTuningJobConfig(tuned_model_display_name="ticket-json"),
        )
    print(job.name)
else:
    with genai.Client(http_options=types.HttpOptions(api_version="v1")) as client:
        job = client.tunings.get(name=args.resource)
        print(job.state)
        if job.state in {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}:
            raise SystemExit(f"作业 {job.name} 失败或终止：{job.error}")
        if job.state != "JOB_STATE_SUCCEEDED":
            raise SystemExit(0)
        if not job.tuned_model or not job.tuned_model.endpoint:
            raise RuntimeError(f"作业 {job.name} 未返回推理端点")
        endpoint = job.tuned_model.endpoint
        print(endpoint)
        if args.test is not None:
            if args.output_dir.exists():
                raise FileExistsError(args.output_dir)
            with args.test.open(encoding="utf-8") as stream:
                prompts = [json.loads(line)["messages"][0]["content"] for line in stream]
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
            )
            predictions = {}
            for name, model in (("base", base_model), ("tuned", endpoint)):
                predictions[name] = [
                    client.models.generate_content(model=model, contents=prompt, config=config).text or ""
                    for prompt in prompts
                ]
            args.output_dir.mkdir()
            for name, outputs in predictions.items():
                with (args.output_dir / f"{name}.jsonl").open("x", encoding="utf-8") as stream:
                    for output in outputs:
                        print(json.dumps(output, ensure_ascii=False), file=stream)
```

提交一次作业，把返回的完整作业名填入 `JOB`。作业还在运行时，只执行查询命令。等作业成功后，在查询命令里加上评测参数。两份输出分别计分，用的是本地实验的评分规则，对比的是 Gemini 自己在微调前后的变化。

```bash
python ticket_gemini.py submit "$TRAIN_URI"
JOB="projects/your-project/locations/us/tuningJobs/your-job-id"
python ticket_gemini.py check "$JOB"
python ticket_gemini.py check "$JOB" \
  --test "$WORK/data/test.jsonl" --output-dir "$WORK/gemini"
python examples/ticket_post_training.py score "$WORK/data/test.jsonl" "$WORK/gemini/base.jsonl"
python examples/ticket_post_training.py score "$WORK/data/test.jsonl" "$WORK/gemini/tuned.jsonl"
```

Gemini 提供的是完整的训练作业。给定数据和配置后，平台负责运行训练。提交前先查[模型和地区是否支持微调](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/tuning/supervised-tuning)。

##### Tinker <a href="#tinker-training" id="tinker-training"></a>

如果需要自己决定每轮怎样采样、怎样算损失，可以用 [Tinker](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/trainingclient/)。先选一个服务支持的起始模型。Tinker 负责提供和管理算力，把采样、计算梯度和更新参数分别做成可调用的接口。读者可以自己把这些操作接成训练循环，也可以从 [Tinker Cookbook](https://tinker-docs.thinkingmachines.ai/cookbook/quickstart/) 的现成配方起步。训练后的 LoRA 适配器可以[下载并转换为推理框架支持的格式](https://tinker-docs.thinkingmachines.ai/cookbook/deployment/basics/)。

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

设请求 $$i$$ 的发送时间为 $$t_{i,0}$$，首个输出词元到达为 $$t_{i,1}$$，末个输出词元到达为 $$t_{i,n_i}$$，输出词元数为 $$n_i$$。客户端口径下：

$$
\mathrm{TTFT}_i=t_{i,1}-t_{i,0},\qquad
\mathrm{TPOT}_i=\frac{t_{i,n_i}-t_{i,1}}{n_i-1}\quad(n_i>1).
$$

单词元响应没有 TPOT，应单列而非填零。p95 TTFT 是请求 TTFT 样本的第 95 百分位；p95 TPOT 是请求内平均出词时间的第 95 百分位。它们都不是“95% 的词元间隔”，一次流式返回多个词元时更要保留分块计时口径，详见 [11.13 节](../11_serving/11.13_best_practices.md)。

若测量窗口长度为 $$T$$，成功完成请求集合为 $$S$$，输出吞吐定义为 $$Q_{\mathrm{out}}=\sum_{i\in S}n_i/T$$。同时报告失败率与在途未完成数，不能把失败请求直接丢掉制造高吞吐。“p95 吞吐”只有在定义了固定窗口的吞吐时间序列后才有意义；通常更有用的是指定到达率下的 p95 延迟和整体吞吐。

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

### A.7.8 正文迁出的实现对照

正文各章只讲设计决定及其理由；源码类名、配置键、启动参数、默认值与版本细节随版本变化，集中列在这里，按小节号排序。注了版本的条目以所注版本为准；未注版本的条目描述的是写作时各框架主线的行为。默认值与字段名常随版本变化，升级框架后先对照官方文档。

#### 2.1.5 节：注意力投影在常见实现里的名字

| 实现 | 模块名 | 权重形状 | 偏置 |
| --- | --- | --- | --- |
| Hugging Face Llama | `q_proj`、`k_proj`、`v_proj`、`o_proj`（`nn.Linear`） | `q_proj` 为 `[n_h × d_h, d_model]`，`k_proj`、`v_proj` 为 `[n_kv × d_h, d_model]` | 由 `config.attention_bias` 决定，Llama 系列为 `False` |
| Hugging Face GPT-2 | `c_attn`、`c_proj`（`Conv1D`） | `c_attn` 把 Q/K/V 融合成一个 `[768, 3 × 768]`，前向后按 `split` 拆成三份 | 有 |

两种常见实现里的注意力权重。`nn.Linear` 按“输出 × 输入”存放权重，与正文 $$XW$$ 的 `[输入, 输出]` 互为转置；GPT-2 的 `Conv1D` 按 `[输入, 输出]` 存放，且把三个投影融合成一个矩阵，一次矩阵乘法算出 Q、K、V。GQA 下 `k_proj` 与 `v_proj` 的输出宽度按 `n_kv` 而非 `n_h` 计算。

每头宽度的取值顺序：Hugging Face 先读配置里的 `head_dim`，没有才退回 `hidden_size // num_attention_heads`。Qwen3-4B 的配置是 `hidden_size` 2560、`num_attention_heads` 32、`head_dim` 128，`32 × 128 = 4096` 不等于 2560，由 `o_proj` 投回 2560。

#### 2.2.6 节：PyTorch `scaled_dot_product_attention` 与正文四步的对应

签名为 `scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False)`。

| 正文的步骤 | 接口里的对应物 | 要点 |
| --- | --- | --- |
| 输入的 Q、K、V | `query` 形状 `[..., Hq, L, E]`，`key` 形状 `[..., H, S, E]`，`value` 形状 `[..., H, S, Ev]` | 常见布局是 `[B, n_h, T, d_h]`；`L` 是查询数，`S` 是键值数，两者不必相等（这个 `S` 是 PyTorch 文档的记号，与正文的分数矩阵 $$S$$ 无关）；`E` 即 $$d_k$$，`Ev` 即 $$d_v$$ |
| 缩放因子 $$1/\sqrt{d_k}$$ | `scale` | 默认即 $$1/\sqrt{E}$$，$$E$$ 取 `query` 的最后一维；传入非默认值可实现 μP 那类缩放 |
| Softmax 之前加掩码 | `attn_mask` 或 `is_causal=True` | 布尔掩码中 `True` 表示参与注意力，与 `masked_fill` 的约定相反；浮点掩码直接加到分数上；两者不能与 `is_causal` 同时使用 |
| Softmax 之后的 dropout | `dropout_p` | 作用在注意力权重上，即 Softmax 之后、乘 $$V$$ 之前 |
| K/V 头数少于 Q 头数 | `enable_gqa=True` | 见正文 2.3.7 节 |

该函数在多个后端之间选择，包括不把 `[L, S]` 分数矩阵写回显存的融合实现与朴素的数学实现；后者在半精度输入下把中间结果保持为 FP32。Hugging Face 的 Llama 朴素注意力把 Softmax 写成 `softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)`，即在 FP32 下计算再转回原精度。

#### 2.4.3 节：Hugging Face 的掩码转换

Hugging Face 的掩码工具把布尔掩码转成“可见处为 0.0、屏蔽处为 `torch.finfo(dtype).min`”的浮点掩码再与分数相加，用有限极小值代替 $$-\infty$$，避免半精度下的 `inf` 在后续算子里变成 `NaN`。

#### 2.4.7 节：变长打包的内核接口

FlashAttention 的变长接口 `flash_attn_varlen_func` 用 `cu_seqlens`（各条序列长度的前缀和）表达边界，内核按段处理，天然不会跨段打分，既不需要显式的 `[n, n]` 块对角掩码，也跳过被遮住的块。重置 `position_ids` 只改变位置编码，不改变可见性，不能代替它。

#### 3.1.5 节：读一个真实分词器

Hugging Face 的 `tokenizer.json` 有四项要看：

- `model.vocab`：词元到 ID 的映射。
- `model.merges`：有序合并表，即 3.1.3 里 rank 的来源。
- `pre_tokenizer`：3.1.4 的预分词正则。
- `added_tokens`：特殊词元及其 ID。

tiktoken 按编码名组织，常见的有 `gpt2`、`cl100k_base`、`o200k_base`，其 `n_vocab` 分别为 50,257、100,277、200,019。速度方面，tiktoken 官方 README 给出的是比同类开源分词器快 3 到 6 倍。

GPT-2 的词表文件用 `bytes_to_unicode` 把 256 个字节一一映射到 256 个可打印的 Unicode 字符，再在映射后的字符上跑 BPE，所以词表文件看起来全是可见字符，实质仍是字节。预分词正则收录在 tiktoken 仓库的 `tiktoken_ext/openai_public.py`。SentencePiece 的字节回退由 `byte_fallback` 选项开启。

#### 3.2.7 节：嵌入与 LM head 的模块名

| 框架 / 模型 | 输入嵌入 | 位置嵌入 | 末端归一化 | 输出投影 | 相关配置字段 |
| --- | --- | --- | --- | --- | --- |
| PyTorch | `nn.Embedding(num_embeddings, embedding_dim)`，权重 `[V, d]` | — | — | — | `sparse=True` 启用稀疏行梯度 |
| HF Llama | `model.embed_tokens` | 无（RoPE） | `model.norm` | `lm_head` | `vocab_size`、`hidden_size`、`tie_word_embeddings` |
| HF GPT-2 | `wte` | `wpe` | `ln_f` | 与 `wte` 绑定 | 同上 |

Gemma 2 的 HF 实现把嵌入乘以 `config.hidden_size ** 0.5`，Llama 系列的 `LlamaModel` 不做缩放。

张量并行下，嵌入表按行切分 `V`，每卡只持有自己那段，越界 ID 的结果置零后做一次 all-reduce（见 7.3 节）；词表大小常被补齐到某个倍数，正是为了让这一步切得整齐。logits 张量 `[B, T, V]` 的显存：Llama 3 8B 在 `B = 4`、`T = 8192`、BF16 下是 `4 × 8192 × 128256 × 2 = 8,405,385,216` 字节，约 7.8 GiB。

#### 3.4.5、3.4.7 节：FFN 的字段与模块名

读一个模型的 FFN：`hidden_act` 给出激活函数（Gemma 的字段名是 `hidden_activation`）；MLP 类里有没有门控投影（`gate_proj`，或与 up 合存的 `gate_up_proj`）决定是否门控；MoE 模型的前若干层稠密层数由 `first_k_dense_replace` 一类字段给出（DeepSeek-V3 为 3，Kimi K2 为 1）。

HF 的 `LlamaMLP` 定义三个投影 `gate_proj`、`up_proj`、`down_proj`，激活由 `config.hidden_act` 选取（Llama 系列为 `"silu"`），中间宽度是 `config.intermediate_size`，偏置由 `config.mlp_bias` 控制。前向是 `down_proj(act_fn(gate_proj(x)) * up_proj(x))`；激活加在 gate 这一路，两路互换名字在数学上等价。推理引擎常把 `gate_proj` 与 `up_proj` 横向拼成一个矩阵，一次矩阵乘得到两段结果再切开，省一次内核启动与一次对 `x` 的重复读取，具体命名随引擎版本变化。

几个激活字段的含义：

| 字段 | 含义 |
| --- | --- |
| `gelu_new` | GELU 的 tanh 近似（GPT-2） |
| `gelu_pytorch_tanh` | 同一近似；Gemma 把它放在门控一路，即 GeGLU |
| `silu` | Swish₁，即 SwiGLU 的门控激活 |
| `relu2` | 平方 ReLU，$$\max(0, z)^2$$（Nemotron-4） |

Phi-4 把 `W_gate` 与 `W_up` 横向拼成一个 `[d, 2d_ff]` 矩阵存储，算法不变。gpt-oss 的配置虽写 `silu`，实现另有三处改动：门控取 $$z\,\sigma(1.702z)$$，且先把输入上限截断为 7；另一路截断到 ±7 之后再加 1。

#### 3.5.6 节：层实现里的残差

HF 的 `LlamaDecoderLayer.forward` 里出现两次 `residual = hidden_states` 与两次 `hidden_states = residual + hidden_states`，中间分别夹着 `input_layernorm` 加注意力、`post_attention_layernorm` 加 `mlp`。读一个新模型的层实现，先找这两对语句，就能确定归一化放在哪里。

#### 3.6.8 节：归一化的模块名、$$\epsilon$$ 与参数化

| 模型 | 模块 | 参数与初始化 | $$\epsilon$$ 字段 | 一层里的名字 |
| --- | --- | --- | --- | --- |
| HF Llama | `LlamaRMSNorm` | 一个 `weight`，初始化为全 1 | `rms_norm_eps`，Llama 2/3 为 `1e-5` | `input_layernorm`（注意力前）、`post_attention_layernorm`（FFN 前）、`model.norm`（末端） |
| HF Gemma 2 | `Gemma2RMSNorm` | `weight` 初始化为全 0，前向用 `(1.0 + weight)` 缩放 | 同类字段 | 另有 `pre_feedforward_layernorm`、`post_feedforward_layernorm` |
| HF GPT-2 | `LayerNorm` | `weight`、`bias` | `layer_norm_epsilon` | `ln_1`、`ln_2`、`ln_f` |

`LlamaRMSNorm.forward` 先把输入转成 `torch.float32`，算完平方均值与 `rsqrt` 后再转回原 dtype。Gemma 2 的 `(1.0 + weight)` 参数化在数学上等价于初始值为 1 的 $$\gamma$$，但权重衰减作用在 0 附近而不是 1 附近，正则化行为不同；移植权重时忽略这一点，整个模型的尺度会错。

#### 3.7.4 节：现代层在 `config.json` 里的字段

表 3-12 右边一列的各项可以由 `config.json` 直接确认：`hidden_act`（FFN 激活）、`rms_norm_eps`（RMSNorm）、`intermediate_size`（FFN 宽度）、`rope_theta`（RoPE）、`num_key_value_heads`（GQA）、`attention_bias` 与 `mlp_bias`（偏置）、`tie_word_embeddings`（权重绑定）。

偏置一行常被记反：原始 Transformer 的多头注意力投影不带偏置，只有 FFN 的 $$\max(0, xW_1+b_1)W_2+b_2$$ 带两个。给注意力投影加上偏置的是 GPT-2 一路：HF 实现里这些投影是 `Conv1D`，该模块无条件持有一个 `bias` 参数。

#### 3.7.5 节：GQA 的实现

算注意力分数时，每组 K/V 被复制 $$n_h / n_{kv}$$ 份与对应的 Query 头配对，HF 实现里这一步是 `repeat_kv`。判断一个模型用不用 GQA，看 `num_key_value_heads` 是否小于 `num_attention_heads`。

#### 4.2.5 节：可学习位置表的配置键与 OPT 的位置偏移

| 模型 | 位置表配置键 | 模块名 | 备注 |
| --- | --- | --- | --- |
| GPT-2 | `n_positions`（等价写法 `max_position_embeddings`） | `transformer.wpe`，与词嵌入 `transformer.wte` 并列 | 只相加词嵌入与位置两张表 |
| BERT | `max_position_embeddings` | `embeddings.position_embeddings` | 另有段嵌入 `token_type_embeddings`（`type_vocab_size` 为 2），三张表相加后再过 LayerNorm 与 Dropout |

带 KV 缓存解码时，新词元的位置号是 `arange(1) + past_seen_tokens`，GPT-2 的实现即如此。

OPT 的位置表长度是 $$L_{\max} + 2$$，查表时位置号统一加 2；位置号由 `cumsum(attention_mask) * attention_mask - 1` 算出，从而跳过左侧填充。多乘的那一次掩码不能省：省掉它，掩码为 0 的位置会继承前一格的累加值，而不是统一归到 −1。照抄别的模型的 `position_ids` 会错两位。

#### 4.3.5 节：Llama 3.1 的分段缩放

Llama 3.1 的 `rope_scaling` 写着 `rope_type = "llama3"`、`factor = 8`、`low_freq_factor = 1`、`high_freq_factor = 4`、`original_max_position_embeddings = 8192`。记 $$s = 8$$、$$f_{\text{lo}} = 1$$、$$f_{\text{hi}} = 4$$、$$L_0 = 8192$$：波长 $$\lambda < L_0/f_{\text{hi}} = 2048$$ 的对不动，$$\lambda > L_0/f_{\text{lo}} = 8192$$ 的对频率除以 8，两者之间按波长线性过渡：

$$w_i = \frac{L_0/\lambda_i - f_{\text{lo}}}{f_{\text{hi}} - f_{\text{lo}}}, \qquad \theta_i' = (1 - w_i)\frac{\theta_i}{s} + w_i \theta_i$$

代入 $$b = 500{,}000$$、$$d_h = 128$$，第 0 至 28 对不动，第 29 至 34 对在斜坡上，第 35 至 63 对除以 8。斜坡带的一例是第 32 对：$$\theta = 1.414 \times 10^{-3}$$，$$\lambda = 4{,}443$$，$$w_{32} = (8192/4443 - 1)/(4 - 1) = 0.281$$，$$\theta' = 0.719 \times \theta/8 + 0.281\,\theta = 5.248 \times 10^{-4}$$，相当于只除以 2.69。Meta 参考实现对这组常数的注释是由网格搜索得到。

#### 4.3.6 节：RoPE 的配置键与实现细节

| 配置键 | 含义 | 取值示例 |
| --- | --- | --- |
| `rope_theta` | 底数 $$b$$ | Llama 3 为 `500000.0`；GPT-NeoX 写作 `rotary_emb_base`，为 `10000` |
| `rope_scaling` | 缩放方案与参数 | `rope_type` 取 `ROPE_INIT_FUNCTIONS` 的键：`linear`、`dynamic`、`yarn`、`longrope`、`llama3`、`proportional`；不写则按 `default` 走不缩放的原始公式 |
| `max_position_embeddings` | 声明支持的最大位置 | Llama 3.1 为 131072 |
| `partial_rotary_factor` | 只旋转每个头的前若干维 | GPT-NeoX 写作 `rotary_pct`，20B 模型取 0.25 |

RoPE 相关的常见配置键，取自各模型公开的 `config.json` 与 Transformers 的 `modeling_rope_utils.py`。

- **配对布局。** Hugging Face 的 `rotate_half` 把前半与后半配对（第 0 与第 64 维、第 1 与第 65 维……），cos/sin 表由 `[T, d_h/2]` 拼成 `[T, d_h]`。把 Meta 原始权重转成 HF 格式时，要对 `q_proj`、`k_proj` 做相应的行置换。
- **fp32 的 cos/sin 表。** Transformers 的 Llama 实现在计算这张表时显式关闭自动混合精度，算完再转回激活的 dtype。
- **形状。** `Q[T, d_h] ⊙ cos[T, d_h] → [T, d_h]`；cos、sin 两张表各有 `[T, d_h/2]` 个不同的数，复制成 `[T, d_h]` 与 Q 对齐。

#### 4.4.1 节：ALiBi 的实现形态

偏置是一个与内容无关的常量张量，按 `[n_h, T, T]` 加到掩码上；因 Softmax 对一行加常数不变，实现只需键的位置。BLOOM 构造的张量形状是 `[B × n_h, 1, T]`，只含键位置，靠广播对上查询维。FlashAttention 以 `alibi_slopes` 参数原生支持这项偏置，接受形状 `(n_h,)` 或 `(B, n_h)` 的 fp32 斜率。

#### 4.4.2 节：T5 相对位置偏置的分桶

| 相对距离 | 0 | 1 | 15 | 16 | 17 | 20 | 24 | 32 | 64 | 96 | 128 及以上 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 桶号 | 0 | 1 | 15 | 16 | 16 | 17 | 19 | 21 | 26 | 29 | 31 |

T5 的相对距离分桶（32 个桶、最大距离 128、因果方向），按 Transformers 的 `_relative_position_bucket` 规则算出。前 16 个桶一一对应距离 0 到 15，其余 16 个桶覆盖 16 到 127、宽度按对数增长，128 以上全部落进第 31 号桶。配置键是 `relative_attention_num_buckets` 与 `relative_attention_max_distance`。

#### 4.4.3 节：分层混合位置方案的配置

- **Llama 4。** `no_rope_layers` 逐层标出哪些层用 RoPE，`no_rope_layer_interval` 默认 4（每四层去掉一次）；`attention_chunk_size` 默认 8192，带 RoPE 的层做分块局部注意力。SmolLM3 用同一组配置键。
- **Cohere Command 系列**（`cohere2` 架构）。按层类型判断是否调用 `apply_rotary_pos_emb`：滑窗层施加 RoPE，全局层不施加。
- **部分旋转。** 配置键 `partial_rotary_factor`，GPT-NeoX 写作 `rotary_pct`。
- **Gemma 3。** 滑窗层与全局层各配一个 $$\theta$$：全局层 1,000,000，滑窗层 10,000。

#### 5.1.7 节：因果语言模型的标签移位与忽略标记

Hugging Face 的因果语言模型接口只接收一个与 `input_ids` 等长的 `labels`，移位在损失函数里完成：先把 `labels` 右侧补一格 −100，再取 `labels[..., 1:]` 与 logits 对齐。以 `A B C D` 为例，logits 有 4 行，`labels` 补位后是 `A B C D −100`，去掉第一格得 `B C D −100`，与 4 行 logits 逐行对齐，第 4 行不计损失，共 3 个监督位置。交叉熵的 `ignore_index` 默认取 −100，被标记的位置既不进分子也不进分母。

#### 5.2.7 节：MLM 名额分配的默认参数

BERT 官方的 `create_pretraining_data.py` 默认 `max_seq_length = 128`、`masked_lm_prob = 0.15`、`max_predictions_per_seq = 20`（$$128 \times 0.15 = 19.2$$ 向上取整）。Hugging Face 的 `DataCollatorForLanguageModeling` 用三个参数表达 80/10/10：`mlm_probability = 0.15`、`mask_replace_prob = 0.8`、`random_replace_prob = 0.1`，剩下的 0.1 保持不变；它在每个批次现场采样，属于动态掩码。

#### 5.3.2 节：T5 的哨兵词元

哨兵是加进词表的特殊词元，不对应任何 wordpiece。T5 的 SentencePiece 词表有 32,000 个词，Hugging Face 的 `T5Tokenizer` 默认追加 `extra_ids = 100` 个哨兵，名字从 `<extra_id_0>` 起；`t5-base` 的 `config.json` 里 `vocab_size` 是 32128，即 32,100 再向上对齐。训练时 `T5ForConditionalGeneration` 由 `labels` 右移得到 `decoder_input_ids`。

#### 6.1.5 节：Muon 在 Megatron 中的实现

NVIDIA Megatron Core 为 Muon 配了一个**逐层分布式优化器**：把整层指派给某个数据并行进程，整层的正交化不需要额外通信就能算出，绕开了 ZeRO/FSDP 按元素分片与“正交化需要完整矩阵”之间的冲突。Megatron-LM 命令行一侧的对应项：

| 参数 | 含义 |
| --- | --- |
| `--optimizer muon` | 选用 Muon |
| `--muon-num-ns-steps` | Newton-Schulz 迭代次数，默认 5 |
| `--muon-scalar-optimizer` | 一维参数交给哪个优化器，默认 `adam` |

#### 6.2.4 节：学习率调度在框架里的名字

- Hugging Face `transformers.optimization`：`get_cosine_schedule_with_warmup`、`get_linear_schedule_with_warmup`、`get_inverse_sqrt_schedule`、`get_cosine_with_min_lr_schedule_with_warmup`、`get_wsd_schedule`，通过 `lr_scheduler_type` 选择；WSD 的枚举值是 `SchedulerType.WARMUP_STABLE_DECAY`。
- Megatron-LM：`--lr-decay-style` 取 `constant`、`linear`、`cosine`、`inverse-square-root`、`WSD`；`--min-lr` 指定终值，`--lr-warmup-iters` 或 `--lr-warmup-fraction` 指定预热长度。选 WSD 时 `--lr-wsd-decay-style`（`exponential`、`linear`、`cosine`、`minus_sqrt`）与 `--lr-wsd-decay-iters` 单独控制衰减段。

#### 6.3.2 与 6.3.6 节：分布式梯度裁剪与 soft-cap 的配置

- 参数被张量并行、流水线并行或 FSDP 切开后，全局范数要先跨卡归约平方和再开方。FSDP 提供自己的 `clip_grad_norm_` 方法；Megatron-LM 用 `--clip-grad` 指定阈值（默认 1.0），在分布式优化器内部完成跨组归约。`torch.nn.utils.clip_grad_norm_()` 返回裁剪之前的总范数，可直接作为监控量。
- Gemma 2 的 soft-cap 在 Hugging Face 配置里对应 `attn_logit_softcapping`（默认 50.0）与 `final_logit_softcapping`（默认 30.0）；FlashAttention 的 Python 接口为注意力层的 soft-cap 提供 `softcap` 参数。

#### 6.4.2 节：打包的序列边界

FlashAttention 的变长接口用 `cu_seqlens`（各序列的累计起点）描述打包后的序列边界，等价于块对角的注意力掩码。只重置 `position_ids` 能让位置编号从 0 重新开始，但阻断不了样本之间的注意力。

#### 7.1.4 与 7.1.5 节：NCCL 与 DDP

- NCCL 的 [`NCCL_ALGO` 环境变量](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)列出 Ring、Tree 等算法，默认按拓扑自动选择。
- DDP（[文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)）按 `model.parameters()` 的逆序装桶，桶大小由 `bucket_cap_mb` 控制，默认 25 MiB。
- `no_sync()` 上下文管理器让其内部的反向只在本地累加梯度，退出后的第一次反向再统一同步。
- `find_unused_parameters=True`：每步遍历 autograd 图，找出本步不产生梯度的参数并提前标记就绪，含条件分支的模型需要它，代价是每步多一次图遍历。
- `gradient_as_bucket_view=True`：梯度张量直接作为桶的视图，省掉一份与梯度等大的拷贝。

#### 7.2 节：DeepSpeed ZeRO 与 FSDP2 的接口

DeepSpeed（[配置文档](https://www.deepspeed.ai/docs/config-json/)）：

| 配置项 | 对应的机制 |
| --- | --- |
| `zero_optimization.stage` | 0 到 3：关闭、分优化器状态、再分梯度、再分参数 |
| `overlap_comm` | 梯度归约是否与反向计算重叠 |
| `reduce_bucket_size` | 一次归约的元素数，作用与 DDP 的桶相同 |
| `reduce_scatter` | ZeRO-1 的梯度归约用 Reduce-Scatter 而非 AllReduce，默认开启 |
| `stage3_prefetch_bucket_size` | 参数预取缓冲的大小，调小省显存，但计算更容易等通信 |
| `stage3_max_live_parameters` | 每卡同时驻留的完整参数上限，即 7.2.3 峰值公式里的第二项 |
| `offload_optimizer`、`offload_param` | ZeRO-Offload／Infinity 的卸载；后者只在 stage 3 下有效 |

PyTorch FSDP2（`fully_shard`，[文档](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)）：

- 初始化时 `model.parameters()` 原地从 `torch.Tensor` 转成沿第 0 维切分的 `DTensor`；前向和反向之前由 pre-hook 做 All-Gather 并转回普通张量，之后由 post-hook 释放未分片的参数。
- 每调用一次 `fully_shard(module)` 形成一个通信组，组内参数用一次 All-Gather 取回；文档要求自底向上逐层调用，不要只包最外层。
- `reshard_after_forward`：`True` 是标准的 ZeRO-3；`False` 前向之后不释放，省掉反向的 All-Gather，通信量回到 2Ψ；根单元默认取 `False`；取整数则前向之后重分片到更小的组（例如节点内 8 卡）。
- `MixedPrecisionPolicy`：`param_dtype` 决定 All-Gather 与计算的精度，`reduce_dtype` 决定梯度归约的精度；分片参数本身保持高精度，兼作主权重。
- HSDP 用二维 device mesh 表达：第 0 维是节点、第 1 维是节点内的卡，参数的 placement 为 `(Replicate(), Shard(0))`，即沿 mesh 第 1 维分片、沿第 0 维复制；一维 mesh 的全分片是 `(Shard(0),)`。`Shard(0)` 里的 0 指参数张量自身的第 0 维，与 mesh 的维度编号无关。

#### 7.3 节：张量并行的源码与接口

- Megatron-LM：7.3.3 的 f、g 是 `megatron/core/tensor_parallel/mappings.py` 里的两个 `torch.autograd.Function`，`_CopyToModelParallelRegion`（前向恒等、反向 `_reduce`）与 `_ReduceFromModelParallelRegion`（反之）。层定义在 `megatron/core/tensor_parallel/layers.py`：`ColumnParallelLinear`、`RowParallelLinear`、`VocabParallelEmbedding`；分片 logits 上的交叉熵是 `VocabParallelCrossEntropy`。并行度由 `tensor_model_parallel_size` 配置，序列并行由 `sequence_parallel` 开启；`tp_comm_overlap` 把通信切碎后与同层矩阵乘法交错执行，只能在序列并行开启时使用。
- PyTorch 原生：`torch.distributed.tensor.parallel` 的 `parallelize_module`，配合 `ColwiseParallel`、`RowwiseParallel`、`SequenceParallel`，以及在分片 logits 上算交叉熵的 `loss_parallel`（[文档](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html)）。

#### 7.5 节：重计算的接口

- PyTorch `torch.utils.checkpoint.checkpoint` 默认 `preserve_rng_state=True`，在检查点处保存并恢复随机数状态；[文档](https://docs.pytorch.org/docs/stable/checkpoint.html)建议使用 `use_reentrant=False` 的实现。
- Megatron-LM 的 `CudaRNGStatesTracker` 让张量并行区域内的 dropout 在各卡上取不同种子，区域外取相同种子。
- Megatron-LM 用 `recompute_granularity` 选粒度：`selective` 重算 `recompute_modules` 指定的子模块（默认 `core_attn`）；`full` 重算整层，由 `recompute_method` 决定哪些层参与：`uniform` 把各层均匀分组、每组存一个检查点，`block` 让每个流水线级只对指定数量的层重算、其余层照常保存。

#### 7.6 节：AMP 的接口与默认值

- `torch.amp.GradScaler` 的默认值：`init_scale` 65,536，`backoff_factor` 0.5，`growth_factor` 2.0，`growth_interval` 2,000 步（[文档](https://docs.pytorch.org/docs/stable/amp.html)）。
- 与梯度累积、裁剪配合的调用顺序：全部微批量完成 `scaler.scale(loss).backward()` 后，调用一次 `scaler.unscale_(optimizer)`，再做梯度范数监控与裁剪，最后 `scaler.step(optimizer)`、`scaler.update()`。
- `torch.autocast` 在 CUDA 上：矩阵乘法、卷积、`linear` 等用指定的低精度；`softmax`、`log_softmax`、`layer_norm`、`cross_entropy`、`exp`、`log`、`pow`、`sum`、`norm` 等用 FP32；另有一类按多个输入中最宽的类型执行，未列入规则的算子跟随输入。列表随设备后端而异，CUDA 的列表不能直接当成 CPU 的规则；显式指定 `dtype` 与原位算子也有例外。原生 AMP 下不要再对模型手动调用 `.half()` 或 `.bfloat16()`。

#### 7.7.3 节：分布式检查点（DCP）

`torch.distributed.checkpoint`（[文档](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)）：`save` 让每个 rank 只保存本地分片，每个检查点至少每 rank 一个文件；`async_save` 先把状态拷到暂存区（默认是 CPU 锁页内存），再在单独的线程里写入。状态对象只要实现 `state_dict` 与 `load_state_dict` 就能一并保存，数据加载器位置和随机数状态可照此接入。

#### 8.1.2 节：TRL SFTTrainer 的三种数据格式与掩码开关（v1.7.0）

**进阶：框架对照。** 以下到本小节末是 TRL 里的具体开关，换框架时名字会变，第一遍阅读可以跳过。[TRL 的 `SFTTrainer` 文档](https://huggingface.co/docs/trl/sft_trainer)区分以下情况（对应 [v1.7.0 实现](https://github.com/huggingface/trl/blob/v1.7.0/trl/trainer/sft_trainer.py)；下列默认行为针对未预先提供 `labels` 的文本数据）：

- **语言建模格式**：单个 `text` 字段或完整 `messages` 列表，默认对整个序列的有效目标计损失；出现 `user`、`assistant` 角色并不会自动改成只训练回答。
- **提示词—补全格式**：分开的 `prompt` 与 `completion` 字段，`completion_only_loss=None` 默认只训练补全部分；设置 `completion_only_loss=False` 才取消这层提示词屏蔽。
- **只训练助手消息**：对话数据需显式设置 `assistant_only_loss=True`，且模板要用 `{% generation %}` 与 `{% endgeneration %}` 标出助手生成区间。TRL 可为部分已支持的模型族自动换用带这些标记的训练模板，其他模板需自行核验。若同时启用补全掩码，只训练两个掩码的交集。

**多轮对话的监督范围由这个选择决定。** 8.1.2 节的配方采用完整 `messages` 加助手掩码，因此可训练多个助手回合，产生多段不连续的计损失区间；若把历史放进 `prompt`、最后一轮回答放进 `completion` 并只训练补全，就只监督最后一轮。训练前应抽取实际批次，逐词元对照 `input_ids` 与 `labels`：用户与系统位置是否为 −100，目标助手回合是否保留，以及回答结束词元是否仍在监督范围内。

#### 8.3.5 节：TRL DPOTrainer 的配置项

框架侧，TRL 的 `DPOTrainer` 把 $$\beta$$、损失变体与参考对数概率的预计算都做成了配置项。参数名随版本变动，机制不变：**一个 $$\beta$$、一种损失形状、一份参考对数概率是否落盘。**

#### 9.1.2 节：vLLM 采样器的两组处理器与 logprobs 的取值位置

> [!NOTE]
> **进阶：vLLM 的两处实现。** vLLM 的采样器按“会改变 argmax 吗”把处理器分成两组，先执行会影响贪心结果的，再按请求分出贪心与随机两条路径。`logprobs` 默认在语法掩码之后、①的其余处理与②③④之前就算好：无约束请求拿到的是模型的原始分布，约束请求拿到的是已遮掉不合法词元的分布，两者都不是实际抽样所用的分布。

#### 9.1.3 节：Llama 3 的两个结束词元 ID 与 HF 的 unfinished_sequences

**结束词元可以不止一个。** Llama 3 的[官方分词器](https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py)定义 `stop_tokens = {<|end_of_text|>, <|eot_id|>}`。特殊词元排在 128,000 个基础词元之后，前者是第 2 个，ID 为 128001；后者是第 10 个，ID 为 128009。`<|end_of_text|>` 表示整段文本结束，`<|eot_id|>` 表示一轮对话结束，对话模板在每条消息末尾写的是后者。对话模型答完一轮时选出的是 `<|eot_id|>`；若停止集合里只配了 `<|end_of_text|>`，循环不会停，模型继续往下写，直到撞上长度上限。“模型停不下来”多数是这类配置问题，不是模型问题。

**批内先结束的序列。** 静态批处理中，先结束的序列不能离开，只能继续占位。Hugging Face `generate()` 用一个 `unfinished_sequences` 向量标记，已结束的行此后每轮填入 `pad_token_id`，整批要等最长的一条。连续批处理让结束的序列立即退出、新请求补位（见 [11.2 节](../11_serving/11.2_continuous_batching.md)）。

#### 9.1.4 节：Hugging Face 的 padding_side 与右填充警告

实现上，Hugging Face 的分词器用 `padding_side="left"` 设置；`generate()` 发现仅解码器模型的输入是右填充时会给出警告。“单条推理正常、批量推理胡言乱语”这类故障，多半错在填充方向，模型和权重没有问题。

#### 9.1.5 节：vLLM detokenize_incrementally 的两个下标

**做法。** 增量反分词不解码单个词元，而是解码一个窗口。vLLM 的 [`detokenize_incrementally`](https://github.com/vllm-project/vllm/blob/main/vllm/tokenizers/detokenizer_utils.py) 维护两个下标 `prefix_offset` 和 `read_offset`：

1. 把 `prefix_offset` 到 `read_offset` 的词元解码成前缀文本。
2. 把 `prefix_offset` 到末尾的词元解码成新文本。
3. 新文本若以 U+FFFD 结尾，说明末尾的字节序列尚未完整，本轮不输出，两个下标不动。
4. 否则输出新文本比前缀文本多出的部分，并把两个下标前移。

窗口带上几个已输出的词元，是为了让空格和合并规则在正确的上下文里生效。

#### 9.2.2 节：Hugging Face 束搜索的缓存重排与 early_stopping

**KV 缓存要跟着重排。** 新一轮的第 1 条序列来自旧的第 2 条，它要读的是旧第 2 条的缓存。朴素实现每轮按 `beam_idx` 把各层的 K、V 重新收集一遍，Hugging Face 里对应 `_reorder_cache` 或缓存对象的 `reorder_cache(beam_idx)`。若两条新序列来自同一个父序列，父序列的缓存还要复制一份。分页 KV 缓存把这件事变成改块表里的指针，只在写入共享块时才复制（见 [11.4 节](../11_serving/11.4_kv_memory_management.md)的写时复制）。

**何时结束。** 已完成集合攒够 W 条并不一定能停：仍在扩展的序列可能在归一化之后反超。Hugging Face 用 `early_stopping` 控制：取 `True` 时攒够 `num_beams` 条即停；取 `False` 时用启发式判断继续搜索是否还可能更优；取 `"never"` 时只有确认不可能更优才停。结束后按归一化得分返回前 `num_return_sequences` 条。

#### 9.2.3 节：Hugging Face 的 length_penalty 方向

**实现中的参数名容易读反。** Hugging Face 的 `length_penalty` 是“得分除以 $$|y|^{\alpha}$$”这一长度归一化形式（[9.2.3 节](../09_decoding/9.2_greedy_beam.md)）里的指数：得分除以 `长度 ** length_penalty`，默认 1.0。得分是负数，所以取正值偏向更长的序列，取负值偏向更短。名字叫“惩罚”，调大却得到更长的输出。

#### 9.3.1 节：温度趋零时 vLLM 与 HF 的处理

**两个极限。** $$\tau \to 0$$ 时最大项的指数压倒其余各项，分布收缩到 argmax，等同贪心。实现里不会真的除以 0：vLLM 把温度低于 `1e-5` 的请求直接送入贪心分支；Hugging Face 要求温度严格为正，贪心由 `do_sample=False` 指定。$$\tau \to \infty$$ 时各项趋于相等，分布趋于均匀，Llama 3 词表下每个词元约 `1 ÷ 128,256 ≈ 7.8 × 10⁻⁶`。

#### 9.3.3 节：Top-p 排序成本与 vLLM 的 PyTorch 原生路径

**成本。** 求核需要按概率排序。Llama 3 的 logits 有 128,256 列，即 128,000 个基础词元加 256 个保留的特殊词元（见 9.1.3）。对这么多项做比较排序约 `n log₂ n ≈ 2.2 × 10⁶` 次比较，每行每轮一次，是采样阶段最贵的一步。vLLM 的 PyTorch 原生路径 `apply_top_k_top_p_pytorch` 在文档字符串里直接写明“排序在大批量下可能很慢”；9.3.8 会看到免排序的替代做法。

#### 9.3.6 节：HF 与 vLLM 惩罚项的统计范围

**统计范围不同。** Hugging Face 的重复惩罚默认把 Prompt 里的词元也算作“出现过”，要排除须传 `prompt_ignore_length`。vLLM 的重复惩罚同样统计 Prompt 加输出，频次与存在惩罚则只统计输出。迁移参数时要先确认范围。

#### 9.3.8 节：引擎路径上的四个实现细节（multinomial、FP32/FP64、FlashInfer、词表并行）

**进阶：几个实现细节。** 下面四条是引擎路径上的细节，第一遍可以跳过。

- [9.3.8 节](../09_decoding/9.3_sampling.md)的指数竞赛（对每个词元抽 $$q_i \sim \text{Exp}(1)$$，取 $$p_i/q_i$$ 的 argmax），PyTorch 自己也用。`torch.multinomial` 在只抽 1 个样本或无放回抽样时，走的正是 `q = empty_like(p).exponential_(1)`、相除、argmax 这条快速路径，源码注释写明它来自 Gumbel-Softmax。vLLM 仍然自己实现一遍，文档字符串给的理由是 `torch.multinomial` 会引入 CPU 与 GPU 的同步；自己实现还便于给个别请求换上独立的随机数发生器。
- Softmax 强制用 FP32。FP16 能表示的最小正数约 `6 × 10⁻⁸`，比它小的概率直接变成 0；BF16 只有 7 位尾数，相对精度约 0.8%。12.8 万项的长尾在半精度下会整段失真，vLLM 在处理链一开始就把 logits 转成 FP32，另有 `use_fp64_gumbel` 选项让噪声用 FP64。
- 指定了 Top-k 或 Top-p 时，默认路径并不是 9.3.8 节的指数竞赛实现。CUDA 上若 FlashInfer 可用，`TopKTopPSampler` 走 `forward_cuda`，由 FlashInfer 的拒绝采样内核一步完成截断与抽样，免去 9.3.3 的整表排序。拒绝采样是指先抽出一个候选，不满足条件就丢弃重抽，直到抽中满足条件的为止。源码注明它与 `random_sample` 只是统计等价，同一个种子下两条路径的输出不保证相同。指数竞赛用于未指定 Top-k 与 Top-p 的请求，以及 `forward_native` 回退路径；带 `seed` 的请求（FlashInfer 不支持逐请求随机数发生器）和开启 `use_fp64_gumbel` 时也回到这条路径。
- 词表沿张量并行切分时，裸的全词表抽样可以用“每卡局部最大值加一次全局带下标的 max 规约”完成。一旦有 Top-p、禁词表或约束掩码，就需要全局的候选集合或概率质量，实际通信收益取决于框架的实现。

#### 9.3.9 节：vLLM 逐请求种子

**种子。** 采样的随机性来自指数竞赛里的噪声 $$q$$（[9.3.8 节](../09_decoding/9.3_sampling.md)）。vLLM 的 `seed` 参数为该请求单独建一个随机数发生器，只改写 $$q$$ 中属于它的那一行；源码里的 TODO 写明这条路径逐请求处理、较慢。同一个种子只在同一引擎、同一条代码路径下可复现，换引擎或换路径都不保证。

#### 9.4.7 节：vLLM structured_outputs 字段与后端

> [!NOTE]
> **进阶：vLLM 的字段与后端。** vLLM 把约束放在 `SamplingParams.structured_outputs` 里，类型是 `StructuredOutputsParams`，字段 `json`、`regex`、`choice`、`grammar`、`json_object`、`structural_tag` 各对应一种约束；后端由 `StructuredOutputsConfig.backend` 选择，可取 `auto`、`xgrammar`、`guidance`、`outlines`、`lm-format-enforcer`，默认 `auto`。旧版本把这项功能称作引导解码（guided decoding），参数名为 `guided_json`、`guided_regex` 等，这些字段已在 v0.12.0 移除。读旧文档和旧代码时两套名字指的是同一件事。

#### 9.6.3 节：DiffusionGemma 模型卡的采样器设置

**采样器。** 模型卡给出的配置把 9.6.1 的“保留最有把握的”落到了可操作的判据上：去噪步数上限 48；logits 的温度从 0.8 线性降到 0.4；每一步选出熵最低的一批位置，未选中的位置全部重新掩上；当画布上的平均熵低于阈值、且各位置的最高概率词元连续两步不变时，提前停止。

两处判据针对的都是 9.6.1 的条件独立近似。每步选多少个位置不固定：按熵从低到高选入，并使这批位置之间互信息的上界保持在 0.1 以下。互信息量的正是独立采样忽略掉的位置间依赖，而一个位置与其他位置的互信息不会超过它自己的熵，所以先选熵低的。提前停止要求两个条件同时成立：平均熵低于 0.005 说明各位置的分布已经集中，最高概率词元连续两步不变说明画布不再被改写。只满足前一条时，可能有位置刚被改过，其余位置所依据的条件还在变。

#### 10.2.4 与 10.2.7 节：GQA 与 MLA 的配置键

- Hugging Face 的模型配置里，Q 头数与 KV 头数是 `num_attention_heads` 与 `num_key_value_heads` 两个字段；计算时把每组 K、V 沿头维复制或广播给组内各查询头。
- DeepSeek-V3 的 MLA 配置：`hidden_size = 7168`、`num_attention_heads = 128`、`kv_lora_rank = 512`（缓存的隐向量宽度）、`qk_nope_head_dim = 128`、`qk_rope_head_dim = 64`（RoPE 旁路宽度）、`v_head_dim = 128`；查询侧另有一个 `q_lora_rank = 1536` 的低维隐向量 $$c^{Q}_t$$，$$q^{R}_t = \operatorname{RoPE}(c^{Q}_t W^{QR})$$。
- vLLM 的 MLA 后端在源码注释里把两种计算形态称为 compute friendly（用于 prefill，把 K、V 展开成完整形状）与 data-movement friendly（用于 decode，吸收投影后直接读 576 维的缓存）。

#### 10.2.6 与 10.3.7 节：分页 KV 的张量形状

- **物理块池**：每层一个大张量，形如 `[num_blocks, block_size, n_kv, d_h]`，K、V 各一份。块按需分配，物理上不必相邻。
- **块表**：每个请求一张，形如 `[max_num_blocks_per_seq]` 的整数数组，第 k 项是逻辑第 k 块对应的物理块号。位置 i 的 K/V 在物理块 `block_table[i // block_size]` 的第 `i % block_size` 个槽里。
- `flash_attn` 包的 `flash_attn_with_kvcache` 接受形如 `[batch_size, max_num_blocks_per_seq]` 的 `block_table`，此时 `k_cache` 的形状为 `[num_blocks, page_block_size, nheads_k, headdim]`；同一个调用负责把新词元的 K、V 写入对应槽位。

#### 10.3.3 至 10.3.6 节：FlashAttention 2 到 4 的硬件映射

GPU 的线程层级与存储层级（取自 FlashAttention-3 论文对 H100 的描述）：

| 层级 | 并行单位 | 对应的存储 | 容量与带宽（H100） |
| --- | --- | --- | --- |
| 芯片 | grid | GMEM，即 HBM | 80 GiB，3.35 TB/s |
| GPC | threadblock cluster | L2 缓存 | 50 MiB，12 TB/s |
| SM | threadblock，即 CTA | SMEM，片上共享内存 | 每 SM 228 KiB，全卡 31 TB/s |
| 线程 | thread | RMEM，寄存器 | 每线程至多 256 个寄存器 |

GPU 的线程层级与存储层级。32 个线程组成一个 warp，4 个相邻 warp 组成一个 warpgroup；同一 CTA 内的线程可直接寻址 SMEM。

- **FA3（Hopper）。** 张量内存加速器（TMA）在 GMEM 与 SMEM 之间异步搬运；Tensor Core 经 warpgroup 级的 `WGMMA` 指令异步调用，可直接从共享内存取输入。warp 特化把一个 CTA 内的 warp 分成生产者（只经 TMA 把下一块 K、V 装入 SMEM 的环形缓冲）与消费者（只做矩阵乘），并用 `setmaxnreg` 把寄存器多分给做矩阵乘的 warpgroup。FP8 路径用块级量化（每小块 Q、K、V 独立定标）与非相干处理（量化前乘随机 ±1 对角阵与 Hadamard 矩阵之积）。
- **FA4（Blackwell）。** 每个 SM 新增 256 KB 的 Tensor Memory（TMEM）存放 Tensor Core 的中间结果，矩阵乘分块从 64 × 128 增大到 128 × 128，Tensor Core 操作完全异步并可直接写 TMEM；反向传播配合 2-CTA MMA 模式减少共享内存流量与原子加。指数运算的硬件指令是 `MUFU.EX2`；软件模拟把 $$2^{\lfloor x \rfloor}$$ 交给整数 ALU 改写指数位，小数部分用多项式逼近走 FMA，每行只对 10–25% 的元素这样做，比例按矩阵乘与指数运算的吞吐比经验调定。FA3 的设计搬到 Blackwell 上，共享内存流量和指数运算的耗时超过矩阵乘 25–60%。
- **指令与架构编号。** `cp.async` 要求 sm_80 及以上；`tcgen05.mma` 面向 sm_100a 与 sm_110a（PTX ISA 9.0 起 sm_101a 更名为 sm_110a）。

#### 10.3.8 节：注意力内核的框架入口

- PyTorch 的 `torch.nn.functional.scaled_dot_product_attention` 在 FlashAttention-2、Memory-Efficient Attention 与 C++ 数学实现三种后端之间自动选择，可用上下文管理器 `torch.nn.attention.sdpa_kernel` 限定后端；融合后端不满足输入约束时给出警告并退回数学实现。
- Hugging Face Transformers 在 `from_pretrained` 时用 `attn_implementation` 参数选择实现，如 `"sdpa"` 或 `"flash_attention_2"`。
- `flash_attn` 包提供 `flash_attn_func`、`flash_attn_varlen_func` 与 `flash_attn_with_kvcache`。变长接口把一批序列拼成 `[total_tokens, n_h, d_h]`，另传累计长度数组 `cu_seqlens_q` 与 `cu_seqlens_k`，形状 `[batch_size + 1]`；三条长度 3、5、2 的序列对应 `[0, 3, 8, 10]`。

#### 10.4.7 节：量化权重的存储与框架入口

- 4 位权重打包存放，两个拼成一个字节，或如 AutoGPTQ 的 `qweight` 那样八个拼成一个 32 位整数；缩放因子与零点另存为小张量。Llama 3 8B 的 `down_proj[14336 → 4096]` 用 `g = 128` 时，每个输出通道的 14,336 个输入权重分成 112 组，缩放因子形状为 `[4096, 112]`，FP16 下 0.92 MB；打包后的权重 29.4 MB，BF16 原始大小是 117.4 MB。
- Hugging Face Transformers 通过 `from_pretrained` 的 `quantization_config` 参数加载或生成量化模型，如 `GPTQConfig(bits=4, ...)`。
- vLLM 用 `--quantization` 指定权重量化方法，用 `--kv-cache-dtype` 指定 KV 缓存的存储类型，取值 `auto` 时沿用模型的数据类型，`fp8` 等价于 `fp8_e4m3`。
- GPTQ 的实现以 128 列为一批更新，并用 Cholesky 分解保证数值稳定。

#### 10.6.3 节：投机解码的框架入口

vLLM、TensorRT-LLM 与 SGLang 都支持投机解码。以 vLLM 为例，通过 `--speculative-config` 传入一个 JSON：`method` 选草稿来源，文档列出的常用取值有 `draft_model`、`ngram`、`suffix`、`mtp`、`eagle3`、`dflash`；`model` 指定草稿模型；`num_speculative_tokens` 即 $$\gamma$$。`ngram` 方法另有 `prompt_lookup_min` 与 `prompt_lookup_max` 控制匹配窗口。

#### 11.1.3 节：TensorRT-LLM 的部署参考

TensorRT-LLM 现行版本的 PyTorch 后端直接加载 checkpoint，不需要预先编译 engine；TensorRT engine 后端已[移除](https://nvidia.github.io/TensorRT-LLM/latest/legacy/tensorrt-backend-removal.html)。旧资料中“原始权重 → TensorRT-LLM checkpoint → 构建 engine → 运行”的[历史构建流程](https://nvidia.github.io/TensorRT-LLM/latest/legacy/architecture/workflow.html)只适用于旧版；engine 是推理执行产物，不是可续训的检查点。维护旧服务时须使用相同版本的示例与运行库，不能把旧版转换脚本拼接到新入口。

- **输入。** 受支持模型的完整权重、模型配置与分词器；量化模型还要有匹配的量化配置和缩放因子。[快速入门](https://nvidia.github.io/TensorRT-LLM/latest/quick-start-guide.html)中 `LLM` 可加载本地 Hugging Face 模型目录或受支持的量化 checkpoint，在线服务入口是 `trtllm-serve`。训练框架应先导出推理权重（A.7.3）。
- **启动与请求路径。** 启动时加载权重，确定精度与并行布局，配置 KV 容量、最大批量和词元预算（参数见[服务入口](https://nvidia.github.io/TensorRT-LLM/latest/commands/trtllm-serve/trtllm-serve.html)）。运行时每个 rank 一个 `PyExecutor` 工作进程：`Scheduler` 选本轮工作，`KVCacheManager` 准备缓存块，`ModelEngine` 执行前向，`Sampler` 从 logits 选词元（[架构说明](https://nvidia.github.io/TensorRT-LLM/latest/developer-guide/overview.html)）。与 11.1 节图 11-1 的循环同构。
- **优化与验收。** 先用受支持的高精度配置核对分词、聊天模板、停止条件和保留集质量，再逐项加入量化、多卡或前缀复用。量化的权重格式、缩放方式与可用内核必须匹配（[量化说明](https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html)）；能否同时启用多项优化查[功能组合矩阵](https://nvidia.github.io/TensorRT-LLM/latest/features/feature-combination-matrix.html)。单卡已能容纳权重时，先测单卡基线，再比较两卡 TP 与两份单卡副本。
- **排障。** 加载失败先查架构、权重格式与软硬件支持；启动 OOM 查权重、工作区和 CUDA Graph 的预算；长请求或高并发才出现的容量问题再查 KV 块、准入与抢占；输出异常先对齐输入词元、模板和采样参数，再查量化误差。性能验收按 11.13 节重放相同负载，报告质量、错误率与 TTFT/TPOT。

#### 11.1.3 节：引擎实现细节

- vLLM：PyTorch 模型默认经 `torch.compile` 编译，加分段与整图的 CUDA/HIP Graph，`--enforce-eager` 才退回即时执行（11.10.5）。
- llama.cpp 服务端的 `cache_prompt` 默认开启：新请求与槽位里上一次的 Prompt 比较，只重算不同的后缀。
- DeepSpeed-FastGen 的[论文](https://arxiv.org/abs/2401.08671)报告有效吞吐最高 2.3 倍、平均延迟减半、词元级尾延迟最多降到 1/3.7，基线是尚无分块 Prefill 的早期 vLLM。

#### 11.3.8 节：vLLM 调度器的机制名、状态名与指标名

名称取自 v0.29.0 的 `vllm/config/scheduler.py`、`vllm/v1/core/sched/scheduler.py` 与 `vllm/v1/metrics/loggers.py`，随版本变动。词元预算、并发上限、分块开关、单请求块上限、出队策略的启动参数见 11.10 节。

| 11.3 节的机制 | vLLM 里的名字 | 说明 |
| --- | --- | --- |
| 水位线 | `watermark` | 准入时保持空闲的 KV 块比例 |
| 整条 Prompt 放得下才准入 | `scheduler_reserve_full_isl` | 防止分块 Prefill 下的过度准入 |
| 在途上限 | `max_num_queued_reqs`、`max_num_queued_tokens` | 前者限等待加运行的请求总数，后者限处于 Prefill 阶段的 Prompt 词元总量；超限的新请求返回 HTTP 503 |
| 状态机 | `RequestStatus.WAITING`、`RUNNING`、`PREEMPTED`、`FINISHED_*` | 被抢占时 `num_computed_tokens` 置 0 并放回等待队列，FCFS 下插在队首 |
| 调度记账 | `num_computed_tokens`、`num_tokens_with_spec` | 源码注释说明调度器没有 Prefill 与 Decode 两个阶段，只有前者追赶后者 |
| 观测 | `vllm:num_preemptions`、`vllm:num_requests_waiting`、`vllm:kv_cache_usage_perc` | 对应 11.3.5 的抢占计数、11.3.6 的排队区与容量墙 |

vLLM 调度器的机制、状态与指标名对照。

#### 11.4.1、11.4.3 节：KV 池的参数与张量布局

- 允许引擎占用的显存比例 `u` 在 vLLM 里是 `gpu_memory_utilization`。vLLM 启动时用最大批次的假输入跑一遍前向实测激活峰值，再把剩余显存划给 KV 池。
- 引擎在每步前向之前为本步的每个词元算好槽位，排成一维数组（vLLM 里叫 `slot_mapping`）交给内核。
- 张量的维度次序因后端而异。vLLM 的 FlashAttention 后端一层的缓存张量是 `[块数, n_kv, 块大小, 2 × d_h]`，K 与 V 在最后一维并排；Llama 3 8B、53.94 GB 的池为 `[25720, 8, 16, 256]`，每层 1.69 GB，32 层合 53.94 GB。

#### 11.5.2、11.5.6、11.5.7 节：前缀缓存的实现细节

- vLLM 的[自动前缀缓存](https://docs.vllm.ai/en/latest/design/prefix_caching/)默认哈希函数为 SHA-256，文档提醒换用更快的非密码学哈希会抬高碰撞风险；可命中长度上限为 `T − 1` 再向下对齐到块边界；请求可携带 `cache_salt`，盐进入第 0 块的键；要求返回 Prompt 对数概率的请求跳过缓存查找；适配器名与多模态内容哈希进入块的额外键。共享前缀专用的 cascade attention 内核默认关闭。
- SGLang 的前缀树以命名空间键隔离不同适配器的分支。调度策略 `--schedule-policy` 默认 `fcfs`，`lpm` 为最长前缀优先（11.11.4）。
- SGLang 网关的 `cache_aware` 策略分四步：为每个实例维护一棵只按自己转发过的请求更新的近似前缀树（存原始字符）；新请求求与各树的匹配比例，超过阈值送往该实例，否则送往树最小的实例；待处理请求数的最大值与最小值之差超过绝对阈值、且最大值超过最小值的给定倍数时，改按最短队列分发；后台按 LRU 淘汰近似树的叶子。各阈值的默认值随版本变动。
- vLLM Router（2025 年 12 月发布）提供 `consistent_hash` 策略，按会话或用户 ID 等路由键做一致性哈希；另有 Power of Two、轮询与随机（[发布说明](https://vllm.ai/blog/vllm-router-release)）。

#### 11.6 节：vLLM 多 LoRA 的实现与配置

名称取自 vLLM v0.29.0。

- 分段索引：[`LoRAKernelMeta`](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/lora/ops/triton_ops/lora_kernel_metadata.py) 保存 11.6 节图 11-13 底部的索引。`token_lora_mapping` 是每个词元的适配器号，`num_tokens_per_lora` 是各段词元数，`lora_token_start_loc` 是它带前导 0 的前缀和，即各段边界；vLLM 不搬动词元，而用 `token_indices_sorted_by_lora_ids` 记下排序后的行号，内核据此间接取行。降维与升维对应同目录下的 `lora_shrink_op.py` 与 `lora_expand_op.py`，由 `vllm/lora/punica_wrapper/` 封装后供各线性层调用。
- 静态槽位：[`create_lora_weights`](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/lora/layers/base_linear.py) 为每层分配两组全零张量，打包层（`qkv_proj`、`gate_up_proj`）每个切片一份。`lora_a_stacked` 的形状是 `[max_loras, 1, max_lora_rank, d_in]`，`lora_b_stacked` 是 `[max_loras, 1, d_out, max_lora_rank]`，`set_lora` 把适配器的 `A`、`B` 拷进某个槽的左上角；vLLM 按“输出 × 输入”存放，行列与本书相反。降维内核从堆叠张量的形状读出秩，所以每个适配器都按 `max_lora_rank` 计算。
- 调度约束：[调度器](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/core/sched/scheduler.py)在接纳等待请求时，批里已有 `max_loras` 个不同的适配器、而新请求用的是另一个时，跳过它。
- S-LoRA 的两个内核：Prefill 用 MBGMM（Multi-size Batched Gather Matrix-Matrix Multiplication），Decode 用 MBGMV（矩阵乘向量），Multi-size 指同一批内秩可以不同，Gather 指按页表逐页取数。

| 机制 | 决定它的量 | vLLM v0.29.0 的配置项 |
| --- | --- | --- |
| 同一批里最多几个不同的适配器 | 显存槽位数 | `max_loras`，默认 1 |
| 槽位按多大的秩预留 | 填充开销（11.6.6） | `max_lora_rank`，默认 16；[文档](https://github.com/vllm-project/vllm/blob/v0.29.0/docs/features/lora.md)建议设为实际用到的最大秩 |
| 主机内存里保留几个适配器 | 换入命中率 | `max_cpu_loras`，缺省等于 `max_loras`，不得小于它 |
| 哪些线性层挂增量一路 | 内核启动次数与槽位显存 | `target_modules`（命令行 `--lora-target-modules`，如只保留 `o_proj` 与 `qkv_proj`） |
| 张量并行下适配器怎么切 | 增量一路的通信量 | `fully_sharded_loras`，默认关闭，此时只对增量一路的一半计算做切分 |

vLLM [`LoRAConfig`](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/config/lora.py) 的配置项与 11.6 节各机制的对应；启用多 LoRA 的总开关是 `--enable-lora`。

#### 11.7 节：约束解码库的接口与引擎接线

- XGrammar 的 Python 接口：`GrammarCompiler` 负责编译并自带缓存（`compile_json_schema`、`compile_regex`、`compile_grammar`）；每个请求一个 `GrammarMatcher`，`fill_next_token_bitmask` 填位图（11.7.1），`accept_token` 做采样后的状态转移，`find_jump_forward_string` 对应跳跃前进（11.7.5），`rollback` 用于投机解码的回退（11.7.6；v0.1.21 改用 Earley 解析器后不再限制回退步数）。批量填位图用 `BatchGrammarMatcher`；`apply_token_bitmask_inplace` 以 `indices` 参数只处理指定的行；树状草稿用 `traverse_draft_tree`。论文版本用持久化栈支持常数时间回退，按节点自适应地存合法列表、不合法列表或整张位图，使 JSON 文法的缓存从 160 MB 降到 0.46 MB。XGrammar [v0.1.21](https://github.com/mlc-ai/xgrammar/releases/tag/v0.1.21)（2025 年 7 月）以 Earley 解析器替换下推自动机。
- outlines-core：JSON Schema 经 `build_regex_from_schema` 转成正则，自引用按 `max_recursion_depth`（默认 3）展开；预计算的 `Index` 只为合法词元存落点。
- vLLM：约束放在 `SamplingParams.structured_outputs`；后端默认 `auto`，先校验能否用 XGrammar，失败（Schema 用到 XGrammar 不支持的特性）时退到 `guidance`（llguidance），逻辑在 `vllm/sampling_params.py`。
- SGLang：`--grammar-backend` 可取 `xgrammar`、`outlines`、`llguidance`、`none`，不指定时用 `xgrammar`（[文档](https://docs.sglang.io/advanced_features/structured_outputs.html)）。
- llama.cpp：[文法文档](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md)写明 JSON Schema 中不支持的特性会被静默跳过，Schema 只约束输出、不会注入 Prompt。

#### 11.8.2 节：张量并行的线性层在代码里叫什么

Megatron-LM 把两种切法写成 `ColumnParallelLinear` 和 `RowParallelLinear` 两个模块。vLLM 的 Llama 实现沿用同一思路：

- 列切：`qkv_proj` 是 `QKVParallelLinear`，`gate_up_proj` 是 `MergedColumnParallelLinear`。
- 行切：`o_proj` 和 `down_proj` 是 `RowParallelLinear`，all-reduce 在它的末尾。
- 词表维切：嵌入是 `VocabParallelEmbedding`，输出是 `ParallelLMHead`。

读到这几个类名，就能判断一个矩阵是列切还是行切、all-reduce 发生在哪里。

#### 11.8.8 节：vLLM 的并行参数

| 参数 | 含义 |
| --- | --- |
| `--tensor-parallel-size` | 张量并行度 |
| `--pipeline-parallel-size` | 流水线并行度 |
| `--data-parallel-size` | 数据并行度，稠密模型下即副本数 |
| `--enable-expert-parallel` | 打开后专家层用专家并行，并行度等于 `TP × DP`；不打开时专家层在同一个 `DP × TP` 组内按张量并行切 |

vLLM 的并行参数与 11.8 节四种多卡方式的对应（v0.29.0）。worker 进程数为 `TP × PP`。

#### 11.9.7 节：分离式部署的三个实现

- **vLLM**：两个 vLLM 实例加一个连接器（connector）。启动参数 `--kv-transfer-config` 里，`kv_connector` 选连接器，`kv_role` 取 `kv_producer`（Prefill 实例）或 `kv_consumer`（Decode 实例）。连接器分两半：与调度器同进程的一半决定传哪些块，各 worker 进程里的一半执行传输，并与注意力层配合实现逐层存取。请求级的交接信息放在 `kv_transfer_params` 里，如 `do_remote_decode`、`do_remote_prefill`。传输失败时的处置以 NIXL 连接器为例由 `kv_load_failure_policy` 控制，默认 `fail`（报错），取 `recompute` 才在 Decode 侧本地重算。文档标明该特性仍为实验性质；连接器兼容性文档把两侧张量并行度不同的配置记作 Hetero TP。
- **NVIDIA Dynamo**：路由由 `PrefillRouter` 编排，KV 经 NIXL 在两侧显存之间直传。各后端的交接元数据不同：vLLM 传块 ID 与远端连接信息，SGLang 传用于 RDMA 握手的 `bootstrap_info`。
- **Mooncake**：全局调度器 Conductor、每节点一个负责传输的 Messenger 进程，以及由 CPU 内存和 SSD 组成的 KVCache 池。

连接器清单、参数名与默认阈值变化很快，以各项目当前文档为准。

#### 11.10 节：vLLM v0.29.0 的模块与参数

以下取自 vLLM v0.29.0（提交 `98dff2a`）的 V1 引擎，即 `vllm/v1/` 目录。V1 是引擎架构的代号；worker 里执行前向的 model runner 另有先后两套实现，编号与引擎的 V1 无关。

**一个请求的八步与对应的类。**

1. 输入处理：`Renderer`（`vllm/renderers/`，方法 `render_chat`）套模板并分词；`InputProcessor` 校验采样参数、LoRA 与长度，产出 `EngineCoreRequest`。
2. 跨进程提交：`AsyncLLM` 为请求建输出队列；`EngineCoreClient` 用 msgpack 序列化，经 ZMQ 套接字发给引擎核心。过线的是词元 ID、采样参数、可选的 LoRA 与缓存盐，不含文本。
3. 调度：引擎核心的输入线程把请求放进 `input_queue`，主线程的忙循环 `run_busy_loop` 交给 `Scheduler.add_request`。循环体是 `schedule()`、`execute_model()`、`update_from_output()`；开启异步调度时是 `step_with_batch_queue()`，否则是 `step()`。排批时向 `KVCacheManager` 申请块，产出 `SchedulerOutput`。
4. 下发：多卡时 `MultiprocExecutor` 经共享内存消息队列一次广播给全部 worker；单卡时 `UniProcExecutor` 把 worker 建在引擎核心进程内。
5. 前向与采样：`GPUModelRunner`。
6. 回传：只有最后一个流水级的 TP rank 0 把 `ModelRunnerOutput` 传回。
7. 更新：`update_from_output()` 记新词元、检查停止条件、释放已结束请求的块；新词元 ID 经输出线程和 ZMQ 回到 API 服务进程。
8. 反分词与流式输出：`OutputProcessor`。

**进程数。** [架构文档](https://github.com/vllm-project/vllm/blob/v0.29.0/docs/design/arch_overview.md)给出的进程数是“API 服务数 + 引擎核心数 + GPU 数”：`-tp 4` 的单机部署为 6 个；数据并行时每个 DP rank 一个引擎核心，另加一个协调进程，`-tp 2 -dp 4` 为 17 个。物理核数少于进程数时吞吐和延迟明显变差，见[优化文档的 CPU 资源一节](https://github.com/vllm-project/vllm/blob/v0.29.0/docs/configuration/optimization.md#cpu-resources-for-gpu-deployments)。

**启动时的显存实测。** 顺序是：加载权重；`profile_run` 用 `max_num_batched_tokens` 个假词元跑一次前向记录峰值；`gpu_worker.py` 的 `determine_available_memory` 按 `M_KV = u × M_total − M_weights − M_peak − M_graph` 算出 KV 字节数（`u` 即 `gpu_memory_utilization`，默认 0.92）；`kv_cache_utils.py` 取整为块数；一次性分配；捕获 CUDA Graph。启动日志打印 `GPU KV cache size: N tokens` 与 `Maximum concurrency for <max_model_len> tokens per request: Kx`，后者等于块数除以单请求满长所需块数。`--kv-cache-memory-bytes` 可跳过实测直接指定 KV 字节数。

**每步词元预算与请求数上限的默认分档**（在线服务入口）：

| 显存 | `max_num_batched_tokens` | `max_num_seqs` |
| --- | ---: | ---: |
| 不低于 160 GiB | 16,384 | 1,024 |
| 不低于 70 GiB 且不是 A100 | 8,192 | 1,024 |
| 其余 | 2,048 | 256 |

`scheduler_reserve_full_isl` 默认开启：新请求准入时要求整条输入放得下。`long_prefill_token_threshold` 限制单个请求每步最多占多少预算。

**请求状态 `RequestStatus`。** 排队状态有四种：`WAITING`（普通排队）、`WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`（等语法编译）、`WAITING_FOR_REMOTE_KVS`（等分离式部署的远端 KV）、`WAITING_FOR_STREAMING_REQ`（等流式输入的下一段）；后三种被调度器临时跳过，不挡后面的普通请求。在批里的是 `RUNNING`，被抢占的是 `PREEMPTED`，其后的取值表示结束并记录原因。

**调度器内部。** 运行队列 `running`、等待队列 `waiting`；要块调用 `KVCacheManager.allocate_slots`，返回 `None` 即抢占后重试；接纳时另查 `max_loras`。发给 worker 的批描述里，新请求带全量 `NewRequestData`，老请求只带增量 `CachedRequestData`。异步调度（`async_scheduling`）给每个 Decode 请求记一个输出占位 `num_output_placeholders`。约束解码的掩码由 `get_grammar_bitmask` 在 CPU 上计算，随 `sample_tokens` 送到 worker。

**块管理的数据结构**（`vllm/v1/core/block_pool.py` 与 `kv_cache_utils.py`）：

- `KVCacheBlock`：块号 `block_id`、引用计数 `ref_cnt`、块哈希，以及前后两个空闲链表指针。
- `FreeKVCacheBlockQueue`：空闲块的双向链表，指针长在块对象上，从中间摘除是 O(1)。
- `cached_block_hash_to_block`：块哈希到块的映射。
- `null_block`：0 号块，启动时取走只作占位，可分配的是 `N_blocks − 1` 块。

哈希键由 `hash_block_tokens` 计算，输入是“父块哈希、本块词元、附加键”；该版本把 prompt embeds 的哈希放进附加键，另提供更快但不抗碰撞的 `xxhash`。请求 prompt logprobs 时跳过缓存读取。

**worker 侧。** `BlockTables`（`vllm/v1/worker/gpu/block_table.py`）持有 `block_tables[max_num_reqs, 每请求最大块数]`（int32，每个 KV 组一张）与 `slot_mappings[KV 组数, max_num_batched_tokens]`（int64）；槽位由 Triton 内核 `_compute_slot_mappings_kernel` 算出。拼批的 `prepare_inputs` 产出 `input_ids`、`positions`、`slot_mapping` 与 `query_start_loc`；LM head 只取 `logits_indices` 指定的行。默认的 model runner 是第二套（`vllm/v1/worker/gpu/model_runner.py`，设计文档 `docs/design/model_runner_v2.md`）；第一套 `gpu_model_runner.py` 仍在树里，供尚不支持的特性回退。

**CUDA Graph。** 默认尺寸表是 `[1, 2, 4]`，加 8 到 248 每隔 8 一个，再加 256 起每隔 16 一个，上限 `min(2 × max_num_seqs, 512)`（数据中心级 Blackwell 上为 1,024）；`max_num_seqs = 1,024` 时共 51 个尺寸。`cudagraph_mode` 默认 `FULL_AND_PIECEWISE`：纯 Decode 的批回放整图，混合批回放分段图、注意力留在图外（见 [CUDA Graphs 设计文档](https://github.com/vllm-project/vllm/blob/v0.29.0/docs/design/cuda_graphs.md)）。`--enforce-eager` 关掉全部图。

**主要启动参数。**

| 参数 | v0.29.0 默认值 | 调的是什么 |
| --- | --- | --- |
| `--gpu-memory-utilization` | 0.92 | 本实例可用显存占总量的比例，权重也算在内 |
| `--kv-cache-dtype` | `auto`（随模型精度） | KV 每个数的字节数；`fp8` 使块数翻倍 |
| `--block-size` | 16 | 每块词元数 |
| `--max-num-batched-tokens` | 按显存分档 | 每步词元预算 |
| `--max-num-seqs` | 按显存分档 | 每步请求数上限，也决定 CUDA Graph 尺寸上限 |
| `--enable-chunked-prefill` | 模型支持即开启 | 关闭后放不进剩余预算的 Prompt 整条等下一步 |
| `--long-prefill-token-threshold` | 0（不限） | 单个请求每步最多占用的预算 |
| `--scheduling-policy` | `fcfs` | 排队顺序与抢占对象；`priority` 按请求优先级 |
| `--enable-prefix-caching` | 模型支持即开启 | 块哈希表是否启用 |
| `--prefix-caching-hash-algo` | `sha256` | 哈希的速度与抗碰撞性 |
| `--async-scheduling` | 兼容即开启 | 调度与前向重叠 |
| `--compilation-config` 的 `cudagraph_mode` | `FULL_AND_PIECEWISE` | 哪些批回放整图或分段图 |
| `--enforce-eager` | 关 | 关掉全部 CUDA Graph |
| `--tensor-parallel-size`、`--pipeline-parallel-size`、`--enable-expert-parallel` | 1、1、关 | 三种并行；worker 数为 `TP × PP` |
| `--enable-lora`、`--max-loras`、`--max-lora-rank` | 关、1、16 | 同一批内的适配器数与秩上限 |
| `--structured-outputs-config` | 后端 `auto` | 语法掩码由哪个后端计算 |
| `--kv-transfer-config` | 无 | KV connector：分离式部署与 KV 卸载共用的通路 |
| `--kv-offloading-size` | 无 | 经 KV connector 给前缀缓存加一层主机内存 |
| `--speculative-config` | 无 | 草稿来源与每步草稿数 |

vLLM v0.29.0 的主要启动参数与默认值，取自 `vllm/config/` 下各配置类与 `vllm/engine/arg_utils.py`。

#### 11.11 节：SGLang v0.5.20 的模块与参数

以下取自 [SGLang v0.5.20](https://github.com/sgl-project/sglang/tree/v0.5.20) 的源码与文档；后端运行时在 `python/sglang/srt`。

**前端提示的调用链。** `StreamExecutor.fork` 在分支数大于 1 时先提交一条 `SglCommitLazy`，`RuntimeEndpoint` 把它翻译成一个 `max_new_tokens = 0` 的 `/generate` 请求。

**后端的三个组件**（彼此用 ZMQ 通信，见 `entrypoints/engine.py`）：

| 组件 | 所在进程 | 职责 |
| --- | --- | --- |
| `TokenizerManager` | 主进程，与 HTTP 服务同在 | 分词，把词元 ID 发给调度器；收到文本后返回客户端 |
| `Scheduler` | 子进程 | 收请求、组批、执行前向、处理结果；持有前缀树、显存池、语法管理器 |
| `DetokenizerManager` | 子进程 | 把输出词元 ID 增量还原成文本 |

SGLang 后端的三个组件。`Scheduler` 主循环的四步依次是 `ingest_requests`、`get_next_batch_to_run`、`run_batch`、`process_batch_result`；默认主循环是 `event_loop_overlap`，下一批的输入词元经 `FutureMap` 留在 GPU 上。

**三张表。** 源码里的 `RadixAttention` 类（`layers/radix_attention.py`）只是模型中的注意力层，把计算转交给 FlashInfer、FlashAttention、Triton 等后端；前缀复用逻辑在 `mem_cache/`。KV 池 `MHATokenToKVPool` 每层一个 `k_buffer[N + page_size, n_kv, d_h]` 与同形的 `v_buffer`，多出的一页是占位页；`req_to_token`（由 `ReqToTokenPool` 持有，int32，`[请求数 + 1, 最大上下文]`）是块大小为 1 的块表；树节点 `TreeNode` 存 `key`（词元段）、`value`（int64 槽位下标）、`children`、`lock_ref`、`last_access_time`，备份到主机内存的一份记在 `host_value`。树的操作对应 `match_prefix`、`_split_node`、`insert`、`inc_lock_ref`/`dec_lock_ref`、`cache_unfinished_req`、`cache_finished_req`、`evict`；“可淘汰”“受保护”的计数是 `evictable_size_` 与 `protected_size_`。

v0.5.20 的默认工厂返回 `UnifiedRadixCache`，把全注意力、滑动窗口和 Mamba 状态三类缓存挂在同一棵树上；`radix_cache.py` 的 `RadixCache` 只处理全注意力，不到一千行，适合入门阅读。

**页大小。** 启用 FlashMLA 时 `page_size` 被强制为 64，TensorRT-LLM 的 MLA 后端只接受 32 或 64。页大于 1 时 `RadixKey.page_aligned` 把键截到页的整数倍。

**淘汰策略与键。** `--radix-eviction-policy` 可取 `lru`（默认）、`lfu`、`slru`、`priority`，差别只在堆的排序键：`lru` 用 `last_access_time`，`lfu` 用 `(hit_count, last_access_time)`，`slru` 把命中次数达到阈值（默认 2）的节点归入保护段。键的隔离靠 `RadixKey.extra_key`，带 LoRA 的请求把适配器 ID 拼进它；多租户另有 `cache_salt`。分层缓存（HiCache）由 `--enable-hierarchical-cache` 打开，外部存储另需 `--hicache-storage-backend`。

**排序策略。** `--schedule-policy` 默认 `fcfs`；感知缓存的有 `lpm`、`dfs-weight`、`hrrn`，不感知的有 `fcfs`、`lof`、`random`、`routing-key`。选 `lpm` 时等待队列超过 128 个请求当轮退回 `fcfs`；批内前缀检查的阈值是 32 个词元，只在 `lpm` 与 `hrrn` 下执行。

**准入与撤回的常数。** 准入由 `PrefillAdder` 记账（`--max-prefill-tokens` 默认 16,384，另有 `--chunked-prefill-size`）。预留系数 `ρ` 即 `new_token_ratio`，初值 `0.7 × schedule_conservativeness`，上限 1，每个无撤回的 Decode 步线性下降，600 步降到初值的 0.14 倍；`max_new_tokens` 在余额式里截断到 4,096，新请求另加一页。`check_decode_mem` 发现槽位不够时 `retract_decode` 撤回请求，默认 `--retraction-policy length` 先撤已生成词元最少的；撤回后 `ρ` 重估为 `(已生成词元总数 + 20 × 请求数) ÷ (max_new_tokens 总数 + 1)`，直接覆盖旧值。

**结构化输出。** `--grammar-backend` 可取 `xgrammar`（默认）、`outlines`、`llguidance`、`none`。`GrammarManager` 以 `(类型, 语法字符串)` 为键查编译缓存，未命中时交线程池编译，请求进 `grammar_queue`；掩码由 `SamplingBatchInfo.update_regex_vocab_mask` 分配，采样后 `grammar.accept_token` 推进状态；与重叠调度配合时采样推迟到 `launch_batch_sample_if_needed`。各语法对象仍定义 `try_jump_forward` 与 `jump_and_retokenize`，但 `constrained/` 目录之外没有调用点。

**主要启动参数。**

| 启动参数 | 默认值 | 对应的机制 |
| --- | --- | --- |
| `--disable-radix-cache` | 关 | 关闭前缀树，请求结束即释放全部槽位 |
| `--page-size` | 1 | KV 池与树的粒度 |
| `--radix-eviction-policy` | `lru` | 淘汰堆的排序键 |
| `--enable-hierarchical-cache` | 关 | 在主机内存保留 K/V 备份 |
| `--hicache-storage-backend` | 无 | 第三级外部存储，跨实例共享 |
| `--schedule-policy` | `fcfs` | 等待队列排序 |
| `--schedule-conservativeness` | 1.0 | `ρ` 的初值系数 |
| `--chunked-prefill-size` | 按硬件自动确定 | 分块 Prefill 的块长；−1 为关闭 |
| `--enable-mixed-chunk` | 关 | 分块 Prefill 与 Decode 混排 |
| `--mem-fraction-static` | 按显存自动确定 | 权重与 KV 池占显存的比例 |
| `--disable-overlap-schedule` | 关 | 关闭重叠调度 |
| `--grammar-backend` | `xgrammar` | 约束解码后端 |

SGLang v0.5.20 的主要启动参数。多实例部署时，网关的 `cache_aware` 策略维护近似的字符级前缀树来选实例。

#### 11.13 节：vLLM 的压测参数与指标名

**压测。** `vllm bench serve` 用 `--request-rate` 指定速率，默认按泊松过程到达，`--burstiness` 调突发程度；`--request-rate` 取默认的 `inf` 时所有请求同时到达，测的是最大吞吐；`--max-concurrency` 是闭环的并发上限。

**`/metrics` 端点里值得长期保留的指标。**

- 状态量：`vllm:num_requests_running`、`vllm:num_requests_waiting`、`vllm:kv_cache_usage_perc`。
- 计数器：`vllm:num_preemptions`、`vllm:prefix_cache_queries` 与 `vllm:prefix_cache_hits`（两者之比是命中率）、`vllm:prompt_tokens`、`vllm:generation_tokens`。
- 直方图：`vllm:time_to_first_token_seconds`、`vllm:inter_token_latency_seconds`、`vllm:request_time_per_output_token_seconds`、`vllm:request_queue_time_seconds`、`vllm:e2e_request_latency_seconds`。
- 多模态：前端的处理器缓存由 `vllm:mm_cache_queries` 与 `vllm:mm_cache_hits` 统计；编码器输出缓存由调度器侧的 `EncoderCacheManager` 管理。

**护栏产品。** 规则层如 NeMo Guardrails（用 Colang 描述对话约束）；分类层如 Llama Guard（3-8B 面向文本，4-12B 面向多模态）。网关产品如 LiteLLM、Kong。

#### 12.1.3 与 12.1.7 节：BERT 的配置键与模块树

下表是 Hugging Face `BertConfig` 里决定形状的键，取值以 `google-bert/bert-base-uncased` 与对应 large 权重的 `config.json` 为准。

| 项目 | 配置键 | BERT-Base | BERT-Large |
| --- | --- | ---: | ---: |
| 层数 L | `num_hidden_layers` | 12 | 24 |
| 隐藏宽度 H | `hidden_size` | 768 | 1,024 |
| 注意力头数 A | `num_attention_heads` | 12 | 16 |
| 前馈中间层宽度 | `intermediate_size` | 3,072 | 4,096 |
| 词表 | `vocab_size` | 30,522 | 30,522 |
| 位置表行数 | `max_position_embeddings` | 512 | 512 |
| 段嵌入行数 | `type_vocab_size` | 2 | 2 |

BERT 两档规格对应的配置键。论文称词表为 30,000 个 WordPiece 词元，发布权重的 `vocab_size` 是 30,522。

`BertModel` 的模块树：

- 嵌入：`embeddings.{word,position,token_type}_embeddings` 与 `embeddings.LayerNorm`。
- 每层：`encoder.layer[i].attention.self.{query,key,value}`、`encoder.layer[i].attention.output.dense`、`encoder.layer[i].intermediate.dense`、`encoder.layer[i].output.dense`。
- 句级输出：`pooler.dense`。
- 预训练头：MLM 头是 `cls.predictions.transform.dense`、`cls.predictions.transform.LayerNorm` 与 `cls.predictions.decoder`（与词嵌入共享权重，另有一个 `[30522]` 的偏置）；NSP 头是 `cls.seq_relationship`。

#### 12.2.1 节：RoBERTa 与 BERT 的配置差异

`RobertaModel` 与 `BertModel` 的模块树同构，配置上只有三处不同：`vocab_size=50265`、`type_vocab_size=1`、`max_position_embeddings=514`。514 不是笔误：位置编号从 `padding_idx + 1` 起算，头两个编号被占掉。BERT 论文称自己的分词为 WordPiece，RoBERTa 论文把它描述为“字符级 BPE，词表 30K”，指的是同一张 30,522 行的表。

#### 12.2.2 节：ALBERT 的配置键

`AlbertModel` 的配置键是 `embedding_size=128`（即 E）、`num_hidden_groups=1`（共享组数）、`inner_group_num=1`；`E → H` 的那层投影叫 `embeddings.embedding_hidden_mapping_in`。`vocab_size` 是 30,000，与 BERT 的 30,522 不同。

#### 12.2.3 节：ELECTRA 的类名与发布配置

判别器是 `ElectraForPreTraining`（二分类头叫 `discriminator_predictions`），生成器是 `ElectraForMaskedLM`；判别器没有词表输出头，要当 MLM 用得另接后者。`google/electra-base-discriminator` 的 `hidden_size` 是 768，配套的 `google/electra-base-generator` 是 256，为 1/3，落在论文建议的 1/4 到 1/2 之间；两者的 `embedding_size` 都是 768，对应生成器与判别器共享词嵌入。`google/electra-small-discriminator` 把 `embedding_size` 设为 128、`hidden_size` 设为 256，用的是 ALBERT 的嵌入分解。

#### 12.3.2 至 12.3.4 节：Longformer 与 BigBird 的配置键、位置表与回退门槛

- **Longformer。** `LongformerConfig.attention_window` 是每层一个值的列表，发布权重里 12 层全是 512；前向另收一个 `global_attention_mask` 指定哪些位置走全局。全局注意力另用一套投影 `Q_g / K_g / V_g`，与窗口注意力的 `Q_s / K_s / V_s` 分开学。发布配置的 `max_position_embeddings` 是 4,098，比 4,096 多两行，沿用 RoBERTa 的位置偏移约定；4,096 行由原来的 512 行复制 8 次得到，每份内部相邻位置的嵌入与原表相同，只有两份拷贝相接的 7 处不连续。
- **Longformer 的三种实现。** 纯 PyTorch 的循环版省显存但很慢，只用于测试；分块版支持非空洞的情形，用于预训练与微调；另有一个用 TVM 生成的自定义 CUDA 核，用在语言建模实验上。论文在自回归语言建模实验里还用了逐层增大的窗口与少数头上的空洞（dilation），这两项没有用在预训练的编码器上。论文给出的感受野 `L × w` 是左右两侧合计的宽度近似，不是单向传播距离。
- **BigBird。** `BigBirdConfig.attention_type` 取 `"block_sparse"` 或 `"original_full"`，配合 `block_size` 与 `num_random_blocks`。`google/bigbird-roberta-base` 的取值是 `block_size = 64`、`num_random_blocks = 3`、`max_position_embeddings = 4096`、`vocab_size = 50358`；论文的 BigBird-itc 变体（把序列中已有的若干词元设为全局词元）取全局 `g = 2 × b`、窗口 `w = 3 × b`，`b` 即块大小。
- **回退门槛。** `transformers` 的实现要求序列长度至少为 `(5 + 2 × num_random_blocks) × block_size`，默认值下是 704；低于它就打印警告并把 `attention_type` 切回 `original_full`。

#### 12.3.5 节：ModernBERT 的配置键

| 部件 | ModernBERT-base 的取值 | 配置键 |
| --- | --- | --- |
| 位置编码 | RoPE，全局层 θ = 160,000、局部层 θ = 10,000 | `global_rope_theta`、`local_rope_theta` |
| 前馈 | GeGLU，中间层 1,152、GLU 上投影 2,304 | `intermediate_size` |
| 偏置项 | 除最后的解码线性层外全部关闭 | `attention_bias`、`mlp_bias`、`norm_bias` |
| 注意力 | 每 3 层一层全局，其余为 128 词元的局部滑动窗口 | `global_attn_every_n_layers`、`local_attention` |
| 层数 | 22 | `num_hidden_layers` |
| 词表 | 50,368（64 的倍数，含 83 个预留位） | `vocab_size` |
| 序列长度 | 8,192 | `max_position_embeddings` |
| 池化 | 默认平均 | `classifier_pooling` |

ModernBERT-base 的配置键，取值以 `answerdotai/ModernBERT-base` 的 `config.json` 为准。全局层用 FlashAttention 3、局部层用 FlashAttention 2，因为模型开发时 FlashAttention 3 还不支持滑动窗口。

#### 12.3.6 节：sentence-transformers 的池化配置

`sentence-transformers` 把池化做成独立模块，用 `1_Pooling/config.json` 配置。发布模型 `sentence-transformers/all-MiniLM-L6-v2` 里 `pooling_mode_mean_tokens` 为真、`pooling_mode_cls_token` 为假，后面再串一个 L2 归一化模块，使余弦相似度与内积等价。Sentence-BERT 在分类数据上把两个句向量拼成 `(u, v, |u − v|)` 接 softmax，在回归数据上直接优化余弦相似度。

#### 13.1.6 节：推理词元的接口字段与两种失效状态

以 [OpenAI 推理指南](https://developers.openai.com/api/docs/guides/reasoning)为准，字段名与档位随接口版本调整。

- **用量字段。** 用量对象里的 `output_tokens_details.reasoning_tokens` 给出一次请求用了多少推理词元；它们计入输出词元。
- **截断状态。** 生成的词元撞上上下文上限或 `max_output_tokens` 时，响应状态为 `incomplete`，此时可能一个可见词元都没有产出，输入与推理词元照样计费。指南建议起步时为推理和输出预留至少 25,000 个词元。
- **思考预算。** `reasoning.effort` 提供从 `none` 到 `max` 的若干档，调的是思考预算，不是策略类型；支持哪几档、默认哪一档随模型而定。

#### 13.2.2 节：Llama 3 8B 配置字段对应的模块与权重形状

Hugging Face `LlamaDecoderLayer` 里各字段落到的模块。权重按 `[输出维, 输入维]` 存放，与正文 `[行, 列]` 的数据流写法互为转置。

| 字段 | 取值 | 对应模块与权重形状 |
| --- | ---: | --- |
| `num_hidden_layers` | 32 | `model.layers.0` 到 `.31` |
| `hidden_size` | 4,096 | 每个位置的向量 `x[T, 4096]` |
| `num_attention_heads` | 32 | `self_attn.q_proj.weight[4096, 4096]` |
| `num_key_value_heads` | 8 | `self_attn.k_proj.weight[1024, 4096]`，`v_proj` 同形 |
| `intermediate_size` | 14,336 | `mlp.gate_proj`、`up_proj` 各 `[14336, 4096]`，`down_proj[4096, 14336]` |
| `vocab_size` | 128,256 | `model.embed_tokens.weight[128256, 4096]` |
| `tie_word_embeddings` | `false` | 另有 `lm_head.weight[128256, 4096]` |
| `rms_norm_eps` | 1e-05 | `input_layernorm`、`post_attention_layernorm` 各 `[4096]` |
| `rope_theta` | 500,000 | 不占参数，逐层作用在 q、k 上 |
| `max_position_embeddings` | 8,192 | Llama 3.1 起为 131,072 |

Llama 3 8B 的配置字段与模块名对照。Llama 1 与 Llama 2 的 7B 的 `intermediate_size` 是 11,008，即 $$\frac{8}{3} \times 4{,}096 = 10{,}922.67$$ 向上取到 256 的倍数。

#### 13.4.1 与 13.4.3 节：BART 的掩码词元与 T5 的两份缓存

- BART 论文里的掩码写作 `[MASK]`，fairseq 与 Hugging Face 实现中的词元是 `<mask>`。
- Hugging Face 的 `T5ForConditionalGeneration` 在生成时把编码器结果放在 `encoder_outputs` 里跨步复用，把解码器自注意力与交叉注意力两份 K/V 一起放在 `past_key_values` 里；`generate` 只在第一步计算 `encoder_outputs`。
- `google-t5/t5-base` 的检查点里有三张相对位置偏置表（每张 $$32 \times 12 = 384$$ 个数，32 个相对距离分桶乘 12 个头）：编码器 1 张、解码器 2 张。解码器的第二张挂在交叉注意力上，加载时作为多余的键忽略。

#### 14.2.1 节：MoE 前向六步在框架里的对应

- Hugging Face Transformers：DeepSeek-V3 的 MoE 层是 `DeepseekV3MoE`，第 2、3 步在其 `gate` 子模块（打分、sigmoid、top-k 与归一化），第 5 步在 `experts`。
- vLLM：`FusedMoE` 把第 4–6 步实现为“按专家排序 → 分组 → 分组 GEMM”三段内核。
- Megablocks 一类 dropless 实现改用变长分组 GEMM，把置换和变长一起交给内核，不设容量上限。

DeepSeek-V3 的相关配置键（`config.json`）：`hidden_size = 7168`、`intermediate_size = 18432`（前 3 层稠密 FFN）、`moe_intermediate_size = 2048`、`n_routed_experts = 256`、`n_shared_experts = 1`、`num_experts_per_tok = 8`、`num_hidden_layers = 61`、`first_k_dense_replace = 3`。

#### 14.2.5 节：执行侧的负载均衡（SGLang 的 Waterfill 与 LPLB）

出处：[Improving DeepEP MoE Load Balance in SGLang with Waterfill and LPLB](https://lmsys.org/blog/2026-06-26-waterfill-lplb)。

- **Waterfill** 处理共享专家。共享专家对每个词元都生效，固定绑在某张卡上就成了额外负担。Waterfill 把它变成可调度的槽位：路由专家选定之后，估计各专家并行卡的当前负载，再把共享专家的工作按空闲程度分配过去。
- **LPLB**（Linear-Programming-Based Load Balancer）处理冗余专家副本。它为每一层、每一批求解一个线性规划：在各物理副本之间切分该逻辑专家的词元，目标是最小化各卡负载的峰值；一条守恒约束保证它只重新分配已有流量，不制造也不丢弃词元。

两者都不改变路由器的 top-k 决策：Waterfill 只改共享专家由哪张卡执行，LPLB 只改词元落到同一逻辑专家的哪个副本上，输出不变，因此可以在推理阶段直接启用。在该文的实测配置下（两台 Hopper 节点共 16 卡，TP16-DP16-EP16），吞吐改善是个位数百分比；LPLB 需要确实配置了冗余专家，否则是纯额外开销。

#### 14.3.3 节：Mamba 官方实现的构造参数与内核

- `mamba_ssm` 库中，块结构的超参就是 `Mamba(d_model, d_state, d_conv, expand)` 四个构造参数，默认 `d_state = 16`、`d_conv = 4`、`expand = 2`；扫描内核是 `selective_scan_fn`。
- 论文里 $$f_\Delta$$ 先投到 1 维再广播到全部通道，官方实现把这 1 维推广为宽度 `dt_rank` 的低秩投影。
- 因果卷积与 SiLU 在 `mamba_simple.py` 中写在同一行：`x = self.act(self.conv1d(x)[..., :seqlen])`。

#### 14.3.7 节：Qwen3.5 线性层的状态形状

Transformers 中 `Qwen3_5MoeGatedDeltaNet` 的递推状态形状是 `[batch, linear_num_value_heads, linear_key_head_dim, linear_value_head_dim]`，即每个 value 头一块 `[128, 128]` 的矩阵；每层 64 个 value 头，每层 `64 × 128 × 128 = 1,048,576` 个数，45 层合计 47,185,920 个数（BF16 约 90 MiB），与序列长度无关。`linear_num_key_heads = 16` 小于 `linear_num_value_heads = 64`，实现里把 Query 与 Key 按 4 倍 `repeat_interleave` 广播上去，相当于把 GQA 的共享方式用到线性层，省的是投影参数，不是状态。

#### 14.4.4 节：占位符替换的实现位置

LLaVA 系列在嵌入层查表得到 `inputs_embeds` 之后，由宿主代码把 `<image>` 占位符所在的行替换成投影器输出的视觉特征；Qwen2-VL 用 `<|vision_start|>` 与 `<|vision_end|>` 两个界标包住视觉段。

#### 14.5.2 节：Qwen3 聊天模板的出处

14.5.2 的六步对应 Qwen3 `tokenizer_config.json` 里 `chat_template` 的 Jinja 分支。`<tool_call>` 与 `</tool_call>` 在 `added_tokens_decoder` 里是两个新增词元，`special` 字段为 `false`；该文件里标 `true` 的只有 `<|im_start|>`、`<|im_end|>` 一类控制词元。

#### 14.5.6 节：MCP 的消息字段与传输

- 工具发现：`tools/list` 返回的每个工具含 `name`、`description` 与 `inputSchema`。调用：`tools/call`，参数为 `name` 与 `arguments`；返回里的 `content` 是结果内容，`isError` 标记工具自身是否出错。消息格式是 JSON-RPC 2.0。
- 传输：标准传输有标准输入输出（stdio）与 Streamable HTTP 两种；更早的 HTTP+SSE 传输已被 Streamable HTTP 取代并标为弃用，规范保留一节讲向后兼容。修订版本号与字段集合随规范演进，以[规范](https://modelcontextprotocol.io/specification/latest)的当前修订为准。

#### 14.7.3 节：长上下文相关的配置字段

- Hugging Face 模型配置中，`rope_scaling` 描述长度扩展方案，各字段见 A.7.8 的 4.3.5 节条目。
- `num_attention_heads` 与 `num_key_value_heads` 决定 Ulysses 序列并行的并行度上限（不复制 KV 时不超过 KV 头数）与每词元 KV 字节数（[10.2 节](../10_inference_optimization/10.2_kv_cache.md)）。
- vLLM 的 `max_model_len` 是单个请求允许的最大长度。启动时引擎检查 KV 池至少装得下一个这么长的请求，并在日志里按它报告最大并发数；KV 按实际长度分页分配（[11.4 节](../11_serving/11.4_kv_memory_management.md)），这个数是全部请求都跑满时的并发。字段名与默认值随版本变化，以所用版本的官方文档为准。

#### 14.8.6 节：TransformerLens 的钩子名与形状

`HookedTransformer.run_with_cache` 一次前向就把全部中间激活缓存下来，按钩子名取用：

- `blocks.{l}.hook_resid_pre`、`hook_resid_mid`、`hook_resid_post`：残差流的三个位置，形状 `[batch, pos, d_model]`。
- `blocks.{l}.attn.hook_pattern`：Softmax 之后的注意力权重，形状 `[batch, head_index, query_pos, key_pos]`；检测归纳头只需要它。
- `blocks.{l}.attn.hook_z`：各头读出的 context，形状 `[batch, pos, head_index, d_head]`。

配套库：训练与分析 SAE 的 SAELens、做干预实验的 nnsight、公开 SAE 权重的 Gemma Scope。接口名与默认值随版本变化，以当时的官方文档为准。
