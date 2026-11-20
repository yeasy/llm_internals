from __future__ import annotations

import math
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def table_after(path: Path, marker: str) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"{re.escape(marker)}\s*\n((?:\|.*\n)+)", text, re.MULTILINE)
    if match is None:
        raise AssertionError(f"missing numeric table {marker!r} in {path}")
    lines = [line for line in match.group(1).splitlines() if line.strip()]
    cells = [
        [cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines
    ]
    header = cells[0]
    rows = [row for row in cells[2:] if len(row) == len(header)]
    return [dict(zip(header, row)) for row in rows]


class NumericClaimsTests(unittest.TestCase):
    def test_kv_cache_table_recomputes_from_dimensions(self):
        rows = table_after(
            ROOT / "10_inference_optimization" / "10.2_kv_cache.md",
            "<!-- numeric-claim: kv-cache -->",
        )
        self.assertEqual(
            {row["配置"] for row in rows},
            {"Llama 2-7B MHA", "Llama 2-70B GQA", "Llama 2-70B MHA"},
        )
        values = {}
        for row in rows:
            gib = (
                2
                * int(row["B"])
                * int(row["层数 L"])
                * int(row["KV 头 Hkv"])
                * int(row["头维 dh"])
                * int(row["词元 t"])
                * int(row["字节"])
                / 1024**3
            )
            self.assertTrue(
                math.isclose(gib, float(row["KV 缓存 GiB"]), rel_tol=0, abs_tol=1e-9)
            )
            values[row["配置"]] = gib
        self.assertEqual(values["Llama 2-70B MHA"] / values["Llama 2-70B GQA"], 8)
        self.assertEqual(values["Llama 2-7B MHA"], 1)

    def test_7b_tensor_sizes_recompute_from_shapes(self):
        rows = table_after(
            ROOT / "06_training_techniques" / "6.4_batch_sequence.md",
            "<!-- numeric-claim: 7b-tensors -->",
        )
        self.assertEqual(len(rows), 5)
        expected = {
            "隐藏状态": ([1, 2048, 4096], 16),
            "单个 Q、K 或 V": ([1, 32, 2048, 128], 16),
            "一份 MLP 扩展状态": ([1, 2048, 11008], 43),
            "完整注意力矩阵": ([1, 32, 2048, 2048], 256),
            "单层 K 与 V 缓存合计": ([2, 1, 32, 2048, 128], 32),
        }
        self.assertEqual({row["张量"] for row in rows}, set(expected))
        for row in rows:
            shape = [int(value) for value in re.findall(r"\d+", row["形状"])]
            self.assertEqual(shape, expected[row["张量"]][0])
            size = math.prod(shape) * int(row["每元素字节"]) / 1024**2
            self.assertEqual(size, float(row["大小 MiB"]))
            self.assertEqual(size, expected[row["张量"]][1])

    def test_7b_adam_model_states_have_an_explicit_precision_contract(self):
        rows = table_after(
            ROOT / "06_training_techniques" / "6.4_batch_sequence.md",
            "<!-- numeric-claim: 7b-model-states -->",
        )
        expected = {
            "前后向参数": ("FP16/BF16", 2),
            "参数梯度": ("FP16/BF16", 2),
            "主参数": ("FP32", 4),
            "Adam 一阶矩": ("FP32", 4),
            "Adam 二阶矩": ("FP32", 4),
        }
        self.assertEqual({row["项目"] for row in rows}, set(expected))
        for row in rows:
            precision, byte_count = expected[row["项目"]]
            self.assertEqual(row["精度"], precision)
            self.assertEqual(int(row["每参数字节"]), byte_count)
            self.assertEqual(float(row["7B 开销 GB"]), 7e9 * byte_count / 1e9)
        self.assertEqual(sum(float(row["7B 开销 GB"]) for row in rows), 112)

    def test_peft_does_not_claim_freezing_eliminates_or_preserves_all_activations(self):
        text = (ROOT / "08_alignment" / "8.4_peft.md").read_text(encoding="utf-8")
        self.assertNotIn("LoRA 一点也不省激活显存", text)

    def test_zero_partition_table_recomputes_per_rank_memory(self):
        rows = table_after(
            ROOT / "07_distributed_training" / "7.2_zero.md",
            "<!-- numeric-claim: zero-partitions -->",
        )
        self.assertEqual(
            [row["方案"] for row in rows], ["DDP", "ZeRO-1", "ZeRO-2", "ZeRO-3"]
        )
        ddp = None
        actual: dict[str, float] = {}
        for row in rows:
            p, g, o, ranks = (
                float(row["参数 GB"]),
                float(row["梯度 GB"]),
                float(row["优化器 GB"]),
                int(row["数据并行 K"]),
            )
            expected = {
                "DDP": p + g + o,
                "ZeRO-1": p + g + o / ranks,
                "ZeRO-2": p + (g + o) / ranks,
                "ZeRO-3": (p + g + o) / ranks,
            }[row["方案"]]
            actual[row["方案"]] = float(row["单卡模型状态 GB"])
            self.assertTrue(math.isclose(actual[row["方案"]], expected, abs_tol=1e-6))
            if row["方案"] == "DDP":
                ddp = expected
        self.assertIsNotNone(ddp)
        for row in rows:
            saving = ddp / actual[row["方案"]]
            self.assertTrue(math.isclose(saving, float(row["相对 DDP"]), abs_tol=0.01))

    def test_weight_memory_and_single_gpu_fit_flags_recompute(self):
        rows = table_after(
            ROOT / "11_serving" / "11.12_hardware.md",
            "<!-- numeric-claim: weight-memory -->",
        )
        capacities = {
            "H100-80GB": 80,
            "H200-141GB": 141,
            "B200-180GB": 180,
            "B300-288GB": 288,
        }
        self.assertEqual(len(rows), 4)
        for row in rows:
            expected_gb = float(row["参数 B"]) * float(row["位宽 bit"]) / 8
            self.assertTrue(
                math.isclose(expected_gb, float(row["权重 GB"]), abs_tol=1e-9)
            )
            for column, capacity in capacities.items():
                expected_fit = "是" if expected_gb <= capacity else "否"
                self.assertEqual(
                    row[column], expected_fit, f"{row['模型配置']} on {column}"
                )


if __name__ == "__main__":
    unittest.main()
