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
            {row["配置"] for row in rows}, {"Llama 2-70B GQA", "Llama 2-70B MHA"}
        )
        values = {}
        for row in rows:
            gib = (
                2
                * int(row["B"])
                * int(row["层数L"])
                * int(row["KV头Hkv"])
                * int(row["头维dh"])
                * int(row["词元t"])
                * int(row["字节"])
                / 1024**3
            )
            self.assertTrue(
                math.isclose(gib, float(row["KV缓存GiB"]), rel_tol=0, abs_tol=1e-9)
            )
            values[row["配置"]] = gib
        self.assertEqual(values["Llama 2-70B MHA"] / values["Llama 2-70B GQA"], 8)

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
                float(row["参数GB"]),
                float(row["梯度GB"]),
                float(row["优化器GB"]),
                int(row["数据并行K"]),
            )
            expected = {
                "DDP": p + g + o,
                "ZeRO-1": p + g + o / ranks,
                "ZeRO-2": p + (g + o) / ranks,
                "ZeRO-3": (p + g + o) / ranks,
            }[row["方案"]]
            actual[row["方案"]] = float(row["单卡模型状态GB"])
            self.assertTrue(math.isclose(actual[row["方案"]], expected, abs_tol=1e-6))
            if row["方案"] == "DDP":
                ddp = expected
        self.assertIsNotNone(ddp)
        for row in rows:
            saving = ddp / actual[row["方案"]]
            self.assertTrue(math.isclose(saving, float(row["相对DDP"]), abs_tol=0.01))

    def test_weight_memory_and_single_gpu_fit_flags_recompute(self):
        rows = table_after(
            ROOT / "11_serving" / "11.4_hardware.md",
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
            expected_gb = float(row["参数B"]) * float(row["位宽bit"]) / 8
            self.assertTrue(
                math.isclose(expected_gb, float(row["权重GB"]), abs_tol=1e-9)
            )
            for column, capacity in capacities.items():
                expected_fit = "是" if expected_gb <= capacity else "否"
                self.assertEqual(
                    row[column], expected_fit, f"{row['模型配置']} on {column}"
                )


if __name__ == "__main__":
    unittest.main()
