"""Numerical counterexamples and prose contracts for architecture boundaries."""
import math
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
ENCODER = ROOT / "12_encoder_models/12.3_longformer_bigbird.md"
FRONTIER = ROOT / "13_decoder_models/13.3_deepseek_gemini.md"
ALIGNMENT = ROOT / "08_alignment/8.2_rlhf.md"


class ArchitectureBoundariesTests(unittest.TestCase):
    def test_local_window_uses_one_sided_radius_for_endpoint_reachability(self):
        text = ENCODER.read_text(encoding="utf-8")
        radius, length, layers = 512 // 2, 4096, 12
        self.assertLess(layers * radius, length - 1)
        hops = math.ceil((length - 1) / radius)
        self.assertIn(f"ceil({length - 1} / {radius}) = {hops}", text)
        self.assertNotIn("12 层足以让任意两个位置连通", text)

    def test_bigbird_table_counts_global_and_boundary_query_blocks(self):
        text = ENCODER.read_text(encoding="utf-8")
        length, block, heads, item_bytes = 4096, 64, 12, 2
        blocks = length // block
        scores = (2 * block * length + 2 * block * 7 * block
                  + (blocks - 4) * block * 8 * block)
        row = next(line for line in text.splitlines()
                   if line.startswith("| BigBird 块稀疏"))
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        number = re.search(r"[\d,]+", cells[1].rsplit("=", 1)[-1])
        self.assertIsNotNone(number)
        self.assertEqual(int(number.group().replace(",", "")), scores)
        self.assertEqual(cells[2], f"1/{length ** 2 / scores:.2f}")
        self.assertEqual(cells[3], f"{scores * heads * item_bytes / 1024 ** 2:.1f} MiB")
        self.assertNotIn("正是摘要里“可处理 8 倍长度”的来源", text)

    def test_zero_advantage_does_not_imply_zero_kl_gradient(self):
        # Bernoulli KL(policy || reference) has nonzero derivative away from reference.
        p, q = 0.8, 0.5
        def kl(probability):
            return (probability * math.log(probability / q)
                    + (1 - probability) * math.log((1 - probability) / (1 - q)))
        step = 1e-6
        derivative = (kl(p + step) - kl(p - step)) / (2 * step)
        self.assertGreater(abs(derivative), 1)
        for path in (FRONTIER, ALIGNMENT):
            with self.subTest(path=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertIn("总奖励", text)
                self.assertIn("KL 梯度仍可能非零", text)
                self.assertIn("格式奖励", text)
                for claim in ("这一组对梯度没有任何贡献", "该组对梯度毫无贡献",
                              "整组优势归零、不产生梯度"):
                    self.assertNotIn(claim, text)

    def test_gemini_disclosure_distinguishes_known_structure_from_missing_config(self):
        text = FRONTIER.read_text(encoding="utf-8")
        section = text.split("### 13.3.4", 1)[1].split("### 13.3.5", 1)[0]
        self.assertNotIn("均未公开模型架构", section)
        self.assertNotIn("公布的是能力与评测结果，不是结构", section)
        for disclosed in ("MoE", "Flash", "稠密", "并行", "完整配置"):
            self.assertIn(disclosed, section)

    def test_flat_search_cost_includes_vector_bandwidth(self):
        text = ENCODER.read_text(encoding="utf-8")
        paragraph = text.split("**向量检索这一步。**", 1)[1].split("\n\n", 1)[0]
        vectors, dimensions, item_bytes = 1_000_000, 768, 4
        operations = 2 * vectors * dimensions
        byte_count = vectors * dimensions * item_bytes
        self.assertIn(f"{byte_count / 1e9:.3f} GB", paragraph)
        self.assertIn(f"{operations / byte_count:.1f} FLOP/byte", paragraph)
        self.assertIn("带宽", paragraph)
        self.assertNotIn("不是为百万级准备的", paragraph)
        self.assertIn("Guidelines-to-choose-an-index", paragraph)


if __name__ == "__main__":
    unittest.main()
