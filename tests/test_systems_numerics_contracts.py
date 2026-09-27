"""Focused prose contracts and counterexamples for inference-system claims."""
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SystemsNumericsContracts(unittest.TestCase):
    def read(self, path):
        return (ROOT / path).read_text()

    def test_exact_attention_does_not_promise_bitwise_identity(self):
        for path in ("10_inference_optimization/10.3_flash_attention.md",
                     "10_inference_optimization/summary.md",
                     "14_future_trends/14.1_efficient_attention.md"):
            with self.subTest(path=path):
                text = self.read(path)
                for claim in ("逐位等价", "逐位相同", "不影响质量"):
                    self.assertFalse(claim in text, f"{path}: {claim}")
        text = self.read("10_inference_optimization/10.3_flash_attention.md")
        self.assertTrue("数据类型" in text and "验证" in text)

    def test_kernel_representation_has_explicit_feature_dimension(self):
        text = self.read("14_future_trends/14.1_efficient_attention.md")
        self.assertFalse(r"只要 \operatorname{sim} 非负" in text.replace("$", ""))
        self.assertFalse("不能拆成两个特征的内积" in text)
        self.assertTrue("无限维" in text)
        self.assertTrue("[r, d_v]" in text)
        self.assertTrue("[r]" in text)

    def test_structural_attention_changes_are_not_drop_in_equivalences(self):
        text = self.read("14_future_trends/14.1_efficient_attention.md")
        self.assertTrue("参数共享" in text and "参数化" in text)
        self.assertTrue("重新训练或续训" in text)

    def test_delta_write_documents_unit_key_requirement(self):
        text = self.read("14_future_trends/14.1_efficient_attention.md")
        self.assertTrue("L2 归一化" in text)
        self.assertTrue("单位键" in text and "不正交" in text)

    def test_ssd_requires_scalar_identity_transition(self):
        text = self.read("14_future_trends/14.3_ssm_hybrid.md")
        self.assertTrue(r"\bar{A}_t = a_t I" in text)
        self.assertFalse("两套记号可以互换" in text)
        self.assertTrue(r"C\bar{B}^T" in text)

    def test_disaggregation_ttft_depends_on_proxy_delivery(self):
        metrics = self.read("11_serving/11.13_best_practices.md")
        timeline = self.read("11_serving/11.9_disaggregated_serving.md")
        self.assertFalse("分离式架构下 KV 传输不计入 TTFT" in metrics)
        self.assertTrue("首词元直返" in timeline)
        self.assertTrue("LMCache" in timeline and "TTFT" in timeline)
        self.assertTrue("交接" in metrics)

    def test_reasoning_latency_is_a_scoped_estimate(self):
        text = self.read("14_future_trends/14.6_test_time_scaling.md")
        for claim in ("至少要 48 秒", "无法靠并行摊薄", "不是算力"):
            self.assertFalse(claim in text, claim)
        self.assertTrue("单卡" in text and "投机解码" in text)

    def test_reformer_shared_qk_does_not_rule_out_autoregression(self):
        text = self.read("14_future_trends/14.1_efficient_attention.md")
        paragraph = text.split("**哈希注意力。**", 1)[1].split("\n\n", 1)[0]
        self.assertFalse("没有进入生产" in paragraph)
        self.assertFalse("无法用在" in paragraph)
        self.assertTrue("因果掩码" in paragraph and "自回归" in paragraph)
        self.assertTrue("交叉注意力" in paragraph)

    def test_storing_all_keys_does_not_guarantee_exact_retrieval(self):
        text = self.read("14_future_trends/14.1_efficient_attention.md")
        paragraph = text.split("**第七步：代价在哪。**", 1)[1].split("\n\n", 1)[0]
        self.assertFalse("多大就能分辨多少个位置" in paragraph)
        self.assertTrue("相同" in paragraph and "有限精度" in paragraph)

    def test_identical_visible_keys_receive_identical_softmax_weights(self):
        query = (1.0, 0.0)
        keys = ((1.0, 0.0), (1.0, 0.0))
        values = ((2.0, 0.0), (0.0, 3.0))
        logits = [sum(q * k for q, k in zip(query, key)) / math.sqrt(2)
                  for key in keys]
        exponents = [math.exp(logit - max(logits)) for logit in logits]
        weights = [value / sum(exponents) for value in exponents]
        output = tuple(sum(weights[i] * values[i][j] for i in range(2))
                       for j in range(2))
        self.assertEqual(weights, [0.5, 0.5])
        self.assertEqual(output, (1.0, 1.5))
        self.assertNotIn(output, values)

    def test_nonnegative_matrix_need_not_be_a_feature_gram_matrix(self):
        # A shared real feature map yields a PSD Gram matrix. For v=(1,-1),
        # v^T [[1,2],[2,1]] v = -2, despite every entry being positive.
        matrix = ((1, 2), (2, 1))
        vector = (1, -1)
        self.assertTrue(all(value > 0 for row in matrix for value in row))
        quadratic = sum(vector[i] * matrix[i][j] * vector[j]
                        for i in range(2) for j in range(2))
        self.assertEqual(quadratic, -2)

    def test_delta_full_write_requires_unit_key(self):
        def write_and_read(key):
            old, value, beta = 0.0, 3.0, 1.0
            state = (1 - beta * key * key) * old + beta * key * value
            return key * state

        self.assertEqual(write_and_read(1.0), 3.0)
        self.assertEqual(write_and_read(2.0), 12.0)

    def test_general_diagonal_transition_cannot_be_a_scalar_gate(self):
        inputs = (2.0, 3.0, 5.0)
        keys = ((1.0, 2.0), (3.0, 1.0), (2.0, 4.0))
        queries = ((2.0, 1.0), (1.0, 3.0), (4.0, 2.0))
        gates = (0.5, 0.25, 0.75)

        def recurrence(diagonal):
            state = [0.0, 0.0]
            outputs = []
            for t, value in enumerate(inputs):
                for j in range(2):
                    gate = gates[t] * (0.5 if diagonal and j == 1 else 1.0)
                    state[j] = gate * state[j] + keys[t][j] * value
                outputs.append(sum(queries[t][j] * state[j] for j in range(2)))
            return outputs

        dual = []
        for t in range(len(inputs)):
            output = 0.0
            for s in range(t + 1):
                decay = 1.0
                for r in range(s + 1, t + 1):
                    decay *= gates[r]
                output += decay * sum(queries[t][j] * keys[s][j] for j in range(2)) * inputs[s]
            dual.append(output)
        self.assertEqual(recurrence(diagonal=False), dual)
        self.assertNotEqual(recurrence(diagonal=True)[-1], dual[-1])


if __name__ == "__main__":
    unittest.main()
