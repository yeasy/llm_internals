import unittest

from examples.paper_summary import make_datasets, render_prompt, review_scores


class TeachingTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt=False, return_dict=False):
        tokens = []
        for message in messages:
            tokens.append(message["role"])
            tokens.extend(message["content"].split())
        if add_generation_prompt:
            tokens.append("assistant")
        return tokens


class PaperSummaryTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"paper_id": split, "split": split, "body": f"{split} source",
             "summary": f"{split} target"}
            for split in ("train", "valid", "test")
        ]
        self.tokenizer = TeachingTokenizer()

    def test_holdout_is_not_in_training_or_generation_prompt(self):
        datasets = make_datasets(self.records, self.tokenizer, 200)
        self.assertEqual(datasets["train"][0]["paper_id"], "train")
        self.assertEqual(datasets["test"][0]["paper_id"], "test")
        self.assertEqual(datasets["train"][0]["messages"][-1]["content"], "train target")
        self.assertNotIn("test source", datasets["train"][0]["messages"][0]["content"])
        prompt = render_prompt(datasets["test"][0], self.tokenizer, 200)
        self.assertNotIn("target", prompt)
        self.assertEqual(prompt[-1], "assistant")
        with self.assertRaisesRegex(ValueError, "test.*生成输入"):
            render_prompt(datasets["test"][0], self.tokenizer, len(prompt) - 1)

    def test_rejects_leakage_and_truncation(self):
        with self.assertRaisesRegex(ValueError, "重复.*train"):
            make_datasets(self.records + [{**self.records[0], "split": "test"}],
                          self.tokenizer, 200)
        with self.assertRaisesRegex(ValueError, "test"):
            make_datasets(self.records[:-1], self.tokenizer, 200)
        with self.assertRaisesRegex(ValueError, "train.*序列"):
            make_datasets(self.records, self.tokenizer, 3)

    def test_scores_join_by_id_and_empty_output_cannot_pass(self):
        papers = [{"paper_id": key} for key in ("a", "b", "c")]
        predictions = [
            {"paper_id": "c", "output": " "},
            {"paper_id": "a", "output": "supported"},
            {"paper_id": "b", "output": "incorrect"},
        ]
        reviews = [
            {"paper_id": "b", "unsupported_claims": 2, "missed_key_points": 1},
            {"paper_id": "c", "unsupported_claims": 0, "missed_key_points": 0},
            {"paper_id": "a", "unsupported_claims": 0, "missed_key_points": 0},
        ]
        self.assertEqual(review_scores(papers, predictions, reviews), {
            "papers": 3, "empty_outputs": 1, "unsupported_claims": 2,
            "missed_key_points": 1, "passed": 1,
        })
        with self.assertRaisesRegex(ValueError, "paper_id"):
            review_scores(papers, predictions, reviews[:-1])
        reviews[0]["unsupported_claims"] = -1
        with self.assertRaisesRegex(ValueError, "b.*unsupported_claims"):
            review_scores(papers, predictions, reviews)
