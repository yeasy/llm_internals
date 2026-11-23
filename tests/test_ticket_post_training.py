"""Regression tests for data and scores shared by both ticket recipes."""

import json
from pathlib import Path
import tempfile
import unittest

from examples.ticket_post_training import prepare, read_jsonl, score


class TicketPostTrainingTests(unittest.TestCase):
    def test_preparation_keeps_test_prompts_out_of_training_and_cloud_upload(self):
        with tempfile.TemporaryDirectory() as work:
            data = Path(work) / "data"
            prepare(data)
            splits = {name: read_jsonl(data / f"{name}.jsonl")
                      for name in ("train", "valid", "test")}
            prompts = [{row["messages"][0]["content"] for row in rows}
                       for rows in splits.values()]
            self.assertTrue(all(prompts))
            self.assertEqual(sum(map(len, prompts)), len(set.union(*prompts)))
            cloud = read_jsonl(data / "gemini_train.jsonl")
            expected = [
                {"contents": [
                    {"role": "user", "parts": [{"text": user["content"]}]},
                    {"role": "model", "parts": [{"text": assistant["content"]}]},
                ]}
                for user, assistant in (row["messages"] for row in splits["train"])
            ]
            self.assertEqual(cloud, expected)
            with self.assertRaises(FileExistsError):
                prepare(data)

    def test_scores_distinguish_syntax_schema_and_field_accuracy(self):
        answers = [{"category": "network", "priority": "high"}] * 7
        outputs = [
            '{"priority":"high","category":"network"}',
            '{"category":"network","priority":"normal"}',
            '{"category":"network"}',
            '{"category":"network","priority":"high","extra":0}',
            '{"category":"account","category":"network","priority":"high"}',
            '{"category":"network","priority":"high","extra":NaN}',
            'Here is your result: {"category":"network","priority":"high"}',
        ]
        self.assertEqual(score(answers, outputs), {
            "json_object": 4,
            "schema_valid": 2,
            "category_correct": 4,
            "priority_correct": 2,
            "exact_match": 1,
            "total": 7,
        })

    def test_scoring_rejects_missing_predictions_and_non_string_rows(self):
        answer = {"category": "network", "priority": "high"}
        with self.assertRaisesRegex(ValueError, "same nonzero length"):
            score([answer], [])
        with self.assertRaisesRegex(ValueError, "JSON string"):
            score([answer], [answer])
        self.assertEqual(score([answer], [json.dumps([])])["json_object"], 0)

    def test_read_jsonl_keeps_unicode_separators_inside_model_output(self):
        output = "第一行\u2028第二行"
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "predictions.jsonl"
            path.write_text(json.dumps(output, ensure_ascii=False) + "\n", encoding="utf-8")
            self.assertEqual(read_jsonl(path), [output])


if __name__ == "__main__":
    unittest.main()
