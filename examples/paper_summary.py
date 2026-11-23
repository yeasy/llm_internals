"""Prepare data, generate summaries, and tally human reviews for A.7."""

import argparse
import json
from pathlib import Path

PROMPT = (
    "根据论文正文写一段中文摘要。说明研究问题、方法、主要结果和局限。"
    "保留关键数字及适用条件，只写原文能够支持的结论。\n\n论文正文：\n"
)
SPLITS = ("train", "valid", "test")
REVIEW_FIELDS = ("unsupported_claims", "missed_key_points")


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False), file=stream)


def by_paper_id(rows):
    indexed = {}
    for row in rows:
        paper_id = row.get("paper_id") if isinstance(row, dict) else None
        if not isinstance(paper_id, str) or not paper_id.strip():
            raise ValueError("每条记录都要有非空的 paper_id")
        if paper_id in indexed:
            raise ValueError(f"重复的 paper_id: {paper_id}")
        indexed[paper_id] = row
    return indexed


def render_prompt(row, tokenizer, max_input_tokens):
    tokens = tokenizer.apply_chat_template(
        row["messages"][:-1], add_generation_prompt=True, return_dict=False,
    )
    if len(tokens) > max_input_tokens:
        raise ValueError(f"{row['paper_id']}: 生成输入有 {len(tokens)} 个词元，超过 {max_input_tokens}")
    return tokens


def make_datasets(records, tokenizer, max_seq_length):
    data = {split: [] for split in SPLITS}
    for paper_id, record in by_paper_id(records).items():
        split = record.get("split")
        if split not in SPLITS:
            raise ValueError(f"{paper_id}: split 应为 train、valid 或 test")
        for key in ("body", "summary"):
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise ValueError(f"{paper_id}: {key} 应为非空文本")
        row = {"paper_id": paper_id, "messages": [
            {"role": "user", "content": PROMPT + record["body"]},
            {"role": "assistant", "content": record["summary"]},
        ]}
        tokens = tokenizer.apply_chat_template(row["messages"], return_dict=False)
        if len(tokens) > max_seq_length:
            raise ValueError(f"{paper_id}: 完整序列有 {len(tokens)} 个词元，超过 {max_seq_length}")
        prefix = render_prompt(row, tokenizer, max_seq_length)
        if not prefix or len(prefix) >= len(tokens) or tokens[:len(prefix)] != prefix:
            raise ValueError(f"{paper_id}: 聊天模板未保留可分离的助手回答")
        data[split].append(row)
    for split in SPLITS:
        if not data[split]:
            raise ValueError(f"缺少 {split} 数据")
    return data


def review_scores(papers, predictions, reviews):
    papers, predictions, reviews = map(by_paper_id, (papers, predictions, reviews))
    if not papers or papers.keys() != predictions.keys() or papers.keys() != reviews.keys():
        raise ValueError("论文、生成结果与人工复核必须有完全相同且非空的 paper_id 集合")
    scores = {"papers": len(papers), "empty_outputs": 0, **dict.fromkeys(REVIEW_FIELDS, 0),
              "passed": 0}
    for paper_id, prediction in predictions.items():
        output = prediction.get("output")
        if not isinstance(output, str):
            raise ValueError(f"{paper_id}: output 应为文本")
        counts = [reviews[paper_id].get(field) for field in REVIEW_FIELDS]
        for field, count in zip(REVIEW_FIELDS, counts):
            if type(count) is not int or count < 0:
                raise ValueError(f"{paper_id}: {field} 应为非负整数")
            scores[field] += count
        scores["empty_outputs"] += not output.strip()
        scores["passed"] += bool(output.strip()) and not any(counts)
    return scores


def prepare(args):
    from mlx_lm.tokenizer_utils import load

    data = make_datasets(read_jsonl(args.source), load(args.model), args.max_seq_length)
    args.directory.mkdir()
    for split, rows in data.items():
        write_jsonl(args.directory / f"{split}.jsonl", rows)
    print(json.dumps({split: len(rows) for split, rows in data.items()}))


def predict(args):
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    papers = by_paper_id(read_jsonl(args.test))
    model, tokenizer = load(args.model, adapter_path=args.adapter)
    prompts = {paper_id: render_prompt(row, tokenizer, args.max_input_tokens)
               for paper_id, row in papers.items()}
    sampler = make_sampler(temp=0)
    predictions = [
        {"paper_id": paper_id, "output": generate(
            model, tokenizer, tokens, max_tokens=args.max_new_tokens, sampler=sampler,
        )}
        for paper_id, tokens in prompts.items()
    ]
    write_jsonl(args.output, predictions)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("词元上限应为正整数")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("source", type=Path)
    preparation.add_argument("directory", type=Path)
    preparation.add_argument("--model", type=Path, required=True)
    preparation.add_argument("--max-seq-length", type=positive_int, required=True)
    prediction = commands.add_parser("predict")
    prediction.add_argument("test", type=Path)
    prediction.add_argument("--model", type=Path, required=True)
    prediction.add_argument("--adapter", type=Path)
    prediction.add_argument("--output", type=Path, required=True)
    prediction.add_argument("--max-input-tokens", type=positive_int, required=True)
    prediction.add_argument("--max-new-tokens", type=positive_int, required=True)
    review = commands.add_parser("review")
    review.add_argument("test", type=Path)
    review.add_argument("predictions", type=Path)
    review.add_argument("reviews", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "predict":
        predict(args)
    else:
        scores = review_scores(*(read_jsonl(path) for path in (args.test, args.predictions, args.reviews)))
        print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
