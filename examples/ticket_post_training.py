"""Prepare and score a synthetic ticket extraction lab. MLX is used only for prediction."""

import argparse
from itertools import product
import json
from pathlib import Path


CATEGORIES = ("network", "account", "hardware")
PRIORITIES = ("high", "normal")
PROMPT = (
    "把工单转成 JSON 对象，只输出 category 和 priority 两个字段。"
    "网络连接问题归为 network，账号登录与权限问题归为 account，设备故障归为 hardware。"
    "影响多人或导致工作完全中断且没有替代方案时，priority 为 high，其余为 normal。"
    "\n工单："
)

# Each split introduces new issue descriptions and new impact descriptions.
SPLITS = {
    "train": (
        (
            ("办公 Wi-Fi 连不上", "网线插上后仍显示未连接", "无线网总掉线",
             "内网网关无响应", "楼层交换机无法通信", "连接公司的 VPN 失败"),
            ("忘记登录密码", "账号被锁定", "没有打开共享目录的权限",
             "登录验证码收不到", "无法完成双重验证", "新密码提示认证失败"),
            ("打印机不进纸", "显示器不亮", "键盘按键失灵",
             "鼠标滚轮卡住", "电脑风扇停转", "笔记本无法充电"),
        ),
        (
            ("整个部门都受影响。", "三个班组无法继续上班。", "我完全无法工作，也没有替代方案。"),
            ("只有我受到影响，手头的工作还能继续。", "可以先用备用方案。", "暂时有绕行办法，业务没有中断。"),
        ),
    ),
    "valid": (
        (("网络连接频繁超时",), ("访问系统时提示账号过期",), ("耳机没有声音",)),
        (("办公室所有同事都遇到了这个问题。",), ("能换一种方式完成手头的工作。",)),
    ),
    "test": (
        (
            ("无线接入点一直拒绝连接", "这台机器无法分配到 IP 地址", "公司有线网络时断时续"),
            ("身份验证应用不再弹出确认", "账户无法重置口令", "新的权限还没有开通"),
            ("扫码枪的触发键坏了", "投影仪开机后黑屏", "触控板按下去没有反应"),
        ),
        (("同一房间里的六位同事都受影响。",), ("只有这个工位受影响，可以先用备用方案。",)),
    ),
}


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False), file=stream)


def prepare(directory):
    directory.mkdir()
    for split, (issues_by_category, impacts_by_priority) in SPLITS.items():
        rows = []
        for category, issues in zip(CATEGORIES, issues_by_category):
            for priority, impacts in zip(PRIORITIES, impacts_by_priority):
                for issue, impact in product(issues, impacts):
                    label = {"category": category, "priority": priority}
                    rows.append({"messages": [
                        {"role": "user", "content": PROMPT + issue + "。" + impact},
                        {"role": "assistant", "content": json.dumps(label, separators=(",", ":"))},
                    ]})
        write_jsonl(directory / f"{split}.jsonl", rows)
        if split == "train":
            write_jsonl(directory / "gemini_train.jsonl", (
                {"contents": [
                    {"role": "user", "parts": [{"text": user["content"]}]},
                    {"role": "model", "parts": [{"text": assistant["content"]}]},
                ]}
                for user, assistant in (row["messages"] for row in rows)
            ))


def unique_keys(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON key")
    return result


def reject_constant(value):
    raise ValueError(f"Invalid JSON constant: {value}")


def score(answers, predictions):
    if len(answers) != len(predictions) or not answers:
        raise ValueError("Answers and predictions must have the same nonzero length")
    counts = dict.fromkeys(
        ("json_object", "schema_valid", "category_correct", "priority_correct", "exact_match"), 0
    )
    for answer, prediction in zip(answers, predictions):
        if not isinstance(prediction, str):
            raise ValueError("Each prediction must be a JSON string containing the raw model output")
        try:
            record = json.loads(prediction, object_pairs_hook=unique_keys, parse_constant=reject_constant)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        counts["json_object"] += 1
        category = record.get("category")
        priority = record.get("priority")
        valid = record.keys() == answer.keys() and category in CATEGORIES and priority in PRIORITIES
        counts["schema_valid"] += valid
        counts["category_correct"] += category == answer["category"]
        counts["priority_correct"] += priority == answer["priority"]
        counts["exact_match"] += valid and record == answer
    return {**counts, "total": len(answers)}


def predict(test, model_path, adapter, output):
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(model_path, adapter_path=adapter)
    sampler = make_sampler(temp=0)
    with output.open("x", encoding="utf-8") as stream:
        for row in read_jsonl(test):
            messages = row["messages"][:1]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            response = generate(model, tokenizer, prompt=prompt, max_tokens=64, sampler=sampler)
            print(json.dumps(response, ensure_ascii=False), file=stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare", help="write train, validation, test, and Gemini JSONL")
    preparation.add_argument("directory", type=Path)
    prediction = commands.add_parser("predict", help="generate one raw response per held-out prompt")
    prediction.add_argument("test", type=Path)
    prediction.add_argument("--model", required=True)
    prediction.add_argument("--adapter")
    prediction.add_argument("--output", type=Path, required=True)
    scoring = commands.add_parser("score", help="score the raw responses against held-out labels")
    scoring.add_argument("test", type=Path)
    scoring.add_argument("predictions", type=Path)
    args = parser.parse_args()

    if args.command == "prepare":
        prepare(args.directory)
    elif args.command == "predict":
        predict(args.test, args.model, args.adapter, args.output)
    else:
        answers = [json.loads(row["messages"][1]["content"]) for row in read_jsonl(args.test)]
        print(json.dumps(score(answers, read_jsonl(args.predictions)), indent=2))


if __name__ == "__main__":
    main()
