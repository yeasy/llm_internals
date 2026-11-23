"""Train a tiny CPU Transformer with PPO. Each answer has two character tokens."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical

from post_training import clipped_policy_loss
from tiny_gpt import TinyGPT, WIDTH


RESPONSE_TOKENS = "巴黎里昂马赛柏林"
ANSWER_LENGTH = 2
SAMPLES = 96


def load_questions(path):
    questions = []
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            location = f"{path}:{number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{location}: {exc.msg}") from exc
            if not isinstance(row, dict) or not {"prompt", "answer"} <= row.keys():
                raise ValueError(f"{location}: 需要 prompt 和 answer")
            prompt, answer = row["prompt"], row["answer"]
            if not isinstance(prompt, str) or not prompt.strip() or prompt in seen:
                raise ValueError(f"{location}: prompt 必须是非空且不重复的问题")
            if (not isinstance(answer, str) or len(answer) != ANSWER_LENGTH
                    or any(token not in RESPONSE_TOKENS for token in answer)):
                raise ValueError(f"{location}: answer 必须是演示词表中的两个字")
            seen.add(prompt)
            questions.append((prompt, answer))
    if not questions:
        raise ValueError(f"{path}: 没有可训练的问题")
    return questions


def response_policy(model, prefixes, start):
    logits = model(prefixes)[:, start:, :len(RESPONSE_TOKENS)]
    return Categorical(logits=logits)


@torch.no_grad()
def sample(model, prompt):
    prefixes = prompt.unsqueeze(0).expand(SAMPLES, -1)
    sampled = []
    log_probs = []
    for _ in range(ANSWER_LENGTH):
        distribution = response_policy(model, prefixes, -1)
        action = distribution.sample()
        sampled.append(action)
        log_probs.append(distribution.log_prob(action))
        prefixes = torch.cat((prefixes, action), dim=1)
    return prefixes[:, :-1], torch.cat(sampled, dim=1), torch.cat(log_probs, dim=1)


@torch.no_grad()
def answer_probability(model, prompt, answer):
    prefixes = torch.cat((prompt, answer[:-1])).unsqueeze(0)
    distribution = response_policy(model, prefixes, prompt.numel() - 1)
    return distribution.log_prob(answer.unsqueeze(0)).sum().exp().item()


def train(questions, batches):
    prompt_tokens = set("".join(prompt for prompt, _ in questions))
    vocabulary = list(RESPONSE_TOKENS) + sorted(prompt_tokens - set(RESPONSE_TOKENS))
    token_ids = {token: i for i, token in enumerate(vocabulary)}
    data = [
        (torch.tensor([token_ids[t] for t in prompt]), torch.tensor([token_ids[t] for t in answer]))
        for prompt, answer in questions
    ]
    context_length = max(prompt.numel() for prompt, _ in data) + ANSWER_LENGTH - 1
    actor = TinyGPT(len(vocabulary), context_length)
    critic = TinyGPT(len(vocabulary), context_length)
    critic.lm_head = nn.Linear(WIDTH, 1)
    optimizer = torch.optim.AdamW(list(actor.parameters()) + list(critic.parameters()), lr=0.003)
    before = [answer_probability(actor, prompt, answer) for prompt, answer in data]

    for batch in range(batches):
        rollouts = []
        with torch.no_grad():
            for prompt, answer in data:
                prefixes, actions, old_logp = sample(actor, prompt)
                start = prompt.numel() - 1
                # Only the verifier compares generated tokens with the answer.
                reward = (actions == answer).all(dim=1).float()
                returns = reward.unsqueeze(1).expand_as(old_logp)
                advantages = returns - critic(prefixes)[:, start:, 0]
                rollouts.append((prefixes, start, actions, old_logp, returns, advantages))
        if batch == 0:
            actions, old_logp, returns = rollouts[0][2:5]
            letters = "、".join("".join(RESPONSE_TOKENS[i] for i in row) for row in actions[:4].tolist())
            print(f"第一道题前四次生成：{letters}")
            probabilities = "、".join(f"{p:.2%}" for p in old_logp[0].exp().tolist())
            print(f"第一条的逐词元旧概率：{probabilities}，奖励：{returns[0, 0]:.0f}")
            print(f"第一批第一道题答对：{returns[:, 0].sum().item():.0f}/{SAMPLES}")

        for _ in range(3):
            optimizer.zero_grad()
            losses = []
            for prefixes, start, actions, old_logp, returns, advantages in rollouts:
                new_logp = response_policy(actor, prefixes, start).log_prob(actions)
                policy_loss = clipped_policy_loss(new_logp, old_logp, advantages)
                predicted = critic(prefixes)[:, start:, 0]
                losses.append(policy_loss + (predicted - returns).square().mean())
            torch.stack(losses).mean().backward()
            optimizer.step()

    for (question, label), (prompt, answer), initial in zip(questions, data, before):
        updated = answer_probability(actor, prompt, answer)
        print(f"{question} {label}：{initial:.2%} → {updated:.2%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--batches", type=int, default=30)
    args = parser.parse_args()
    if args.batches < 1:
        parser.error("--batches 必须为正整数")
    try:
        questions = load_questions(args.data)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    torch.set_num_threads(1)
    torch.manual_seed(7)
    train(questions, args.batches)


if __name__ == "__main__":
    main()
