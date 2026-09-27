"""CPU-only post-training building blocks; no models or datasets are downloaded.

Run: python examples/post_training.py

SFT uses a toy next-token lookup table; LoRA wraps one linear layer; DPO uses
two possible one-token responses. GRPO demonstrates grouped advantages and a
clipped policy objective only. This is not a complete PPO/RLHF/GRPO trainer:
there is no rollout engine, reward model, value model, KL penalty, or tokenizer.
The hand-written assistant mask represents a prepared dataset/template output.

Sources:
https://huggingface.co/docs/trl/sft_trainer
https://huggingface.co/docs/peft/package_reference/lora
https://arxiv.org/abs/2305.18290 (DPO, equation 7)
https://arxiv.org/abs/2402.03300 (DeepSeekMath, GRPO)
https://huggingface.co/docs/trl/grpo_trainer
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


IGNORE_INDEX = -100


def sft_batch(tokens, assistant_mask, attention_mask=None):
    """Shift [batch, length] IDs and the *target-token* mask exactly once.

    True in assistant_mask means that token is a response target. EOS may be
    included; role markers and padding must follow the dataset's chosen policy.
    attention_mask here selects valid targets; it is not a model attention mask.
    Use contiguous, right-padded examples for this teaching function.
    """
    if tokens.ndim != 2 or tokens.shape[1] < 2 or tokens.dtype != torch.long:
        raise ValueError("tokens must be int64 [batch, length >= 2]")
    if assistant_mask.shape != tokens.shape or assistant_mask.dtype != torch.bool:
        raise ValueError("assistant_mask must be boolean with the tokens shape")
    if attention_mask is None:
        attention_mask = torch.ones_like(assistant_mask)
    if attention_mask.shape != tokens.shape or attention_mask.dtype != torch.bool:
        raise ValueError("attention_mask must be boolean with the tokens shape")
    labels = tokens[:, 1:].clone()
    keep = assistant_mask[:, 1:] & attention_mask[:, 1:]
    labels.masked_fill_(~keep, IGNORE_INDEX)
    return tokens[:, :-1], labels


def _check_targets(logits, labels):
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or labels.dtype != torch.long:
        raise ValueError("expected logits [batch, length, vocabulary] and int64 labels [batch, length]")
    return labels.ne(IGNORE_INDEX)


def sft_loss(logits, labels):
    """Mean cross-entropy over supervised tokens, not a mean of example means."""
    keep = _check_targets(logits, labels)
    if not keep.any():
        raise ValueError("SFT requires at least one supervised token")
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                           ignore_index=IGNORE_INDEX, reduction="mean")


def sequence_log_probs(logits, labels):
    """Sum response-token log probabilities for each sequence (the DPO input).

    logits and labels are already shifted, as returned by sft_batch/model(x).
    Padding/prompt labels are ignored. This sum is deliberately not length
    normalized: replacing it with a mean changes the original DPO objective.
    """
    keep = _check_targets(logits, labels)
    if not keep.any(dim=-1).all():
        raise ValueError("Each response requires at least one supervised token")
    safe_labels = labels.masked_fill(~keep, 0)
    token_logp = logits.log_softmax(-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return token_logp.masked_fill(~keep, 0).sum(-1)


class LoRALinear(nn.Module):
    """W x + (alpha/r) B A x; the base layer, including its bias, is frozen.

    A: [rank, in_features], B: [out_features, rank]. The supplied base module
    is reused and frozen in place. No dropout, quantization, or merge is used.
    """
    def __init__(self, base, rank=2, alpha=2.):
        super().__init__()
        if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            raise ValueError("rank must be a positive integer")
        self.base = base
        self.base.requires_grad_(False)
        self.scale = alpha / rank
        self.A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(x, self.A), self.B)


def dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=0.1):
    """Original sigmoid DPO, averaged over pairs of answers to the same prompt.

    Inputs are response sequence log-probability sums with shape [pairs]. The
    fixed reference is detached. Stable logsigmoid avoids log(sigmoid(x))
    underflow when the preferred response initially has a very poor margin.
    """
    values = (policy_chosen, policy_rejected, reference_chosen, reference_rejected)
    if policy_chosen.ndim != 1 or policy_chosen.numel() == 0:
        raise ValueError("DPO requires nonempty vectors of paired log probabilities")
    if any(value.shape != policy_chosen.shape for value in values) or beta <= 0:
        raise ValueError("DPO inputs must have identical shapes and beta must be positive")
    policy_margin = policy_chosen - policy_rejected
    reference_margin = reference_chosen.detach() - reference_rejected.detach()
    return -F.logsigmoid(beta * (policy_margin - reference_margin)).mean()


def group_relative_advantages(rewards, epsilon=1e-4):
    """Normalize rewards within each prompt's group: [prompts, responses >= 2].

    Uses sample std (correction=1), with epsilon added to the denominator.
    Equal rewards give exactly zero advantage. Singleton groups are rejected,
    since sample std is undefined. Rewards are fixed observations, not learned
    through this objective. This illustrates group scaling, not all TRL modes.
    """
    if rewards.ndim != 2 or rewards.shape[1] < 2 or rewards.shape[0] == 0:
        raise ValueError("rewards must have shape [prompts >= 1, responses >= 2]")
    if not rewards.is_floating_point() or epsilon <= 0:
        raise ValueError("rewards must be floating point and epsilon must be positive")
    rewards = rewards.detach()
    return (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + epsilon)


def clipped_policy_loss(new_logp, old_logp, advantages, clip_epsilon=0.2):
    """Negative clipped surrogate mean for fixed sampled actions.

    All three tensors have the same shape; each entry is one sampled action.
    The demo uses single-token responses, so token and response are identical.
    For real variable-length responses, token masks, the chosen reduction, KL,
    and rollout-policy accounting must be specified separately.
    """
    if new_logp.shape != old_logp.shape or new_logp.shape != advantages.shape or new_logp.numel() == 0:
        raise ValueError("new/old log probabilities and advantages must have the same nonempty shape")
    if not 0 < clip_epsilon < 1:
        raise ValueError("clip_epsilon must be between zero and one")
    ratio = (new_logp - old_logp.detach()).exp()
    advantages = advantages.detach()
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
    return -torch.minimum(ratio * advantages, clipped * advantages).mean()


def main():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    print("CPU core experiments; not a complete PPO/RLHF/GRPO trainer; no downloads.")

    # 0=padding, 1=user, 2=question, 3=assistant, 4=answer, 5=EOS.
    tokens = torch.tensor([[1, 2, 3, 4, 5, 0], [1, 2, 3, 4, 4, 5]])
    assistant = torch.tensor([[0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1]], dtype=torch.bool)
    x, labels = sft_batch(tokens, assistant, tokens.ne(0))
    # Lookup table is a trainable bigram distribution, not a Transformer.
    model = nn.Embedding(6, 6)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    logits = model(x)
    before = sft_loss(logits, labels)
    optimizer.zero_grad()
    before.backward()
    optimizer.step()
    with torch.no_grad():
        after = sft_loss(model(x), labels)
    print(f"SFT input_shape={tuple(x.shape)} logits_shape={tuple(logits.shape)} labels={labels.tolist()}")
    print(f"SFT supervised_tokens={labels.ne(IGNORE_INDEX).sum().item()} loss={before.item():.6f}->{after.item():.6f}")
    print(f"SFT response_logp_sums={sequence_log_probs(model(x), labels).detach().tolist()}")

    layer = LoRALinear(nn.Linear(4, 3), rank=2, alpha=4.)
    inputs = torch.ones(2, 4)
    initial_error = (layer(inputs) - layer.base(inputs)).abs().max().item()
    base_before = [p.detach().clone() for p in layer.base.parameters()]
    a_before, b_before = layer.A.detach().clone(), layer.B.detach().clone()
    optimizer = torch.optim.SGD([p for p in layer.parameters() if p.requires_grad], lr=0.1)
    loss = layer(inputs).square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    base_change = max((p - old).abs().max().item() for p, old in zip(layer.base.parameters(), base_before))
    print(f"LoRA A_shape={tuple(layer.A.shape)} B_shape={tuple(layer.B.shape)} scale={layer.scale:.1f} initial_max_error={initial_error:.6f}")
    print(f"LoRA base_max_change={base_change:.6f} A_max_change={(layer.A - a_before).abs().max().item():.6f} B_max_change={(layer.B - b_before).abs().max().item():.6f}")
    print("LoRA first-step A gradient is zero because B starts at zero; B can update.")

    # Two possible one-token answers to one prompt: index 0 is preferred.
    scores = nn.Parameter(torch.zeros(2))
    reference = scores.detach().log_softmax(-1)
    policy = scores.log_softmax(-1)
    loss = dpo_loss(policy[:1], policy[1:], reference[:1], reference[1:], beta=1.)
    optimizer = torch.optim.SGD([scores], lr=0.5)
    optimizer.zero_grad()
    loss.backward()
    gradient = scores.grad.detach().tolist()
    optimizer.step()
    updated = scores.log_softmax(-1)
    updated_loss = dpo_loss(updated[:1], updated[1:], reference[:1], reference[1:], beta=1.)
    print(f"DPO loss={loss.item():.6f}->{updated_loss.item():.6f} preferred_probability={policy[0].exp().item():.6f}->{updated[0].exp().item():.6f} score_grad={gradient}")
    print(f"DPO policy_margin={(updated[0] - updated[1]).item():.6f} reference_margin={(reference[0] - reference[1]).item():.6f}")

    rewards = torch.tensor([[0., 1., 2.], [5., 5., 5.]])
    advantages = group_relative_advantages(rewards)
    scores = nn.Parameter(torch.zeros_like(rewards))
    old_logp = scores.detach().log_softmax(-1)
    optimizer = torch.optim.SGD([scores], lr=0.5)
    loss = clipped_policy_loss(scores.log_softmax(-1), old_logp, advantages)
    optimizer.zero_grad()
    loss.backward()
    gradient_norm = scores.grad.norm().item()
    optimizer.step()
    print(f"GRPO rewards_shape={tuple(rewards.shape)} advantages={advantages.tolist()} group_means={advantages.mean(-1).tolist()}")
    print(f"GRPO surrogate_loss={loss.item():.6f} score_grad_norm={gradient_norm:.6f} updated_probabilities={scores.softmax(-1).detach().tolist()}")
    print("GRPO zero mean can give zero surrogate loss at ratio=1 while the gradient is nonzero.")


if __name__ == "__main__":
    main()
