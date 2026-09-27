"""CPU teaching example: train -> save -> load -> autoregressive generation.

Run from the book directory:
    python examples/tiny_gpt.py --checkpoint /tmp/tiny-gpt.pt
The four training strings demonstrate memorization, not arithmetic generalization.
"""
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


CORPUS = ("1+1=2。", "1+2=3。", "2+2=4。", "2+3=5。")
VOCAB = ["<eos>"] + sorted(set("".join(CORPUS)))
EOS = 0
CONTEXT = 16
WIDTH = 32
HEADS = 2
LAYERS = 2


def encode(text):
    if any(char not in VOCAB for char in text):
        raise ValueError("Input contains a character outside the teaching vocabulary")
    return [VOCAB.index(char) for char in text]


def training_batch():
    # Four equal-length sequences: no padding or padding loss mask is needed.
    tokens = torch.tensor([encode(text) + [EOS] for text in CORPUS])
    return tokens[:, :-1], tokens[:, 1:]


class CausalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = HEADS
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.output = nn.Linear(WIDTH, WIDTH)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # [B, T, 32] -> [B, 2, T, 16] for each of Q, K and V.
        q, k, v = [a.view(batch, length, self.heads, width // self.heads)
                   .transpose(1, 2) for a in (q, k, v)]
        scores = q @ k.transpose(-2, -1) / (width // self.heads) ** 0.5
        future = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        weights = scores.masked_fill(future, float("-inf")).softmax(dim=-1)
        context = (weights @ v).transpose(1, 2).contiguous().view(batch, length, width)
        return self.output(context)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(WIDTH)
        self.attention = CausalAttention()
        self.norm2 = nn.LayerNorm(WIDTH)
        self.mlp = nn.Sequential(nn.Linear(WIDTH, 4 * WIDTH), nn.GELU(),
                                 nn.Linear(4 * WIDTH, WIDTH))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(len(VOCAB), WIDTH)
        self.position_embedding = nn.Embedding(CONTEXT, WIDTH)
        self.blocks = nn.Sequential(*(Block() for _ in range(LAYERS)))
        self.norm = nn.LayerNorm(WIDTH)
        self.lm_head = nn.Linear(WIDTH, len(VOCAB), bias=False)

    def forward(self, token_ids):
        length = token_ids.size(1)
        if not 0 < length <= CONTEXT:
            raise ValueError(f"Sequence length must be between 1 and {CONTEXT}")
        positions = torch.arange(length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        return self.lm_head(self.norm(self.blocks(x)))


def train(model, steps=200):
    if steps < 1:
        raise ValueError("steps must be positive")
    model.train()
    inputs, targets = training_batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, len(VOCAB)), targets.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def generate(model, prompt, max_new_tokens=8):
    ids = encode(prompt)
    if not 0 < len(ids) <= CONTEXT or max_new_tokens < 0:
        raise ValueError("Prompt must be nonempty and fit the context; token limit must be nonnegative")
    model.eval()
    for _ in range(max_new_tokens):
        if len(ids) >= CONTEXT:
            break
        # Recompute the full prefix for clarity; this example does not use KV cache.
        logits = model(torch.tensor([ids]))
        next_id = logits[0, -1].argmax().item()
        if next_id == EOS:
            break
        ids.append(next_id)
    return "".join(VOCAB[index] for index in ids)


def save_model(model, path):
    with Path(path).open("xb") as checkpoint:
        torch.save({"state_dict": model.state_dict(), "vocab": VOCAB}, checkpoint)


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["vocab"] != VOCAB:
        raise ValueError("Checkpoint vocabulary does not match this example")
    model = TinyGPT()
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--load-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    if not args.load_only:
        if args.checkpoint.exists():
            parser.error("Checkpoint already exists; choose a new path or use --load-only")
        model = TinyGPT()
        losses = train(model, args.steps)
        save_model(model, args.checkpoint)
        print(f"loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    restored = load_model(args.checkpoint)
    print(f"generation: {generate(restored, '2+3=')}")


if __name__ == "__main__":
    main()
