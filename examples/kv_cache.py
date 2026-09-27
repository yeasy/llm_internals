"""Inspect prefill and incremental decoding using the unchanged tiny_gpt weights.

Run: python examples/kv_cache.py
No training or downloads are needed: logit equivalence holds for random weights.
Cache layout is tuple((K, V), ...) with each tensor [batch, heads, time, head_dim].
Use only with the same unchanged model and the exact matching token prefix.
This small CPU example has no padding, sliding window, autocast, or serving API.

References:
https://huggingface.co/docs/transformers/main/en/cache_explanation
https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
"""
import torch

if __package__:
    from .tiny_gpt import CONTEXT, VOCAB, TinyGPT, encode
else:
    from tiny_gpt import CONTEXT, VOCAB, TinyGPT, encode


def _past_length(model, token_ids, cache):
    """Validate a dense, unpadded cache before computing any layer."""
    if model.training:
        raise ValueError("Call model.eval() before cached inference")
    if (not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 2
            or min(token_ids.shape) == 0
            or token_ids.dtype not in (torch.int32, torch.int64)):
        raise ValueError("token_ids must be a nonempty integer tensor [batch, time]")
    weight = model.token_embedding.weight
    if token_ids.device != weight.device:
        raise ValueError("token_ids and model must use the same device")
    if (token_ids < 0).any() or (token_ids >= weight.shape[0]).any():
        raise ValueError("token_ids contain an index outside the vocabulary")
    if cache is None:
        return 0
    if not isinstance(cache, (tuple, list)) or len(cache) != len(model.blocks):
        raise ValueError("cache must contain one (K, V) pair per model layer")
    past = None
    for block, pair in zip(model.blocks, cache):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("each cache layer must contain a (K, V) pair")
        k, v = pair
        if (not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor)
                or k.ndim != 4 or k.shape != v.shape):
            raise ValueError("cache K and V must have equal [batch, heads, time, head_dim] shapes")
        heads = block.attention.heads
        if (k.shape[0] != token_ids.shape[0] or k.shape[1] != heads
                or k.shape[3] != weight.shape[1] // heads or k.shape[2] == 0):
            raise ValueError("cache dimensions do not match the model and input batch")
        if any(t.device != weight.device or t.dtype != weight.dtype for t in pair):
            raise ValueError("cache dtype and device must match the model")
        if any(t.requires_grad for t in pair):
            raise ValueError("cache is for inference and must not require gradients")
        if past is not None and past != k.shape[2]:
            raise ValueError("all cache layers must have the same time length")
        past = k.shape[2]
    return past


@torch.no_grad()
def cached_forward(model, token_ids, cache=None):
    """Return logits for NEW tokens and a NEW per-layer cache; never mutate cache.

    cache=None performs prefill. Subsequent calls pass only the unprocessed
    tokens, which may be a single token or a multi-token chunk. Cache provenance
    is the caller's responsibility: shape checks cannot detect different weights
    or a different prefix with the same shape. Do not mutate returned tensors.
    """
    past = _past_length(model, token_ids, cache)
    batch, length = token_ids.shape
    total = past + length
    if total > model.position_embedding.num_embeddings:
        raise ValueError("past + new tokens exceed the model context length")
    positions = torch.arange(past, total, device=token_ids.device)
    x = model.token_embedding(token_ids) + model.position_embedding(positions)

    # A [new_time, total_time] mask, aligned to absolute query positions.
    # For past=2, new=2: [[F, F, F, T], [F, F, F, F]]. True is blocked.
    # A rectangular upper-left causal mask would incorrectly hide old tokens.
    future = torch.arange(total, device=x.device)[None, :] > positions[:, None]
    updated = []
    for layer, block in enumerate(model.blocks):
        attention = block.attention
        width = x.shape[-1]
        head_dim = width // attention.heads
        q, k, v = attention.qkv(block.norm1(x)).chunk(3, dim=-1)
        q, k, v = [t.view(batch, length, attention.heads, head_dim).transpose(1, 2)
                   for t in (q, k, v)]
        if cache is not None:
            old_k, old_v = cache[layer]
            k = torch.cat((old_k, k), dim=2)
            v = torch.cat((old_v, v), dim=2)
        scores = q @ k.transpose(-2, -1) / head_dim ** 0.5
        probabilities = scores.masked_fill(future, float("-inf")).softmax(dim=-1)
        context = (probabilities @ v).transpose(1, 2).contiguous().view(batch, length, width)
        # Match Block.forward exactly: pre-norm attention, residual, pre-norm MLP.
        x = x + attention.output(context)
        x = x + block.mlp(block.norm2(x))
        updated.append((k, v))
    return model.lm_head(model.norm(x)), tuple(updated)


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = TinyGPT().eval()
    tokens = torch.tensor([encode("2+3=5。")])
    for label, chunks in (("single-token decode", (3, 1, 1, 1)),
                          ("chunked decode", (2, 2, 2))):
        print(label)
        cache = None
        offset = 0
        full_positions = 0
        max_error = 0.0
        for size in chunks:
            logits, cache = cached_forward(model, tokens[:, offset:offset + size], cache)
            end = offset + size
            reference = model(tokens[:, :end])[:, offset:]
            error = (logits - reference).abs().max().item()
            torch.testing.assert_close(logits, reference, atol=1e-6, rtol=1e-5)
            max_error = max(max_error, error)
            phase = "prefill" if offset == 0 else "decode"
            print(f"  {phase}: offset={offset}, new={size}, logits={tuple(logits.shape)}")
            for layer, (k, v) in enumerate(cache):
                print(f"    layer {layer}: K={tuple(k.shape)}, V={tuple(v.shape)}")
            full_positions += end
            offset = end
        print(f"  max logit error: {max_error:.3e}")
        print(f"  positions processed per layer: full-prefix={full_positions}, cached={offset}")
    print("Position counts describe repeated projections/MLPs, not measured speedup.")
    print("Attention still reads the growing cache; this tiny CPU run need not be faster.")


if __name__ == "__main__":
    main()
