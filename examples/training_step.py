"""Hand-check one output-head SGD step; this is NOT a Transformer backward pass.

Run: python examples/training_step.py
X represents two already-computed hidden states, W projects them to three tokens.
We differentiate X for inspection but update only W and b. Float64 reduces
round-off in the central finite-difference check (epsilon = 1e-6).
"""
import math

import torch
from torch.nn import functional as F


def run_step(lr=0.1):
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    x = torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=torch.float64)
    weight = torch.tensor([[0.1, -0.2, 0.3], [0.4, 0.0, -0.1]], dtype=torch.float64)
    bias = torch.tensor([0.0, 0.1, -0.1], dtype=torch.float64)
    targets = torch.tensor([2, 0])
    logits = x @ weight + bias
    probabilities = logits.softmax(-1)
    # Mean cross-entropy: L = -sum(log P[row, target])) / N.
    loss = -logits.log_softmax(-1)[torch.arange(2), targets].mean()
    # dZ = (P - one_hot(y)) / N; the 1/N matches the mean reduction.
    dz = (probabilities - F.one_hot(targets, num_classes=3)) / x.size(0)
    manual = {"x": dz @ weight.T, "weight": x.T @ dz, "bias": dz.sum(0)}

    values = {"x": x, "weight": weight, "bias": bias}
    leaves = {name: value.clone().requires_grad_() for name, value in values.items()}
    # CrossEntropy expects logits, not probabilities: it includes log_softmax.
    auto_loss = F.cross_entropy(leaves["x"] @ leaves["weight"] + leaves["bias"], targets)
    auto_loss.backward()
    autograd = {name: value.grad.clone() for name, value in leaves.items()}
    torch.testing.assert_close(loss, auto_loss.detach())

    finite_difference = {}
    epsilon = 1e-6
    for name, value in values.items():
        gradient = torch.empty_like(value)
        for index in range(value.numel()):
            plus = {key: tensor.clone() for key, tensor in values.items()}
            minus = {key: tensor.clone() for key, tensor in values.items()}
            plus[name].view(-1)[index] += epsilon
            minus[name].view(-1)[index] -= epsilon
            high = F.cross_entropy(plus["x"] @ plus["weight"] + plus["bias"], targets)
            low = F.cross_entropy(minus["x"] @ minus["weight"] + minus["bias"], targets)
            gradient.view(-1)[index] = (high - low) / (2 * epsilon)
        finite_difference[name] = gradient
        torch.testing.assert_close(manual[name], autograd[name], rtol=0, atol=1e-12)
        torch.testing.assert_close(manual[name], gradient, rtol=0, atol=1e-8)

    # A plain SGD update, without momentum or weight decay.
    updated = {name: values[name] - lr * manual[name] for name in ("weight", "bias")}
    loss_after = F.cross_entropy(x @ updated["weight"] + updated["bias"], targets)
    return {**values, "targets": targets, "logits": logits, "probabilities": probabilities,
            "dlogits": dz, "manual": manual, "autograd": autograd,
            "finite_difference": finite_difference, "updated": updated,
            "loss_before": loss.item(), "loss_after": loss_after.item()}


def main():
    torch.set_num_threads(1)
    torch.set_printoptions(precision=6)
    result = run_step()
    print("Output head only; not a hand-derived full Transformer backward pass.")
    for name in ("x", "weight", "bias", "targets", "logits", "probabilities", "dlogits"):
        print(f"{name} shape={tuple(result[name].shape)}\n{result[name]}")
    for name in ("x", "weight", "bias"):
        gradient = result["manual"][name]
        auto_error = (gradient - result["autograd"][name]).abs().max().item()
        fd_error = (gradient - result["finite_difference"][name]).abs().max().item()
        print(f"d{name} shape={tuple(gradient.shape)}\n{gradient}")
        print(f"  max error: autograd={auto_error:.3e}, finite_difference={fd_error:.3e}")
    for name, value in result["updated"].items():
        print(f"updated {name}\n{value}")
    print(f"mean CE after SGD(lr=0.1): {result['loss_before']:.6f} -> {result['loss_after']:.6f}")


if __name__ == "__main__":
    main()
