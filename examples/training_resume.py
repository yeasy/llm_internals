"""CPU TinyGPT training with validation and exact step-boundary resumption.

Run: python examples/training_resume.py --checkpoint /tmp/resume-demo.pt
The default compares 40 uninterrupted steps with 17 + save/load + 23 steps.
Continue a trusted local checkpoint into a NEW file:
    python examples/training_resume.py --resume /tmp/resume-demo.pt \
        --steps 23 --checkpoint /tmp/continued.pt
The checkpoint above is from step 17, not the end of the comparison run.
Three original arithmetic strings train the model; one is held out. This tiny
split teaches validation plumbing and is not evidence of arithmetic generalization.
Exact equality is checked within one CPU/PyTorch environment, not across devices
or releases. There is no AMP, distributed sampler, CUDA RNG, or mid-step resume.
"""
import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

if __package__:
    from .tiny_gpt import CORPUS, VOCAB, WIDTH, HEADS, LAYERS, CONTEXT, TinyGPT, training_batch
else:
    from tiny_gpt import CORPUS, VOCAB, WIDTH, HEADS, LAYERS, CONTEXT, TinyGPT, training_batch


TRAIN_TEXTS = CORPUS[:3]
VALID_TEXTS = CORPUS[3:]
CONTRACT = {"format": 1, "vocab": VOCAB, "train_texts": list(TRAIN_TEXTS),
            "valid_texts": list(VALID_TEXTS), "width": WIDTH, "heads": HEADS,
            "layers": LAYERS, "context": CONTEXT, "optimizer": "AdamW",
            "scheduler": "ExponentialLR", "clip_norm": 1.0}


class Trainer:
    def __init__(self):
        self.model = TinyGPT()  # Two causal Transformer blocks, two heads per block.
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.003, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
        inputs, targets = training_batch()
        self.train_x, self.train_y = inputs[:3], targets[:3]
        self.valid_x, self.valid_y = inputs[3:], targets[3:]
        self.step = 0
        self.order = torch.empty(0, dtype=torch.long)
        self.cursor = 0

    def train(self, steps):
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        self.model.train()
        history = []
        for _ in range(steps):
            if self.cursor == len(self.order):
                self.order = torch.randperm(len(self.train_x))
                self.cursor = 0
            index = self.order[self.cursor].item()
            # Batch size 1 keeps sample order and data progress visible.
            inputs, targets = self.train_x[index:index + 1], self.train_y[index:index + 1]
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(inputs)  # [1, 6] -> [1, 6, vocab_size]
            loss = F.cross_entropy(logits.reshape(-1, len(VOCAB)), targets.reshape(-1))
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0,
                                                  error_if_nonfinite=True)
            lr = self.optimizer.param_groups[0]["lr"]
            self.optimizer.step()
            self.scheduler.step()  # Advance only after the optimizer update.
            self.step += 1
            self.cursor += 1
            history.append({"step": self.step, "sample": index, "loss": loss.item(),
                            "lr": lr, "grad_norm_before_clip": norm.item()})
        return history

    @torch.no_grad()
    def validate(self):
        was_training = self.model.training
        self.model.eval()
        try:
            logits = self.model(self.valid_x)
            return F.cross_entropy(logits.reshape(-1, len(VOCAB)),
                                   self.valid_y.reshape(-1)).item()
        finally:
            self.model.train(was_training)

    def state_dict(self):
        # Only save after complete optimizer/scheduler steps. Gradients are cleared
        # before the next backward pass, so no gradient-accumulation state is needed.
        return {"contract": CONTRACT.copy(), "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                "step": self.step, "cpu_rng": torch.get_rng_state(),
                "data_order": self.order.clone(), "data_cursor": self.cursor}

    def save(self, path):
        # Exclusive creation also refuses an existing symlink: never overwrite.
        with Path(path).open("xb") as stream:
            torch.save(self.state_dict(), stream)

    @classmethod
    def load(cls, path):
        # weights_only restricts unpickling; still load only trusted local files.
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("contract") != CONTRACT:
            raise ValueError("Checkpoint contract does not match this example")
        trainer = cls()  # Construction consumes RNG; restore RNG last.
        trainer.model.load_state_dict(state["model"])
        # Scheduler must exist before optimizer state is restored.
        trainer.scheduler.load_state_dict(state["scheduler"])
        trainer.optimizer.load_state_dict(state["optimizer"])
        trainer.step = state["step"]
        trainer.order = state["data_order"]
        trainer.cursor = state["data_cursor"]
        torch.set_rng_state(state["cpu_rng"])
        return trainer


def same_state(left, right):
    """Exact equality for nested checkpoint tensors and scalar metadata."""
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same_state(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(same_state(a, b) for a, b in zip(left, right))
    return left == right


def compare_resume(path, steps=40, split_step=17, seed=7):
    if not 0 < split_step < steps:
        raise ValueError("Require 0 < split_step < steps")
    torch.manual_seed(seed)
    full = Trainer()
    validation_before = full.validate()
    full_history = full.train(steps)
    expected = full.state_dict()
    validation_after = full.validate()
    torch.manual_seed(seed)
    interrupted = Trainer()
    prefix = interrupted.train(split_step)
    interrupted.save(path)
    torch.rand(19)  # Simulate unrelated work between interruption and loading.
    resumed = Trainer.load(path)
    suffix = resumed.train(steps - split_step)
    actual = resumed.state_dict()
    checks = {key: same_state(expected[key], actual[key]) for key in expected}
    checks["history"] = full_history == prefix + suffix
    checks["validation"] = validation_after == resumed.validate()
    if not all(checks.values()):
        raise AssertionError(f"Resume mismatch: {checks}")
    return {"checks": checks, "initial_train_loss": full_history[0]["loss"],
            "final_train_loss": full_history[-1]["loss"], "validation_before": validation_before,
            "validation_after": validation_after, "first_lr": full_history[0]["lr"],
            "last_lr": full_history[-1]["lr"], "first_grad_norm": full_history[0]["grad_norm_before_clip"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="New output file; refuses overwrite")
    parser.add_argument("--steps", type=int, default=40, help="Total comparison steps or additional resume steps")
    parser.add_argument("--split-step", type=int, default=17)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume", type=Path, help="Trusted local checkpoint to continue")
    args = parser.parse_args()
    if args.checkpoint.exists() or args.checkpoint.is_symlink():
        parser.error("Checkpoint already exists; choose a new output path")
    if args.steps < 1:
        parser.error("steps must be positive")
    if args.resume is None and not 0 < args.split_step < args.steps:
        parser.error("Require 0 < split-step < steps")
    torch.set_num_threads(1)
    if args.resume is not None:
        trainer = Trainer.load(args.resume)
        history = trainer.train(args.steps)
        trainer.save(args.checkpoint)
        print(f"step={trainer.step}; last train CE={history[-1]['loss']:.6f}; validation CE={trainer.validate():.6f}")
    else:
        result = compare_resume(args.checkpoint, args.steps, args.split_step, args.seed)
        print(f"input/target shape=(1, 6); logits shape=(1, 6, {len(VOCAB)})")
        print(f"train CE: {result['initial_train_loss']:.6f} -> {result['final_train_loss']:.6f}")
        print(f"held-out CE: {result['validation_before']:.6f} -> {result['validation_after']:.6f}")
        print(f"lr used: {result['first_lr']:.8f} -> {result['last_lr']:.8f}; first pre-clip norm={result['first_grad_norm']:.6f}")
        print(f"exact resume checks: {result['checks']}")
        print(f"saved checkpoint step={args.split_step}; compared final step={args.steps}")
    print("Teaching corpus only; validation loss is not evidence of arithmetic generalization.")


if __name__ == "__main__":
    main()
