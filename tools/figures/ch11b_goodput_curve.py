"""图：吞吐还在涨，Goodput 已经到头（11.13.2）。

模型：一个 Prefill 实例视作 M/D/1 队列（泊松到达、定长服务、先到先服务、不合批），
单请求 Prefill 时间 D = 0.367 s（11.9.4 的算例：Llama 3 70B、4000 词元输入、4 张 H100）。
对每个到达率用 Lindley 递推仿真，3 个随机种子各 20 万个请求，P90 与 “TTFT <= 1 s” 的达标率取种子间平均。
Goodput = 达标率仍不低于 90% 的最大到达率；饱和吞吐 = 1/D。

这是排队模型，不是实测；要表达的是两条线之间的那段差距。
"""

from __future__ import annotations

import random

import matplotlib.pyplot as plt

from _diagram import ACCENT, DATA_EDGE, INK, MUTED, WEIGHT_EDGE
from _style import image_path, use_cjk_font

OUTPUT = image_path("11_serving", "ch11b_goodput_curve.png")

D = 0.367
SLO = 1.0
TARGET = 0.90
N_REQ = 200_000
SEEDS = (7, 11, 23)


def simulate(lam: float, seed: int) -> list[float]:
    rnd = random.Random(seed)
    wait, out = 0.0, []
    for _ in range(N_REQ):
        out.append(wait + D)
        wait = max(0.0, wait + D - rnd.expovariate(lam))
    out.sort()
    return out


def attainment(ttft: list[float]) -> float:
    return sum(1 for x in ttft if x <= SLO) / len(ttft)


def stats(lam: float) -> tuple[float, float]:
    """(达标率, P90)，各为 SEEDS 个仿真的平均。"""
    runs = [simulate(lam, s) for s in SEEDS]
    att = sum(attainment(t) for t in runs) / len(runs)
    p90 = sum(t[int(0.9 * len(t))] for t in runs) / len(runs)
    return att, p90


def main() -> None:
    use_cjk_font()
    rates = [0.2 + 0.1 * k for k in range(25)]          # 0.2 … 2.6
    att, p90 = [], []
    for lam in rates:
        a, p = stats(lam)
        att.append(a * 100)
        p90.append(p)

    lo, hi = 1.0, 1.6                                     # 二分找 Goodput
    for _ in range(12):
        mid = (lo + hi) / 2
        if stats(mid)[0] >= TARGET:
            lo = mid
        else:
            hi = mid
    goodput, sat = lo, 1 / D
    print(f"Goodput = {goodput:.2f} 请求/秒，饱和吞吐 = {sat:.2f}，比值 {goodput / sat:.2f}")

    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    ax.plot(rates, att, color=DATA_EDGE, lw=2.3)
    ax.axhline(TARGET * 100, color=MUTED, lw=1.0, ls="--")
    ax.text(0.22, TARGET * 100 - 5.5, "达标率下限 90%", fontsize=10.5, color=MUTED)
    ax.set_xlabel("到达率（请求/秒）；未饱和时它也就是吞吐", fontsize=11)
    ax.set_ylabel("TTFT ≤ 1 s 的达标率（%）", color=DATA_EDGE, fontsize=11)
    ax.tick_params(axis="y", colors=DATA_EDGE)
    ax.set_xlim(0.2, 3.2)
    ax.set_ylim(0, 105)

    ax2 = ax.twinx()
    ax2.plot(rates, p90, color=WEIGHT_EDGE, lw=2.3, ls="--")
    ax2.set_ylabel("TTFT 的 P90（秒）", color=WEIGHT_EDGE, fontsize=11)
    ax2.tick_params(axis="y", colors=WEIGHT_EDGE)
    ax2.set_ylim(0, 10.5)

    ax.axvline(goodput, color=ACCENT, lw=1.4)
    ax.axvline(sat, color=INK, lw=1.4, ls=":")
    ax.axvspan(goodput, sat, color=ACCENT, alpha=0.07)
    ax.text(goodput - 0.04, 40, f"Goodput\n{goodput:.2f} 请求/秒", color=ACCENT, fontsize=11, ha="right")
    ax.text(sat + 0.04, 62, f"饱和吞吐\n1/D = {sat:.2f}", color=INK, fontsize=11, ha="left")
    ax.text(1.86, 30, "吞吐还能涨，\n但已不满足 SLO", color=ACCENT, fontsize=10.5, ha="center")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150, facecolor="white")
    print(f"已写入 {OUTPUT}")


if __name__ == "__main__":
    main()
