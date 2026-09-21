"""生成 11.3 节插图：玩具调度器的 11 步轨迹（混排、分块 Prefill、抢占、重算）。

正文位置：11_serving/11.3_scheduler_loop.md
输出：11_serving/_images/ch11_3_schedule_trace.png

脚本内含一个最小调度器，规则与正文 11.3.2 的六步一致：先给运行中的请求分词元，
KV 不够就抢占运行队尾；本步没有发生抢占才从等待队列准入，且整条序列放得下才准入
（simulate 的 watermark 参数可再要求准入后留下若干空闲词元，正文用它做对照）。
KV 池按词元计数，不分块（块粒度见 11.4 节）。运行脚本会同时打印逐步轨迹，
正文的轨迹表由这份输出抄录。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _diagram import (ACCENT, DATA_EDGE, DATA_FACE, FS_SMALL, FS_TEXT, INK, MUTED, NEUTRAL_EDGE,
                      NEUTRAL_FACE, NEW_EDGE, NEW_FACE, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("11_serving", "ch11_3_schedule_trace.png")

# 名字: (到达的步, Prompt 词元数, 输出词元数)
REQUESTS = {"A": (1, 6, 8), "B": (1, 8, 5), "C": (2, 20, 8), "D": (4, 8, 5)}
BUDGET, POOL, MAX_SEQS, N_STEPS = 12, 48, 4, 11


def simulate(watermark=0):
    st = {k: dict(arr=v[0], prompt=v[1], out=v[2], done=0, computed=0, status="-")
          for k, v in REQUESTS.items()}
    waiting, running, log = [], [], []
    for step in range(1, N_STEPS + 1):
        for k, r in st.items():
            if r["arr"] == step:
                r["status"] = "WAITING"
                waiting.append(k)
        free = POOL - sum(r["computed"] for r in st.values() if r["status"] == "RUNNING")
        left, sched, preempted = BUDGET, {}, []
        i = 0
        while i < len(running) and left > 0:          # 先排运行中的请求
            k = running[i]
            r = st[k]
            need = min(r["prompt"] + r["done"] - r["computed"], left)
            blocked = False
            while free < need:                        # KV 不够：抢占运行队尾
                victim = running.pop()
                v = st[victim]
                free += v["computed"]
                if victim in sched:
                    left += sched.pop(victim)
                v["computed"], v["status"] = 0, "PREEMPTED"
                waiting.insert(0, victim)
                preempted.append(victim)
                if victim == k:
                    blocked = True
                    break
            if blocked:
                break
            sched[k] = need
            free -= need
            left -= need
            i += 1
        if not preempted:                             # 本步无抢占才准入
            while waiting and left > 0 and len(running) < MAX_SEQS:
                k = waiting[0]
                r = st[k]
                total = r["prompt"] + r["done"]
                if free - total < watermark or free < total:   # 整条放得下、且留足水位线才准入
                    break                                      # 队首不满足就停，不看后面的请求
                waiting.pop(0)
                need = min(total, left)
                r["status"] = "RUNNING"
                running.append(k)
                sched[k] = need
                free -= need
                left -= need
        rec = dict(step=step, sched=dict(sched), preempted=preempted, kind={}, first={},
                   finished=[], recompute={k: st[k].get("hit", False) for k in sched})
        for k in preempted:
            st[k]["hit"] = True
        for k, n in sched.items():
            r = st[k]
            rec["kind"][k] = "D" if (n == 1 and r["computed"] >= r["prompt"]) else "P"
            r["computed"] += n
            if r["computed"] == r["prompt"] + r["done"]:     # 追平：采样一个词元
                rec["first"][k] = r["done"] == 0
                r["done"] += 1
                if r["done"] == r["out"]:
                    r["status"], r["computed"] = "FINISHED", 0
                    running.remove(k)
                    rec["finished"].append(k)
        rec["status"] = {k: r["status"] for k, r in st.items()}
        rec["budget_used"] = BUDGET - left
        rec["pool_used"] = sum(r["computed"] for r in st.values() if r["status"] == "RUNNING")
        log.append(rec)
    return log


def main() -> None:
    use_cjk_font()
    log = simulate()
    for title, trace in (("无水位线（图中画的）", log), ("水位线 4 个词元（正文对照）", simulate(4))):
        print(title)
        for rec in trace:
            cells = " ".join(f"{k}:{rec['kind'][k]}{n}" for k, n in rec["sched"].items())
            print(f"  第 {rec['step']:>2} 步 | {cells:<22} | 预算 {rec['budget_used']:>2}/{BUDGET}"
                  f" | KV 池 {rec['pool_used']:>2}/{POOL} | 抢占 {rec['preempted']}"
                  f" | 完成 {rec['finished']}")

    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    cw, ch, x0 = 2.62, 1.9, 6.9
    names = list(REQUESTS)
    top = len(names) * ch

    for j in range(N_STEPS):
        label(ax, x0 + (j + 0.5) * cw, top + 0.5, f"{j + 1}", fontsize=FS_TEXT, color=MUTED)
    label(ax, x0 - 0.3, top + 0.5, "第几步", fontsize=FS_SMALL, color=MUTED, ha="right")
    for i, k in enumerate(names):
        y = top - (i + 1) * ch
        _, p, o = REQUESTS[k]
        label(ax, x0 - 0.3, y + ch / 2, f"{k}：Prompt {p}，输出 {o}", fontsize=FS_SMALL, ha="right")
        for j, rec in enumerate(log):
            x = x0 + j * cw
            face, edge, text, color = "white", "#d5d4ce", "", INK
            if k in rec["sched"]:
                n = rec["sched"][k]
                if rec["kind"][k] == "P":
                    face, edge, text = NEW_FACE, NEW_EDGE, f"P {n}"
                else:
                    face, edge, text = DATA_FACE, DATA_EDGE, "D 1"
                if rec["kind"][k] == "P" and rec["recompute"][k]:
                    text += "\n重算"
                elif rec["first"].get(k):
                    text += "\n首词元"
                if k in rec["finished"]:
                    text += "\n完成"
            elif k in rec["preempted"]:
                face, edge, text, color = "white", ACCENT, "被抢占\n释放 8", ACCENT
            elif rec["status"][k] in ("WAITING", "PREEMPTED"):
                face, edge, text, color = NEUTRAL_FACE, NEUTRAL_EDGE, "等待", MUTED
            ax.add_patch(Rectangle((x + 0.06, y + 0.06), cw - 0.12, ch - 0.12, facecolor=face,
                                   edgecolor=edge, linewidth=1.8 if edge == ACCENT else 1.1))
            if text:
                label(ax, x + cw / 2, y + ch / 2, text, fontsize=FS_SMALL - 1.5, color=color,
                      linespacing=1.25)

    yb = -1.1
    label(ax, x0 - 0.3, yb, f"本步词元数 / 预算 {BUDGET}", fontsize=FS_SMALL, ha="right")
    label(ax, x0 - 0.3, yb - 1.3, f"步末 KV 占用 / 池 {POOL}", fontsize=FS_SMALL, ha="right")
    for j, rec in enumerate(log):
        x = x0 + (j + 0.5) * cw
        full = rec["budget_used"] == BUDGET
        label(ax, x, yb, f"{rec['budget_used']}", fontsize=FS_TEXT, bold=full,
              color=INK if full else MUTED)
        tight = rec["pool_used"] >= POOL - 1
        label(ax, x, yb - 1.3, f"{rec['pool_used']}", fontsize=FS_TEXT, bold=tight,
              color=ACCENT if tight else MUTED)

    # 图例
    yl = -4.3
    items = [(NEW_FACE, NEW_EDGE, "P n：Prefill 块，本步算 n 个词元"),
             (DATA_FACE, DATA_EDGE, "D 1：Decode，本步算 1 个词元"),
             (NEUTRAL_FACE, NEUTRAL_EDGE, "在等待队列里"),
             ("white", ACCENT, "本步被抢占")]
    for x, (face, edge, text) in zip((0.3, 12.7, 24.3, 30.3), items):
        ax.add_patch(Rectangle((x, yl - 0.35), 1.0, 0.7, facecolor=face, edgecolor=edge, linewidth=1.2))
        label(ax, x + 1.3, yl, text, fontsize=FS_SMALL - 1, ha="left", color=MUTED)

    finish(fig, ax, OUTPUT, xlim=(0, x0 + N_STEPS * cw + 0.3), ylim=(yl - 0.9, top + 1.1))


if __name__ == "__main__":
    main()
