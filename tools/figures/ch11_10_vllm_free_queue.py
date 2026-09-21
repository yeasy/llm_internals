"""生成 11.10 节插图：vLLM 的空闲块队列怎样同时充当分配器与前缀缓存的 LRU。

正文位置：11_serving/11.10_vllm_internals.md
输出：11_serving/_images/ch11_10_vllm_free_queue.png

下面的 Pool 用普通列表复刻 vLLM v0.29.0 中 BlockPool 与 FreeKVCacheBlockQueue 的四条规则
（vllm/v1/core/block_pool.py、kv_cache_utils.py），图中每一行的队列内容都由它算出：

1. 分配从队首取块（get_new_blocks）；取到带哈希的块，先把它的哈希从表里删掉（淘汰）。
2. 请求结束时按逆序释放（尾块在前）；引用计数归零的块，无哈希的插回队首，有哈希的追加到队尾。
3. 前缀命中时 touch：引用计数加 1，若块还在空闲队列里，就从队列中间摘走。
4. 0 号块是占位的 null_block，启动时就从队列里取走，永不分配。

教学设定：块大小 16 个词元，池子共 9 块。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import ACCENT, FS_NAME, FS_SMALL, FS_TEXT, MUTED, arrow, box, label
from _style import image_path, use_cjk_font

OUTPUT = image_path("11_serving", "ch11_10_vllm_free_queue.png")

BLOCK = 16


class Pool:
    def __init__(self, n):
        self.free = list(range(1, n))      # 0 号是 null_block
        self.ref = {b: 0 for b in range(n)}
        self.hash_of = {}                  # 块号 -> 哈希名
        self.table = {}                    # 哈希名 -> 块号
        self.evicted = []

    def get_new_blocks(self, k):
        out, self.free = self.free[:k], self.free[k:]
        for b in out:
            if b in self.hash_of:          # 规则 1：复用带哈希的块 = 淘汰
                h = self.hash_of.pop(b)
                del self.table[h]
                self.evicted.append((b, h))
            self.ref[b] += 1
        return out

    def cache_full_blocks(self, blocks, names):
        for b, h in zip(blocks, names):
            self.hash_of[b], self.table[h] = h, b

    def touch(self, blocks):
        for b in blocks:                   # 规则 3
            if self.ref[b] == 0:
                self.free.remove(b)
            self.ref[b] += 1

    def free_blocks(self, blocks):
        first, last = [], []
        for b in reversed(blocks):         # 规则 2
            self.ref[b] -= 1
            if self.ref[b] == 0:
                (last if b in self.hash_of else first).append(b)
        self.free = first + self.free + last


def simulate():
    rows = []
    p = Pool(9)
    rows.append(("启动", "0 号块留作占位，\n其余按块号排队", list(p.free), {}, dict(p.table), []))

    a = p.get_new_blocks(3)                # 请求 A：40 个词元 → 3 块，前 2 块写满
    p.cache_full_blocks(a[:2], ["h1", "h2"])
    rows.append(("请求 A 进入", "40 个词元：从队首取 3 块，\n写满的 2 块登记哈希",
                 list(p.free), {"A": a}, dict(p.table), []))

    p.free_blocks(a)
    rows.append(("请求 A 结束", "逆序释放：无哈希的 3 号\n回队首，2、1 号进队尾",
                 list(p.free), {}, dict(p.table), []))
    snapshot = (list(p.free), dict(p.ref), dict(p.hash_of), dict(p.table))

    hit = [p.table["h1"], p.table["h2"]]   # 请求 B：前 32 个词元与 A 相同，共 42 个
    p.touch(hit)
    b = hit + p.get_new_blocks(1)
    rows.append(("接着来请求 B", "前 32 个词元与 A 相同：\n1、2 号从队列中间摘走",
                 list(p.free), {"B": b}, dict(p.table), []))

    p.free, p.ref, p.hash_of, p.table = snapshot   # 回到“请求 A 结束”之后，换一种走向
    p.evicted = []
    c = p.get_new_blocks(7)                # 请求 C：100 个词元，无公共前缀 → 7 块，前 6 块写满
    p.cache_full_blocks(c[:6], [f"h{k}" for k in range(3, 9)])
    rows.append(("或者来请求 C", "100 个词元、无公共前缀：取 7 块，\n淘汰 h2；写满的 6 块登记 h3-h8",
                 list(p.free), {"C": c}, dict(p.table), list(p.evicted)))
    return rows, p.hash_of


def hash_lines(table):
    """把哈希表压成几行文字；哈希号与块号都连续的一段（3 个以上）合并成一行，如 h3-h8 → 3-8 号。"""
    items = sorted(((int(h[1:]), b) for h, b in table.items()))
    runs, cur = [], []
    for h, b in items:
        if cur and (h, b) == (cur[-1][0] + 1, cur[-1][1] + 1):
            cur.append((h, b))
        else:
            cur = [(h, b)]
            runs.append(cur)
    out = []
    for run in runs:
        (h0, b0), (h1, b1) = run[0], run[-1]
        if len(run) >= 3:
            out.append(f"h{h0}-h{h1} → {b0}-{b1} 号")
        else:
            out.extend(f"h{h} → {b} 号" for h, b in run)
    return out


def main():
    use_cjk_font()
    rows, _ = simulate()
    fig = plt.figure(figsize=(10.0, 7.54))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])

    cw, ch = 1.05, 1.05
    x_q, x_use, x_tab = 7.3, 16.7, 22.75
    top = 17.2
    label(ax, 0.4, top + 1.15, "时刻", ha="left", fontsize=FS_NAME, bold=True)
    label(ax, x_q, top + 1.15, "空闲队列（左端是队首，先被取走）", ha="left", fontsize=FS_NAME, bold=True)
    label(ax, x_use, top + 1.15, "请求占用的块", ha="left", fontsize=FS_NAME, bold=True)
    label(ax, x_tab, top + 1.15, "哈希表", ha="left", fontsize=FS_NAME, bold=True)

    hashed_at = [set(), {1, 2}, {1, 2}, {1, 2}, {1}]
    for i, (title, note, free, used, table, evicted) in enumerate(rows):
        y = top - i * 3.45
        label(ax, 0.4, y - 0.3, title, ha="left", fontsize=FS_TEXT, bold=True)
        label(ax, 0.4, y - 1.5, note, ha="left", fontsize=FS_SMALL, color=MUTED)
        for j, blk in enumerate(free):
            kind = "data" if blk in hashed_at[i] else "neutral"
            box(ax, x_q + j * (cw + 0.08), y - ch - 0.35, cw, ch, str(blk), kind,
                fontsize=FS_TEXT, rounded=False)
        if not free:
            label(ax, x_q, y - 0.9, "（空）", ha="left", fontsize=FS_SMALL, color=MUTED)
        for name, blks in used.items():
            label(ax, x_use, y - 0.88, name, ha="left", fontsize=FS_TEXT, bold=True)
            for j, blk in enumerate(blks):
                hit = name == "B" and j < 2
                box(ax, x_use + 0.75 + j * (cw * 0.62 + 0.06), y - ch - 0.35, cw * 0.62, ch, str(blk),
                    "data" if hit else "new", fontsize=FS_SMALL, rounded=False)
        text = "\n".join(hash_lines(table)) or "空"
        label(ax, x_tab, y - 0.88, text, ha="left", fontsize=FS_SMALL)
        if evicted:
            label(ax, x_tab, y - 1.7, "h2 已删", ha="left", fontsize=FS_SMALL, color=ACCENT)
        if i < len(rows) - 1:
            ax.plot([0.3, 25.7], [y - 2.75, y - 2.75], color="#d9d8d3", lw=0.8, zorder=0)

    # 图例
    yl = -0.1
    box(ax, 0.4, yl, 0.8, 0.8, "", "neutral", rounded=False)
    label(ax, 1.4, yl + 0.4, "空闲、无哈希", ha="left", fontsize=FS_SMALL)
    box(ax, 5.6, yl, 0.8, 0.8, "", "data", rounded=False)
    label(ax, 6.6, yl + 0.4, "带哈希：可被命中；留在队列里时也可被淘汰", ha="left", fontsize=FS_SMALL)
    box(ax, 17.6, yl, 0.8, 0.8, "", "new", rounded=False)
    label(ax, 18.6, yl + 0.4, "本次新分配", ha="left", fontsize=FS_SMALL)
    arrow(ax, (x_q - 0.1, top + 0.45), (x_q + 1.6, top + 0.45), color=MUTED, lw=1.0)

    ax.set_xlim(0, 26)
    ax.set_ylim(-0.6, 19.0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(OUTPUT, dpi=150, facecolor="white")
    print(f"已写入 {OUTPUT}")
    for r in rows:
        print(r[0], "| 空闲队列", r[2], "| 占用", r[3], "| 哈希表", r[4], "| 淘汰", r[5])


if __name__ == "__main__":
    main()
