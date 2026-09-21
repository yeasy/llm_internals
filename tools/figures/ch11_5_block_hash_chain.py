"""生成 11.5 节插图：按块链式哈希怎样匹配前缀。

正文位置：11_serving/11.5_prefix_reuse.md
输出：11_serving/_images/ch11_5_block_hash_chain.png

教学设定：块大小 4 个词元。请求 A 有 14 个词元，前 3 块写满并进入缓存；请求 B 与 A
共享前 10 个词元。B 的第 3 块里只有 2 个词元相同，整块的哈希不同，所以只命中 8 个词元。
图中的哈希值取 SHA-256 的前 4 位十六进制，只为显示“相同输入得到相同的键”。
"""

from __future__ import annotations

import hashlib

import matplotlib.pyplot as plt

from _diagram import (ACCENT, FS_NAME, FS_SMALL, FS_TEXT, FS_TITLE, MUTED,
                      arrow, box, finish, grid, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("11_serving", "ch11_5_block_hash_chain.png")

BLOCK = 4
REQ_A = list("ABCDEFGHIJKLMN")
REQ_B = list("ABCDEFGHIJ") + list("wxyz")
CW, CH, GAP = 1.0, 0.95, 0.7   # 词元格的宽、高，块与块之间的空隙
X0 = 6.2                       # 第 0 块的左边界


def chain(tokens):
    """返回每个写满的块的键（前 4 位十六进制）。"""
    parent, keys = b"seed", []
    for i in range(0, len(tokens) // BLOCK * BLOCK, BLOCK):
        digest = hashlib.sha256(repr((parent, tuple(tokens[i:i + BLOCK]))).encode()).digest()
        keys.append(digest.hex()[:4])
        parent = digest
    return keys


def block_x(b):
    return X0 + b * (BLOCK * CW + GAP)


def draw_request(ax, top, tokens, kinds, block_ids):
    """画一行词元，按块分组；kinds[b] 是第 b 块的配色。"""
    n_blocks = (len(tokens) + BLOCK - 1) // BLOCK
    for b in range(n_blocks):
        cells = tokens[b * BLOCK:(b + 1) * BLOCK]
        grid(ax, block_x(b), top, [cells], kind=kinds[b], cell_w=CW, cell_h=CH, fontsize=FS_TEXT)
        label(ax, block_x(b) + len(cells) * CW / 2, top + 0.45, block_ids[b],
              fontsize=FS_SMALL, color=MUTED)


def main():
    use_cjk_font()
    keys_a, keys_b = chain(REQ_A), chain(REQ_B)
    fig, ax = plt.subplots(figsize=(12.4, 8.2))

    # ---------- 请求 A ----------
    top_a = 15.0
    label(ax, 0.0, top_a + 1.5, "请求 A：14 个词元，已经算完", ha="left",
          fontsize=FS_TITLE, bold=True)
    label(ax, 0.0, top_a - CH / 2, "词元，每块 4 个", ha="left", fontsize=FS_SMALL, color=MUTED)
    draw_request(ax, top_a, REQ_A, ["data", "data", "data", "neutral"],
                 ["物理块 0", "物理块 1", "物理块 2", "物理块 3（未满）"])
    label(ax, 0.0, top_a - 2.35, "块的键", ha="left", fontsize=FS_SMALL, color=MUTED)
    parents = ["种子", keys_a[0], keys_a[1]]
    for b in range(3):
        cells = "".join(REQ_A[b * BLOCK:(b + 1) * BLOCK])
        box(ax, block_x(b), top_a - 3.0, BLOCK * CW, 1.3,
            f"H({parents[b]}, {cells})\n= {keys_a[b]}", kind="data", fontsize=FS_SMALL)
        arrow(ax, (block_x(b) + 2 * CW, top_a - CH), (block_x(b) + 2 * CW, top_a - 1.7), color=MUTED)
        if b:
            arrow(ax, (block_x(b - 1) + BLOCK * CW, top_a - 2.35), (block_x(b), top_a - 2.35),
                  color=ACCENT, lw=1.8)
    label(ax, block_x(3) + CW, top_a - 2.35, "没写满，\n不算键、不共享",
          fontsize=FS_SMALL, color=MUTED)
    label(ax, block_x(1) - GAP / 2, top_a - 3.75, "父块的键进入子块的键（紫色箭头）",
          fontsize=FS_SMALL, color=ACCENT)

    # ---------- 哈希表 ----------
    ty = 9.0
    label(ax, 0.0, ty + 0.65, "缓存索引", ha="left", fontsize=FS_NAME, bold=True)
    label(ax, 0.0, ty - 0.1, "键 → 物理块", ha="left", fontsize=FS_SMALL, color=MUTED)
    for b in range(3):
        box(ax, block_x(b), ty - 0.35, BLOCK * CW, 1.3, f"{keys_a[b]} → 块 {b}",
            kind="neutral", fontsize=FS_TEXT)

    # ---------- 请求 B ----------
    top_b = 3.4
    label(ax, 0.0, top_b + 3.85, "请求 B：与 A 共享\n前 10 个词元", ha="left",
          fontsize=FS_TITLE, bold=True)
    label(ax, 0.0, top_b - CH / 2, "词元", ha="left", fontsize=FS_SMALL, color=MUTED)
    draw_request(ax, top_b, REQ_B, ["data", "data", "new", "new"],
                 ["复用块 0", "复用块 1", "新分配块 7", "新分配块 8"])
    label(ax, 0.0, top_b + 2.35, "逐块算键、查表", ha="left", fontsize=FS_SMALL, color=MUTED)
    verdict = ["命中", "命中", "未命中"]
    for b in range(3):
        kind = "data" if keys_b[b] == keys_a[b] else "new"
        box(ax, block_x(b), top_b + 1.75, BLOCK * CW, 1.2, f"{keys_b[b]}：{verdict[b]}",
            kind=kind, fontsize=FS_SMALL)
        x = block_x(b) + 2 * CW
        if keys_b[b] == keys_a[b]:
            arrow(ax, (x, top_b + 2.95), (x, ty - 0.35), color=MUTED)
    label(ax, block_x(2) + 2 * CW, top_b + 3.55, "I、J 相同，w、x 不同：整块的键\n不同，查表到此停止",
          fontsize=FS_SMALL, color=ACCENT)

    # 下方：哪些词元要重算
    y = top_b - CH - 0.35
    ax.plot([block_x(0), block_x(1) + BLOCK * CW], [y, y], color=MUTED, lw=1.6)
    label(ax, (block_x(0) + block_x(1) + BLOCK * CW) / 2, y - 0.5, "命中 8 个词元，不再计算",
          fontsize=FS_SMALL)
    ax.plot([block_x(2), block_x(3) + 2 * CW], [y, y], color=ACCENT, lw=1.6)
    label(ax, (block_x(2) + block_x(3) + 2 * CW) / 2, y - 0.5,
          "Prefill 这 6 个词元（含共享的 I、J）", fontsize=FS_SMALL, color=ACCENT)
    label(ax, 0.0, y - 1.7, "B 的块表：[0, 1, 7, 8]", ha="left", fontsize=FS_TEXT)

    finish(fig, ax, OUTPUT, xlim=(-0.4, block_x(3) + 3.6), ylim=(y - 2.4, top_a + 2.3))


if __name__ == "__main__":
    main()
