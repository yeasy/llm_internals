from __future__ import annotations

import re
import unittest
from collections import defaultdict, deque
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
ENDPOINTS = {"Beginner", "Algorithm", "Systems", "Research"}


def route_mermaid(text: str) -> str:
    match = re.search(
        r"^## 学习路线图\s*$.*?^```mermaid\s*$\n(.*?)^```\s*$",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise AssertionError("missing learning-route Mermaid graph")
    return match.group(1)


def route_table(text: str) -> list[dict[str, str]]:
    match = re.search(
        r"(\| 读者角色 \| 前置路线 \| 推导验收 \| 计算验收 \| 代码验收 \|\n(?:\|.*\n)+)",
        text,
    )
    if match is None:
        raise AssertionError("missing role acceptance table")
    lines = match.group(1).splitlines()
    cells = [
        [cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines
    ]
    return [dict(zip(cells[0], row)) for row in cells[2:] if len(row) == len(cells[0])]


class LearningRouteTests(unittest.TestCase):
    def setUp(self):
        self.text = README.read_text(encoding="utf-8")

    def test_route_is_reachable_acyclic_prerequisite_dag(self):
        graph = route_mermaid(self.text)
        edges = re.findall(
            r"(?m)^\s*([A-Za-z][A-Za-z0-9_]*)[^\n]*?-->\s*([A-Za-z][A-Za-z0-9_]*)",
            graph,
        )
        self.assertGreaterEqual(len(edges), 14)
        adjacency: dict[str, set[str]] = defaultdict(set)
        indegree: dict[str, int] = defaultdict(int)
        nodes = {"Start"}
        for source, target in edges:
            if target not in adjacency[source]:
                adjacency[source].add(target)
                indegree[target] += 1
            nodes.update((source, target))
        queue = deque(node for node in nodes if indegree[node] == 0)
        visited = []
        while queue:
            node = queue.popleft()
            visited.append(node)
            for target in adjacency[node]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
        self.assertEqual(set(visited), nodes, "learning route must be acyclic")
        reachable = {"Start"}
        queue = deque(["Start"])
        while queue:
            for target in adjacency[queue.popleft()]:
                if target not in reachable:
                    reachable.add(target)
                    queue.append(target)
        self.assertTrue(ENDPOINTS <= reachable)
        self.assertTrue(all(not adjacency[endpoint] for endpoint in ENDPOINTS))
        for prerequisite in (
            ("Ch1", "Ch2"),
            ("Ch2", "Ch3"),
            ("Ch3", "Ch4"),
            ("Ch5", "Ch6"),
            ("Ch6", "Ch7"),
            ("Ch7", "Ch8"),
            ("Ch9", "Ch10"),
            ("Ch10", "Ch11"),
            ("Ch13", "Ch14"),
        ):
            self.assertIn(prerequisite[1], adjacency[prerequisite[0]])

    def test_every_role_has_valid_links_and_three_acceptance_gates(self):
        rows = route_table(self.text)
        self.assertEqual(len(rows), 4)
        expected_roles = {"AI 初学者", "算法工程师", "系统工程师", "研究人员"}
        self.assertEqual(
            {re.sub(r"\*", "", row["读者角色"]) for row in rows}, expected_roles
        )
        for row in rows:
            route_links = re.findall(
                r"\[[^]]+\]\(([^)]+\.md(?:#[^)]+)?)\)", row["前置路线"]
            )
            self.assertGreaterEqual(len(route_links), 2, row["读者角色"])
            for column in ("推导验收", "计算验收", "代码验收"):
                self.assertNotIn(row[column].strip(), {"", "-", "无"})
                links = re.findall(r"\[[^]]+\]\(([^)]+\.md(?:#[^)]+)?)\)", row[column])
                self.assertGreaterEqual(len(links), 1, f"{row['读者角色']} {column}")
                route_links.extend(links)
            for target in route_links:
                relative = unquote(target.split("#", 1)[0])
                self.assertTrue((ROOT / relative).is_file(), target)

    def test_chapter_four_is_direct_prerequisite_for_chapter_five(self):
        graph = route_mermaid(self.text)
        edges = set(
            re.findall(
                r"(?m)^\s*([A-Za-z][A-Za-z0-9_]*)[^\n]*?-->\s*"
                r"([A-Za-z][A-Za-z0-9_]*)",
                graph,
            )
        )
        self.assertIn(("Ch4", "Ch5"), edges)
        self.assertNotIn(("Ch3", "Ch5"), edges)

    def test_every_role_endpoint_requires_chapter_four(self):
        graph = route_mermaid(self.text)
        adjacency: dict[str, set[str]] = defaultdict(set)
        for source, target in re.findall(
            r"(?m)^\s*([A-Za-z][A-Za-z0-9_]*)[^\n]*?-->\s*"
            r"([A-Za-z][A-Za-z0-9_]*)",
            graph,
        ):
            adjacency[source].add(target)

        reachable_without_chapter_four = {"Start"}
        queue = deque(["Start"])
        while queue:
            for target in adjacency[queue.popleft()]:
                if target == "Ch4" or target in reachable_without_chapter_four:
                    continue
                reachable_without_chapter_four.add(target)
                queue.append(target)

        self.assertTrue(
            ENDPOINTS.isdisjoint(reachable_without_chapter_four),
            "every role requires chapters 1-4, so no endpoint may bypass Ch4",
        )


if __name__ == "__main__":
    unittest.main()
