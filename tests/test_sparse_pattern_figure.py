"""Check the sparse mask itself without requiring the plotting runtime."""
import ast
from pathlib import Path
import random
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SparsePatternTests(unittest.TestCase):
    def test_bigbird_global_queries_and_keys_are_visible(self):
        tree = ast.parse((ROOT / "tools/figures/ch12_sparse_patterns.py").read_text())
        names = {"N", "BLOCK", "RANDOM_BLOCKS", "SEED"}
        nodes = [node for node in tree.body if
                 (isinstance(node, ast.FunctionDef) and node.name == "build_bigbird") or
                 (isinstance(node, ast.Assign) and any(
                     isinstance(target, ast.Name) and target.id in names
                     for target in node.targets))]
        scope = {"random": random}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "figure-mask", "exec"), scope)
        visible = scope["build_bigbird"]()
        n, block = scope["N"], scope["BLOCK"]
        self.assertTrue(all(visible(i, j) for i in range(block) for j in range(n)))
        self.assertTrue(all(visible(i, j) for i in range(n) for j in range(block)))
        self.assertTrue(any(not visible(i, j) for i in range(block, n) for j in range(n)))
