"""Cross-module name resolution for the code_ui/ package.

Regression test for the split-migration bug class where a helper moved
modules but its import didn't (e.g. ``_sessions`` → NameError at runtime).
Python only fails on these when the line executes, so import smoke alone
is not enough — this AST check fails fast in CI instead.
"""

import ast
import builtins
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent / "code_ui"
SKIP = {"__init__.py"}


class Scope:
    def __init__(self, parent=None):
        self.parent = parent
        self.names = set()

    def resolve(self, name):
        s = self
        while s is not None:
            if name in s.names:
                return True
            s = s.parent
        return False


def _args(args, scope):
    for a in list(args.args) + list(args.kwonlyargs):
        scope.names.add(a.arg)
    if args.vararg:
        scope.names.add(args.vararg.arg)
    if args.kwarg:
        scope.names.add(args.kwarg.arg)


def _visit(node, scope, problems, fname):
    if isinstance(node, ast.Name):
        if isinstance(node.ctx, ast.Store):
            scope.names.add(node.id)
        elif isinstance(node.ctx, ast.Load) and not scope.resolve(node.id):
            if node.id not in dir(builtins):
                problems.append(f"{fname}:{node.lineno} undefined {node.id!r}")
        return
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope.names.add(child.name)
            fn = Scope(scope)
            _args(child.args, fn)
            _visit(child, fn, problems, fname)
        elif isinstance(child, ast.ClassDef):
            scope.names.add(child.name)
            _visit(child, Scope(scope), problems, fname)
        elif isinstance(child, ast.Import):
            for a in child.names:
                scope.names.add((a.asname or a.name).split(".")[0])
        elif isinstance(child, ast.ImportFrom):
            for a in child.names:
                scope.names.add(a.asname or a.name)
        elif isinstance(child, ast.NamedExpr):
            scope.names.add(child.target.id)
            _visit(child.value, scope, problems, fname)
        elif isinstance(child, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            comp = Scope(scope)
            for gen in child.generators:
                _visit(gen.iter, scope, problems, fname)
                _visit(gen.target, comp, problems, fname)
                for cond in gen.ifs:
                    _visit(cond, comp, problems, fname)
            if isinstance(child, ast.DictComp):
                _visit(child.key, comp, problems, fname)
                _visit(child.value, comp, problems, fname)
            else:
                _visit(child.elt, comp, problems, fname)
        elif isinstance(child, ast.ExceptHandler):
            if child.name:
                sub = Scope(scope)
                sub.names.add(child.name)
                for n in child.body:
                    _visit(n, sub, problems, fname)
            else:
                for n in child.body:
                    _visit(n, scope, problems, fname)
        elif isinstance(child, (ast.For, ast.AsyncFor)):
            _visit(child.iter, scope, problems, fname)
            _visit(child.target, scope, problems, fname)
            for n in child.body + child.orelse:
                _visit(n, scope, problems, fname)
        elif isinstance(child, (ast.With, ast.AsyncWith)):
            for item in child.items:
                _visit(item.context_expr, scope, problems, fname)
                if item.optional_vars:
                    _visit(item.optional_vars, scope, problems, fname)
            for n in child.body:
                _visit(n, scope, problems, fname)
        elif isinstance(child, ast.Lambda):
            fn = Scope(scope)
            _args(child.args, fn)
            _visit(child.body, fn, problems, fname)
        else:
            _visit(child, scope, problems, fname)


def _preregister(tree, scope):
    for child in ast.iter_child_nodes(tree):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope.names.add(child.name)
        elif isinstance(child, ast.Import):
            for a in child.names:
                scope.names.add((a.asname or a.name).split(".")[0])
        elif isinstance(child, ast.ImportFrom):
            for a in child.names:
                scope.names.add(a.asname or a.name)
        elif isinstance(child, ast.Assign):
            for t in child.targets:
                if isinstance(t, ast.Name):
                    scope.names.add(t.id)


def test_all_names_resolve():
    problems = []
    for path in sorted(PKG.glob("*.py")):
        if path.name in SKIP:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        scope = Scope()
        _preregister(tree, scope)
        _visit(tree, scope, problems, path.name)
    problems = [p for p in problems if "Any" not in p]
    assert problems == [], "\n".join(sorted(set(problems)))
