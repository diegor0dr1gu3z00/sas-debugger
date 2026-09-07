# VENDORED_PRIOR_ART
# ==================
# Vendored from RegLLM ~/Documents/PwC/regllm (apdq/expr.py).
# Dependency-free SAS-expression recomputation oracle / SQL compiler.
# Attribution: RegLLM development team.
# Original module docstring follows.

"""Restricted formula language for binding manifests.

Formulas document how a *derived* field is computed from other fields of
the same row. The language is deliberately small so that every formula can
be (a) evaluated in Python to build the clean twin, and (b) compiled to
SQL to build the recomputation oracle. Supported:

* numeric and string literals, column references (bare identifiers)
* ``+ - * /``, unary ``-``, parentheses
* comparisons ``= != < <= > >=``, boolean ``and or not``
* functions ``min max abs sqrt round if`` (``if(cond, then, else)``)
* NULL handling: ``null()`` (the NULL literal), ``isnull(X)`` (true when
  X is missing — the only comparison that never propagates NULL)
* ``yyyymm(D)`` — the YYYYMM integer of an ISO date string, for
  date↔period consistency invariants

NULL semantics mirror SQL: arithmetic with a NULL input yields NULL, and
a NULL comparison is falsy — so recomputation oracles silently skip rows
with missing inputs (missing inputs are the completeness class's job);
``isnull``/``null`` exist precisely for *conditional* completeness rules
("a closed cycle must carry a closure date").
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


class ExprError(ValueError):
    """Raised on parse errors or unknown identifiers/functions."""


# ── tokenizer ────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(
    r"\s*(?:"
    r"(?P<num>\d+\.\d*|\.\d+|\d+)"
    r"|(?P<str>'[^']*')"
    r"|(?P<ident>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<op><=|>=|!=|<>|[-+*/()=<>,])"
    r")"
)

_KEYWORDS = {"and", "or", "not"}
_FUNCTIONS = {"min", "max", "abs", "sqrt", "round", "if",
              "isnull", "null", "yyyymm"}


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if not m or m.end() == pos:
            rest = text[pos:].strip()
            if not rest:
                break
            raise ExprError(f"cannot tokenize {rest[:20]!r} in formula {text!r}")
        pos = m.end()
        if m.lastgroup == "num":
            tokens.append(("num", m.group("num")))
        elif m.lastgroup == "str":
            tokens.append(("str", m.group("str")[1:-1]))
        elif m.lastgroup == "ident":
            word = m.group("ident")
            kind = "kw" if word.lower() in _KEYWORDS else "ident"
            tokens.append((kind, word.lower() if kind == "kw" else word))
        else:
            op = m.group("op")
            tokens.append(("op", "!=" if op == "<>" else op))
    return tokens


# ── AST ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Node:
    kind: str                     # num | str | col | binop | unop | func
    value: object = None          # literal value, column name, op, func name
    children: tuple["Node", ...] = ()


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]], text: str):
        self.toks = tokens
        self.i = 0
        self.text = text

    def peek(self) -> tuple[str, str] | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self) -> tuple[str, str]:
        tok = self.peek()
        if tok is None:
            raise ExprError(f"unexpected end of formula {self.text!r}")
        self.i += 1
        return tok

    def expect_op(self, op: str) -> None:
        tok = self.take()
        if tok != ("op", op):
            raise ExprError(f"expected {op!r}, got {tok} in {self.text!r}")

    # or_expr := and_expr (OR and_expr)*
    def parse_expr(self) -> Node:
        node = self.parse_and()
        while self.peek() == ("kw", "or"):
            self.take()
            node = Node("binop", "or", (node, self.parse_and()))
        return node

    def parse_and(self) -> Node:
        node = self.parse_not()
        while self.peek() == ("kw", "and"):
            self.take()
            node = Node("binop", "and", (node, self.parse_not()))
        return node

    def parse_not(self) -> Node:
        if self.peek() == ("kw", "not"):
            self.take()
            return Node("unop", "not", (self.parse_not(),))
        return self.parse_cmp()

    def parse_cmp(self) -> Node:
        node = self.parse_add()
        tok = self.peek()
        if tok and tok[0] == "op" and tok[1] in ("=", "!=", "<", "<=", ">", ">="):
            self.take()
            node = Node("binop", tok[1], (node, self.parse_add()))
        return node

    def parse_add(self) -> Node:
        node = self.parse_mul()
        while True:
            tok = self.peek()
            if tok and tok[0] == "op" and tok[1] in ("+", "-"):
                self.take()
                node = Node("binop", tok[1], (node, self.parse_mul()))
            else:
                return node

    def parse_mul(self) -> Node:
        node = self.parse_unary()
        while True:
            tok = self.peek()
            if tok and tok[0] == "op" and tok[1] in ("*", "/"):
                self.take()
                node = Node("binop", tok[1], (node, self.parse_unary()))
            else:
                return node

    def parse_unary(self) -> Node:
        if self.peek() == ("op", "-"):
            self.take()
            return Node("unop", "-", (self.parse_unary(),))
        return self.parse_primary()

    def parse_primary(self) -> Node:
        kind, val = self.take()
        if kind == "num":
            return Node("num", float(val) if "." in val else int(val))
        if kind == "str":
            return Node("str", val)
        if kind == "op" and val == "(":
            node = self.parse_expr()
            self.expect_op(")")
            return node
        if kind == "ident":
            if self.peek() == ("op", "("):
                fname = val.lower()
                if fname not in _FUNCTIONS:
                    raise ExprError(f"unknown function {val!r} in {self.text!r}")
                self.take()
                args: list[Node] = []
                if self.peek() != ("op", ")"):
                    args.append(self.parse_expr())
                    while self.peek() == ("op", ","):
                        self.take()
                        args.append(self.parse_expr())
                self.expect_op(")")
                _check_arity(fname, len(args), self.text)
                return Node("func", fname, tuple(args))
            return Node("col", val)
        raise ExprError(f"unexpected token {val!r} in {self.text!r}")


_ARITY = {"abs": (1, 1), "sqrt": (1, 1), "round": (1, 2),
          "min": (2, 99), "max": (2, 99), "if": (3, 3),
          "isnull": (1, 1), "null": (0, 0), "yyyymm": (1, 1)}


def _check_arity(fname: str, n: int, text: str) -> None:
    lo, hi = _ARITY[fname]
    if not lo <= n <= hi:
        raise ExprError(f"{fname}() takes {lo}..{hi} args, got {n} in {text!r}")


def parse(text: str) -> Node:
    tokens = _tokenize(text)
    if not tokens:
        raise ExprError("empty formula")
    parser = _Parser(tokens, text)
    node = parser.parse_expr()
    if parser.peek() is not None:
        raise ExprError(f"trailing tokens after formula in {text!r}")
    return node


def columns(node: Node) -> set[str]:
    """All column names referenced by the expression."""
    out: set[str] = set()
    if node.kind == "col":
        out.add(str(node.value))
    for child in node.children:
        out |= columns(child)
    return out


# ── Python evaluation (used by the twin generator) ───────────────────────────

def eval_row(node: Node, row: dict) -> object:
    k = node.kind
    if k == "num" or k == "str":
        return node.value
    if k == "col":
        name = str(node.value)
        if name not in row:
            raise ExprError(f"unknown column {name!r} referenced by formula")
        return row[name]
    if k == "unop":
        val = eval_row(node.children[0], row)
        if node.value == "-":
            return None if val is None else -val
        return (not _truthy(val))
    if k == "binop":
        op = str(node.value)
        if op == "and":
            return _truthy(eval_row(node.children[0], row)) and \
                _truthy(eval_row(node.children[1], row))
        if op == "or":
            return _truthy(eval_row(node.children[0], row)) or \
                _truthy(eval_row(node.children[1], row))
        a = eval_row(node.children[0], row)
        b = eval_row(node.children[1], row)
        if a is None or b is None:
            return None
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            return None if b == 0 else a / b
        if op == "=":
            return a == b
        if op == "!=":
            return a != b
        if op == "<":
            return a < b
        if op == "<=":
            return a <= b
        if op == ">":
            return a > b
        if op == ">=":
            return a >= b
    if k == "func":
        fname = str(node.value)
        if fname == "if":
            cond = eval_row(node.children[0], row)
            branch = node.children[1] if _truthy(cond) else node.children[2]
            return eval_row(branch, row)
        if fname == "null":
            return None
        if fname == "isnull":
            return eval_row(node.children[0], row) is None
        args = [eval_row(c, row) for c in node.children]
        if any(a is None for a in args):
            return None
        if fname == "yyyymm":
            text = str(args[0])
            try:
                return int(text[:4]) * 100 + int(text[5:7])
            except (ValueError, IndexError):
                return None
        if fname == "abs":
            return abs(args[0])
        if fname == "sqrt":
            return math.sqrt(args[0]) if args[0] >= 0 else None
        if fname == "round":
            return round(args[0], int(args[1])) if len(args) == 2 else round(args[0])
        if fname == "min":
            return min(args)
        if fname == "max":
            return max(args)
    raise ExprError(f"cannot evaluate node {node.kind}:{node.value}")


def _truthy(val: object) -> bool:
    return bool(val) and val is not None


# ── SQL compilation (used by oracle generators) ──────────────────────────────

def to_sql(node: Node, qualifier: str = "") -> str:
    """Compile to SQLite SQL. ``qualifier`` prefixes column refs (``t.``)."""
    k = node.kind
    if k == "num":
        return repr(node.value)
    if k == "str":
        return "'" + str(node.value).replace("'", "''") + "'"
    if k == "col":
        return f"{qualifier}{node.value}"
    if k == "unop":
        inner = to_sql(node.children[0], qualifier)
        return f"(-{inner})" if node.value == "-" else f"(NOT {inner})"
    if k == "binop":
        op = {"=": "=", "!=": "<>", "and": "AND", "or": "OR"}.get(
            str(node.value), str(node.value))
        a = to_sql(node.children[0], qualifier)
        b = to_sql(node.children[1], qualifier)
        return f"({a} {op} {b})"
    if k == "func":
        fname = str(node.value)
        if fname == "if":
            c, t, e = (to_sql(ch, qualifier) for ch in node.children)
            return f"(CASE WHEN {c} THEN {t} ELSE {e} END)"
        if fname == "null":
            return "NULL"
        if fname == "isnull":
            return f"({to_sql(node.children[0], qualifier)} IS NULL)"
        if fname == "yyyymm":
            return (f"CAST(STRFTIME('%Y%m', "
                    f"{to_sql(node.children[0], qualifier)}) AS INTEGER)")
        args = ", ".join(to_sql(ch, qualifier) for ch in node.children)
        sql_name = {"sqrt": "APDQ_SQRT"}.get(fname, fname.upper())
        return f"{sql_name}({args})"
    raise ExprError(f"cannot compile node {node.kind}:{node.value}")


def register_sql_functions(conn) -> None:
    """Register UDFs the compiled SQL may need (sqrt is not guaranteed
    to exist in every SQLite build, so we ship our own)."""
    conn.create_function(
        "APDQ_SQRT", 1,
        lambda x: math.sqrt(x) if x is not None and x >= 0 else None)
