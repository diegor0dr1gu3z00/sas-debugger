# VENDORED_PRIOR_ART
# ==================
# Vendored from RegLLM ~/Documents/PwC/regllm (src/sas_logic_tree.py).
# SAS AST + row-level simulator / lineage. Attribution: RegLLM development team.
# Original module docstring follows.

"""SAS logic tree — comprehensive parser and simulator for SAS DATA steps.

Parses SAS source code into an AST covering all common DATA step constructs,
then walks the tree with example variable values to trace row-level execution.

Supported SAS constructs
------------------------
- DATA statement (single/multiple output datasets, _NULL_)
- SET, MERGE (with IN=, WHERE=, FIRSTOBS=/OBS= dataset options)
- BY (group processing keys)
- WHERE (subsetting filter)
- IF/THEN/ELSE (inline, DO blocks, nested)
- DO loops: iterative (TO/BY), DO WHILE, DO UNTIL, bare DO groups
- SELECT/WHEN/OTHERWISE (value-based and condition-based)
- ARRAY (explicit size, *, _TEMPORARY_, initial values)
- RETAIN (with optional initial values)
- OUTPUT (explicit, optional target dataset)
- DELETE (exclude row from output)
- CALL routines (CALL SYMPUT, CALL MISSING, etc.)
- LINK/RETURN/STOP/GOTO (control flow markers)
- Macro language: %LET, %MACRO/%MEND, %IF/%THEN/%ELSE, macro calls
- PROC blocks (captured with raw text; SQL parsed for table name)
- Expressions: all SAS operators, functions, IN/BETWEEN/LIKE/IS MISSING

Unsupported (silently skipped or captured raw)
----------------------------------------------
- INFILE/INPUT/FILE/PUT (file I/O — too format-specific to simulate)
- WINDOW/DISPLAY (interactive screens)
- MODIFY/UPDATE statements
- %INCLUDE
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union


# ─────────────────────────────────────────────────────────────────────────────
# AST node types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AssignNode:
    var: str
    expr: str

    def display(self, indent: int = 0) -> str:
        return "  " * indent + f"{self.var} = {self.expr}"

    def to_dict(self) -> dict:
        return {"type": "assign", "var": self.var, "expr": self.expr}


@dataclass
class IfNode:
    condition: str
    then_branch: list["AnyNode"]
    else_branch: list["AnyNode"] = field(default_factory=list)

    def display(self, indent: int = 0) -> str:
        pad = "  " * indent
        lines = [pad + f"IF {self.condition}"]
        for n in self.then_branch:
            lines.append(n.display(indent + 1))
        if self.else_branch:
            lines.append(pad + "ELSE")
            for n in self.else_branch:
                lines.append(n.display(indent + 1))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "type": "if",
            "condition": self.condition,
            "then": [n.to_dict() for n in self.then_branch],
            "else": [n.to_dict() for n in self.else_branch],
        }


@dataclass
class FilterNode:
    condition: str

    def display(self, indent: int = 0) -> str:
        return "  " * indent + f"WHERE {self.condition}"

    def to_dict(self) -> dict:
        return {"type": "filter", "condition": self.condition}


@dataclass
class DataStepNode:
    output_dataset: str
    input_dataset: str
    body: list["AnyNode"]
    output_datasets: list[str] = field(default_factory=list)   # extra outputs
    merge_datasets: list[str] = field(default_factory=list)    # MERGE sources
    by_keys: list[str] = field(default_factory=list)           # BY keys
    datalines_data: list[dict] = field(default_factory=list)   # rows from DATALINES

    def display(self, indent: int = 0) -> str:
        pad = "  " * indent
        src = ", ".join(self.merge_datasets) if self.merge_datasets else self.input_dataset
        verb = "MERGE" if self.merge_datasets else "SET"
        lines = [pad + f"DATA {self.output_dataset}  ({verb} {src})"]
        for n in self.body:
            lines.append(n.display(indent + 1))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d: dict = {
            "type": "data_step",
            "output": self.output_dataset,
            "input": self.input_dataset,
            "body": [n.to_dict() for n in self.body],
        }
        if self.merge_datasets:
            d["merge_datasets"] = self.merge_datasets
        if self.by_keys:
            d["by_keys"] = self.by_keys
        if self.output_datasets:
            d["output_datasets"] = self.output_datasets
        if self.datalines_data:
            d["datalines_data"] = self.datalines_data[:3]  # sample for display
            d["datalines_count"] = len(self.datalines_data)
        return d


@dataclass
class ProcNode:
    kind: str
    data: str
    raw: str
    output_table: str = ""
    input_tables: list[str] = field(default_factory=list)
    select_fields: list[tuple[str, str]] = field(default_factory=list)

    def display(self, indent: int = 0) -> str:
        if self.output_table:
            return "  " * indent + f"PROC {self.kind}  CREATE TABLE {self.output_table}"
        return "  " * indent + f"PROC {self.kind}  DATA={self.data}"

    def to_dict(self) -> dict:
        d: dict = {"type": "proc", "kind": self.kind, "data": self.data, "raw": self.raw}
        if self.output_table:
            d["output_table"] = self.output_table
        if self.input_tables:
            d["input_tables"] = self.input_tables
        if self.select_fields:
            d["select_fields"] = [{"alias": a, "expr": e} for a, e in self.select_fields]
        return d


# ── New node types ────────────────────────────────────────────────────────────

@dataclass
class DoLoopNode:
    """DO loop — iterative, WHILE, or UNTIL."""
    var: str = ""              # loop variable (empty for WHILE/UNTIL)
    start: str = ""
    stop: str = ""
    by_step: str = "1"         # BY increment, default 1
    while_cond: str = ""       # DO WHILE(cond)
    until_cond: str = ""       # DO UNTIL(cond)
    body: list["AnyNode"] = field(default_factory=list)

    def display(self, indent: int = 0) -> str:
        pad = "  " * indent
        if self.var:
            hdr = f"DO {self.var} = {self.start} TO {self.stop}"
            if self.by_step != "1":
                hdr += f" BY {self.by_step}"
            if self.while_cond:
                hdr += f" WHILE ({self.while_cond})"
            if self.until_cond:
                hdr += f" UNTIL ({self.until_cond})"
        elif self.while_cond:
            hdr = f"DO WHILE ({self.while_cond})"
        elif self.until_cond:
            hdr = f"DO UNTIL ({self.until_cond})"
        else:
            hdr = "DO"
        lines = [pad + hdr]
        for n in self.body:
            lines.append(n.display(indent + 1))
        lines.append(pad + "END")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d: dict = {"type": "do_loop", "body": [n.to_dict() for n in self.body]}
        if self.var:
            d.update({"var": self.var, "start": self.start, "stop": self.stop,
                      "by_step": self.by_step})
        if self.while_cond:
            d["while_cond"] = self.while_cond
        if self.until_cond:
            d["until_cond"] = self.until_cond
        return d


@dataclass
class WhenNode:
    """Single WHEN branch inside a SELECT block."""
    values: list[str]          # values/conditions — empty means OTHERWISE
    body: list["AnyNode"]

    def to_dict(self) -> dict:
        return {
            "type": "when",
            "values": self.values,
            "body": [n.to_dict() for n in self.body],
        }


@dataclass
class SelectNode:
    """SELECT/WHEN/OTHERWISE block."""
    select_expr: str = ""      # expression in SELECT(expr); empty → condition-based
    whens: list[WhenNode] = field(default_factory=list)
    otherwise: list["AnyNode"] = field(default_factory=list)

    def display(self, indent: int = 0) -> str:
        pad = "  " * indent
        hdr = f"SELECT ({self.select_expr})" if self.select_expr else "SELECT"
        lines = [pad + hdr]
        for w in self.whens:
            vals = ", ".join(w.values)
            lines.append(pad + f"  WHEN ({vals})")
            for n in w.body:
                lines.append(n.display(indent + 2))
        if self.otherwise:
            lines.append(pad + "  OTHERWISE")
            for n in self.otherwise:
                lines.append(n.display(indent + 2))
        lines.append(pad + "END")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "type": "select",
            "select_expr": self.select_expr,
            "whens": [w.to_dict() for w in self.whens],
            "otherwise": [n.to_dict() for n in self.otherwise],
        }


@dataclass
class ArrayNode:
    """ARRAY declaration."""
    name: str
    dims: str                  # e.g. "3", "*", "{3,2}"
    vars: list[str]
    temporary: bool = False
    initial_values: list[str] = field(default_factory=list)

    def display(self, indent: int = 0) -> str:
        tmp = " _TEMPORARY_" if self.temporary else ""
        return "  " * indent + f"ARRAY {self.name}{{{self.dims}}}{tmp}  [{', '.join(self.vars)}]"

    def to_dict(self) -> dict:
        d: dict = {
            "type": "array",
            "name": self.name,
            "dims": self.dims,
            "vars": self.vars,
        }
        if self.temporary:
            d["temporary"] = True
        if self.initial_values:
            d["initial_values"] = self.initial_values
        return d


@dataclass
class RetainNode:
    """RETAIN statement — preserve variable values across iterations."""
    vars: list[str]
    initial: str = ""          # optional initial value (same for all vars)

    def display(self, indent: int = 0) -> str:
        init = f" {self.initial}" if self.initial else ""
        return "  " * indent + f"RETAIN {' '.join(self.vars)}{init}"

    def to_dict(self) -> dict:
        return {"type": "retain", "vars": self.vars, "initial": self.initial}


@dataclass
class OutputNode:
    """OUTPUT statement — write current row explicitly."""
    dataset: str = ""          # optional target dataset

    def display(self, indent: int = 0) -> str:
        ds = f" {self.dataset}" if self.dataset else ""
        return "  " * indent + f"OUTPUT{ds}"

    def to_dict(self) -> dict:
        return {"type": "output", "dataset": self.dataset}


@dataclass
class DeleteNode:
    """DELETE statement — exclude current row from all outputs."""

    def display(self, indent: int = 0) -> str:
        return "  " * indent + "DELETE"

    def to_dict(self) -> dict:
        return {"type": "delete"}


@dataclass
class CallNode:
    """CALL routine (CALL SYMPUT, CALL MISSING, CALL EXECUTE, etc.)."""
    routine: str
    args: str = ""

    def display(self, indent: int = 0) -> str:
        return "  " * indent + f"CALL {self.routine}({self.args})"

    def to_dict(self) -> dict:
        return {"type": "call", "routine": self.routine, "args": self.args}


@dataclass
class MergeNode:
    """MERGE statement inside a DATA step."""
    datasets: list[str]        # dataset names (options stripped)
    in_vars: list[str] = field(default_factory=list)  # IN= variable names

    def display(self, indent: int = 0) -> str:
        return "  " * indent + f"MERGE {' '.join(self.datasets)}"

    def to_dict(self) -> dict:
        return {"type": "merge", "datasets": self.datasets, "in_vars": self.in_vars}


@dataclass
class ByNode:
    """BY statement (group-processing keys in DATA step or PROC)."""
    keys: list[str]
    descending: list[bool] = field(default_factory=list)  # per-key descending flag

    def display(self, indent: int = 0) -> str:
        parts = []
        for i, k in enumerate(self.keys):
            prefix = "DESCENDING " if self.descending and i < len(self.descending) and self.descending[i] else ""
            parts.append(prefix + k)
        return "  " * indent + f"BY {' '.join(parts)}"

    def to_dict(self) -> dict:
        return {"type": "by", "keys": self.keys}


@dataclass
class MacroLetNode:
    """%LET macro variable assignment."""
    var: str
    value: str

    def display(self, indent: int = 0) -> str:
        return "  " * indent + f"%LET {self.var} = {self.value}"

    def to_dict(self) -> dict:
        return {"type": "macro_let", "var": self.var, "value": self.value}


@dataclass
class MacroDefNode:
    """%MACRO ... %MEND block."""
    name: str
    params: str = ""
    body: list["AnyNode"] = field(default_factory=list)

    def display(self, indent: int = 0) -> str:
        pad = "  " * indent
        p = f"({self.params})" if self.params else "()"
        lines = [pad + f"%MACRO {self.name}{p}"]
        for n in self.body:
            lines.append(n.display(indent + 1))
        lines.append(pad + f"%MEND {self.name}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "type": "macro_def",
            "name": self.name,
            "params": self.params,
            "body": [n.to_dict() for n in self.body],
        }


@dataclass
class MacroCallNode:
    """%macroname(...) call."""
    name: str
    args: str = ""

    def display(self, indent: int = 0) -> str:
        return "  " * indent + f"%{self.name}({self.args})"

    def to_dict(self) -> dict:
        return {"type": "macro_call", "name": self.name, "args": self.args}


@dataclass
class LinkNode:
    """LINK label — jump to label subroutine."""
    label: str

    def display(self, indent: int = 0) -> str:
        return "  " * indent + f"LINK {self.label}"

    def to_dict(self) -> dict:
        return {"type": "link", "label": self.label}


@dataclass
class GotoNode:
    """GOTO label — unconditional jump."""
    label: str

    def display(self, indent: int = 0) -> str:
        return "  " * indent + f"GOTO {self.label}"

    def to_dict(self) -> dict:
        return {"type": "goto", "label": self.label}


@dataclass
class ReturnNode:
    """RETURN — return from LINK subroutine or end DATA step iteration."""

    def display(self, indent: int = 0) -> str:
        return "  " * indent + "RETURN"

    def to_dict(self) -> dict:
        return {"type": "return"}


@dataclass
class StopNode:
    """STOP — halt DATA step processing."""

    def display(self, indent: int = 0) -> str:
        return "  " * indent + "STOP"

    def to_dict(self) -> dict:
        return {"type": "stop"}


AnyNode = Union[
    AssignNode, IfNode, FilterNode, DataStepNode, ProcNode,
    DoLoopNode, SelectNode, ArrayNode, RetainNode, OutputNode, DeleteNode,
    CallNode, MergeNode, ByNode,
    MacroLetNode, MacroDefNode, MacroCallNode,
    LinkNode, GotoNode, ReturnNode, StopNode,
]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation trace
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TraceStep:
    kind: str    # assign | if_taken | if_skipped | filter_pass | filter_block
                 # loop_iter | select_when | select_otherwise | select_no_match
                 # output | delete | call
    label: str
    var: str | None = None
    old_val: Any = None
    new_val: Any = None
    data_step: str = ""


@dataclass
class EvalTrace:
    initial: dict[str, Any]
    final: dict[str, Any]
    steps: list[TraceStep]
    row_passes_filter: bool = True
    filter_results: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["── Input ──────────────────────────────────────────"]
        for k, v in self.initial.items():
            lines.append(f"  {k} = {v!r}")

        lines.append("── Trace ──────────────────────────────────────────")
        cur_step = ""
        for s in self.steps:
            if s.data_step and s.data_step != cur_step:
                cur_step = s.data_step
                lines.append(f"  ┌─ {cur_step}")
            if s.kind == "assign":
                lines.append(f"  ASSIGN  {s.label}")
                if s.old_val != s.new_val:
                    lines.append(f"          {s.var}: {s.old_val!r} → {s.new_val!r}")
            elif s.kind == "if_taken":
                lines.append(f"  IF  ✓   {s.label}")
            elif s.kind == "if_skipped":
                lines.append(f"  IF  ✗   {s.label}")
            elif s.kind == "filter_pass":
                lines.append(f"  WHERE ✓ {s.label}")
            elif s.kind == "filter_block":
                lines.append(f"  WHERE ✗ {s.label}  ← row excluded from {cur_step}")
            elif s.kind == "loop_iter":
                lines.append(f"  LOOP    {s.label}")
            elif s.kind == "select_when":
                lines.append(f"  WHEN ✓  {s.label}")
            elif s.kind == "select_otherwise":
                lines.append(f"  OTHERWISE  {s.label}")
            elif s.kind == "select_no_match":
                lines.append(f"  SELECT ✗ (no match)")
            elif s.kind == "output":
                lines.append(f"  OUTPUT  {s.label or '(implicit)'}")
            elif s.kind == "delete":
                lines.append(f"  DELETE  ← row excluded")

        if self.filter_results:
            lines.append("── WHERE filters ──────────────────────────────────")
            for fr in self.filter_results:
                icon = "✓" if fr["passed"] else "✗"
                lines.append(f"  {icon} [{fr['data_step']}]  {fr['condition'][:80]}")

        lines.append("── Output ─────────────────────────────────────────")
        for k, v in self.final.items():
            orig = self.initial.get(k)
            tag = f"  (was {orig!r})" if k in self.initial and orig != v else ""
            lines.append(f"  {k} = {v!r}{tag}")
        if not self.row_passes_filter:
            lines.append("  ⚠  Row excluded by a qualifying WHERE condition")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Expression translation: SAS → Python
# ─────────────────────────────────────────────────────────────────────────────

_KEYWORD_OPS = [
    (r"\bGE\b", ">="),
    (r"\bLE\b", "<="),
    (r"\bGT\b", ">"),
    (r"\bLT\b", "<"),
    (r"\bNE\b", "!="),
    (r"\bEQ\b", "=="),
    (r"\bAND\b", " and "),
    (r"\bOR\b",  " or "),
    (r"\bNOT\b", " not "),
]

_SAFE_GLOBALS = {"__builtins__": None}
_SAFE_LOCALS: dict[str, Any] = {
    # Math
    "max": max, "min": min, "abs": abs,
    "round": lambda x, n=1: round(x / n) * n if n else x,
    "len": len,
    "int": int, "float": float, "str": str,
    # Constants
    "True": True, "False": False, "None": None,
    # SAS numeric functions
    "log": __import__("math").log,
    "log2": __import__("math").log2,
    "log10": __import__("math").log10,
    "exp": __import__("math").exp,
    "sqrt": __import__("math").sqrt,
    "floor": __import__("math").floor,
    "ceil": __import__("math").ceil,
    "sign": lambda x: (1 if x > 0 else -1 if x < 0 else 0),
    "mod": lambda a, b: a % b,
    # SAS string functions (basic)
    "upcase": lambda s: str(s).upper() if s is not None else None,
    "lowcase": lambda s: str(s).lower() if s is not None else None,
    "strip": lambda s: str(s).strip() if s is not None else None,
    "trim": lambda s: str(s).rstrip() if s is not None else None,
    "substr": lambda s, p, n=None: (str(s)[p-1:p-1+n] if n else str(s)[p-1:]) if s is not None else None,
    "index": lambda s, sub: (str(s).find(str(sub)) + 1) if s is not None else 0,
    "scan": lambda s, n, d=" ": str(s).split(d)[n-1] if s is not None and len(str(s).split(d)) >= n else "",
    "compress": lambda s, c="": str(s).replace(c, "") if s is not None else None,
    "cats": lambda *args: "".join(str(a).strip() for a in args if a is not None),
    "cat": lambda *args: "".join(str(a) for a in args if a is not None),
    "catx": lambda sep, *args: sep.join(str(a).strip() for a in args if a is not None),
    "reverse": lambda s: str(s)[::-1] if s is not None else None,
    "repeat": lambda s, n: str(s) * (n + 1) if s is not None else None,
    # SAS date (simplified — return numeric placeholders)
    "today": lambda: 23376,   # SAS date for reference date
    "year": lambda d: 2024,
    "month": lambda d: 1,
    "day": lambda d: 1,
    # Logic
    "coalesce": lambda *args: next((a for a in args if a is not None), None),
    "coalescec": lambda *args: next((a for a in args if a is not None and a != ""), None),
    "missing": lambda x: x is None,
    "nmiss": lambda *args: sum(1 for a in args if a is None),
    "n": lambda *args: sum(1 for a in args if a is not None),
    "ifn": lambda cond, t, f: t if cond else f,
    "ifc": lambda cond, t, f: t if cond else f,
    "sum": lambda *args: sum(a for a in args if a is not None),
    "mean": lambda *args: (sum(a for a in args if a is not None) / sum(1 for a in args if a is not None))
            if any(a is not None for a in args) else None,
    # Lag placeholder (returns None — real lag needs multi-row context)
    "lag": lambda x, n=1: None,
    "lag1": lambda x: None,
    "lag2": lambda x: None,
    # Put/input (type conversion)
    "input": lambda val, fmt=None: (float(val) if val is not None else None),
    "put": lambda val, fmt=None: (str(val) if val is not None else ""),
    # Rank / quantile (placeholder)
    "probnorm": lambda x: 0.5,
}


def _to_python(expr: str, is_condition: bool = False) -> str:
    """Translate a SAS expression to a Python-evaluable string."""
    result = expr.strip()

    # Strip SAS format/informat literals like PD best12. → just PD (heuristic)
    # Only strip format specs starting with a letter (avoids munging floats like 0.0003)
    result = re.sub(r"\s+[A-Za-z]\w*\.\d*\b", "", result)

    # NOT IN before IN translation
    result = re.sub(
        r"\b(\w+)\s+NOT\s+IN\s*\(([^)]+)\)",
        lambda m: f"{m.group(1)} not in ({m.group(2)})",
        result, flags=re.IGNORECASE,
    )
    # IN (list) operator
    result = re.sub(
        r"\b(\w+)\s+IN\s*\(([^)]+)\)",
        lambda m: f"{m.group(1)} in ({m.group(2)})",
        result, flags=re.IGNORECASE,
    )
    # BETWEEN a AND b  →  (a <= x <= b)
    result = re.sub(
        r"(\w+)\s+BETWEEN\s+([\w.'\"]+)\s+AND\s+([\w.'\"]+)",
        lambda m: f"({m.group(2)} <= {m.group(1)} <= {m.group(3)})",
        result, flags=re.IGNORECASE,
    )
    # IS MISSING / IS NOT MISSING
    result = re.sub(r"\bIS\s+NOT\s+MISSING\b", "is not None", result, flags=re.IGNORECASE)
    result = re.sub(r"\bIS\s+MISSING\b",       "is None",     result, flags=re.IGNORECASE)
    # MISSING(x) → (x is None)
    result = re.sub(r"\bMISSING\s*\((\w+)\)", r"(\1 is None)", result, flags=re.IGNORECASE)
    # NMISS(a,b) → nmiss(a,b)
    result = re.sub(r"\bNMISS\s*\(", "nmiss(", result, flags=re.IGNORECASE)

    # SAS string concatenation || → string concat — wrap in str() calls
    result = re.sub(r"\|\|", " + ", result)

    # Keyword operators
    for pattern, replacement in _KEYWORD_OPS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # ^= ~= → !=
    result = re.sub(r"[\^~]=", "!=", result)
    # SAS missing value literal . (standalone)
    result = re.sub(r"(?<!\w)\.(?!\w)", "None", result)

    # SAS function name remapping (case-insensitive → lowercase Python names)
    _fn_map = [
        ("_LOOKUP", "_lookup"),
        ("LOOKUP", "_lookup"),
        ("MAX", "max"), ("MIN", "min"), ("ABS", "abs"), ("ROUND", "round"),
        ("INT", "int"), ("FLOOR", "floor"), ("CEIL", "ceil"), ("SQRT", "sqrt"),
        ("LOG2", "log2"), ("LOG10", "log10"), ("LOG", "log"), ("EXP", "exp"),
        ("MOD", "mod"), ("SIGN", "sign"),
        ("LENGTH", "len"), ("TRIM", "trim"), ("STRIP", "strip"),
        ("UPCASE", "upcase"), ("LOWCASE", "lowcase"),
        ("COMPRESS", "compress"), ("SUBSTR", "substr"),
        ("SCAN", "scan"), ("INDEX", "index"), ("REVERSE", "reverse"),
        ("CATS", "cats"), ("CAT", "cat"), ("CATX", "catx"), ("REPEAT", "repeat"),
        ("COALESCE", "coalesce"), ("COALESCEC", "coalescec"),
        ("IFN", "ifn"), ("IFC", "ifc"),
        ("SUM", "sum"), ("MEAN", "mean"),
        ("N(?!ONE)", "n"), ("TODAY", "today"), ("DATE", "today"),
        ("YEAR", "year"), ("MONTH", "month"), ("DAY", "day"),
        ("LAG1", "lag1"), ("LAG2", "lag2"), ("LAG", "lag"),
        ("INPUT", "input"), ("PUT", "put"),
        ("PROBNORM", "probnorm"),
    ]
    for sas_fn, py_fn in _fn_map:
        result = re.sub(rf"\b{sas_fn}\s*\(", f"{py_fn}(", result, flags=re.IGNORECASE)

    # In conditions, standalone = means equality (not already handled by keyword ops)
    if is_condition:
        result = re.sub(r"(?<![<>!=^~])=(?!=)", "==", result)

    return result


def _mk_lookup(ref_data: dict[str, list[dict]]) -> Any:
    """Build lookup function that searches reference datasets."""
    def _lookup(table_name: str, key_field: str, key_value: Any, return_field: str) -> Any:
        table = ref_data.get(table_name, [])
        for row in table:
            if row.get(key_field) == key_value:
                return row.get(return_field)
        return None
    return _lookup


def _eval_expr(expr: str, env: dict[str, Any], is_condition: bool = False) -> Any:
    """Safely evaluate a SAS expression. Returns None/False on any error."""
    py = _to_python(expr, is_condition=is_condition)
    ref_data = env.get("_REFERENCE_DATA", {})
    try:
        return eval(py, _SAFE_GLOBALS,
                    {**_SAFE_LOCALS, "_lookup": _mk_lookup(ref_data), **env})  # noqa: S307
    except Exception:
        return False if is_condition else None


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

# Statements that carry no business logic — capture as metadata only
_METADATA_KEYWORDS = re.compile(
    r"^(LIBNAME|FILENAME|OPTIONS|TITLE|FOOTNOTE|ODS|RUN|QUIT|"
    r"LABEL|ATTRIB|FORMAT|INFORMAT|LENGTH|KEEP|DROP|RENAME|"
    r"FILE|PUT|INPUT|INFILE|WINDOW|DISPLAY|NOTE|"
    r"%GLOBAL|%LOCAL|%PUT|%INCLUDE)\b",
    re.IGNORECASE,
)

_MAX_LOOP_ITERS = 1000   # safety cap for DO loop evaluation


def _strip_comments(code: str) -> str:
    """Remove SAS block comments /* ... */ and line comments * ... ;"""
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    # SAS line comment: '*' must START a statement (after ';' or at file start),
    # NOT appear mid-statement like a SELECT *, clause.
    code = re.sub(r"(?:(?<=;)|\A)\s*\*[^;]*;", "", code)
    return code


def _strip_macro_refs(code: str) -> str:
    """Replace &macrovar references with MVAR[name] so table names stay readable.

    Uses square-bracket delimiters (``MVAR[name]``) so the macro variable name
    is unambiguous even when the name itself contains underscores (e.g.
    ``&PREF_PROG.`` → ``MVAR[PREF_PROG]``).  Adjacent macro vars are separated
    by the next literal character — e.g. ``&LIBWORK.&TABLE.`` → ``MVAR[LIBWORK]MVAR[TABLE]``.
    """
    # Double-ampersand (&&var.) must be handled first
    code = re.sub(r"&&(\w+)\.?", r"MVAR[\1]", code)
    code = re.sub(r"&(\w+)\.?", r"MVAR[\1]", code)
    return code


def _split_statements(code: str) -> list[str]:
    """Split SAS code by ';' returning non-empty stripped statements."""
    return [s.strip() for s in code.split(";") if s.strip()]


def _split_statements_with_lines(code: str) -> list[tuple[int, str]]:
    """Split SAS code by ';' returning (line_number, statement) tuples.

    Each statement's line number is the 1-indexed line of its first character
    in *code*, computed by counting newlines up to the statement's position.
    """
    stmts = _split_statements(code)
    result: list[tuple[int, str]] = []
    pos = 0
    for stmt in stmts:
        idx = code.find(stmt, pos)
        line_no = code[:idx].count('\n') + 1 if idx >= 0 else 0
        pos = idx + len(stmt) if idx >= 0 else pos + len(stmt) + 1
        result.append((line_no, stmt))
    return result


def _collect_sql_fragment_macros(code: str) -> dict[str, tuple[list[str], str]]:
    """Find %macro definitions whose bodies contain no semicolons.

    These are *SQL field-list fragment* macros — their body is a bare SQL
    expression (e.g. ``, CASE … AS alias``) meant to be inlined inside a
    ``SELECT`` clause.  They have no ``DATA``/``PROC`` structure and cannot be
    parsed as standalone statements, so we expand them before the main parse.

    Returns ``{lowercase_name: (param_list, body_template)}``.
    """
    macros: dict[str, tuple[list[str], str]] = {}
    pattern = re.compile(
        r"%macro\s+(\w+)\s*\(([^)]*)\)\s*;(.*?)%mend\b",
        re.IGNORECASE | re.DOTALL,
    )
    for m in pattern.finditer(code):
        name = m.group(1).lower()
        raw_params = m.group(2)
        body = m.group(3).lstrip(";").strip()
        # Skip structural macros (bodies that contain statements / DATA steps / PROCs)
        if ";" in body:
            continue
        params = [p.strip() for p in raw_params.split(",") if p.strip()]
        macros[name] = (params, body)
    return macros


def _expand_sql_fragment_macros(
    code: str,
    macros: dict[str, tuple[list[str], str]],
) -> str:
    """Replace ``%name(args)`` call sites with the expanded macro body.

    Only macros present in *macros* (the SQL-fragment set) are touched;
    structural macro calls such as ``%LGD_BE_EST(...)`` are left unchanged.

    Parameter substitution handles both ``&PARAM.`` (with trailing dot) and
    ``&PARAM`` (without), case-insensitively.
    """
    if not macros:
        return code

    def _replacer(m: re.Match) -> str:
        name = m.group(1).lower()
        if name not in macros:
            return m.group(0)       # leave unknown/structural calls unchanged
        params, body = macros[name]
        # Simple comma-split — fragment-macro args are always simple tokens
        args = [a.strip() for a in m.group(2).split(",")]
        result = body
        for param, arg in zip(params, args):
            result = re.sub(
                rf"&{re.escape(param)}\.?", arg, result, flags=re.IGNORECASE
            )
        return result

    # %macroname(possibly, multiple, args) — args never contain nested parens
    # in the known fragment macros, so a simple [^)]* is safe.
    return re.sub(r"%(\w+)\s*\(([^)]*)\)", _replacer, code, flags=re.IGNORECASE)


def _collect_structural_macros(code: str) -> dict[str, tuple[list[str], str]]:
    """Find ``%macro NAME(params); ... %mend;`` definitions whose body contains
    statements (a ``;``).

    These are the real program macros (DATA steps / PROC SQL wrapped in a macro
    with table-name parameters). Returns ``{lowercase_name: (params, body)}``.
    SQL-fragment macros (no ``;`` in the body) are handled separately and skipped.
    """
    defs: dict[str, tuple[list[str], str]] = {}
    pattern = re.compile(
        r"%macro\s+(\w+)\s*(?:\(([^)]*)\))?\s*;(.*?)%mend\b\s*;?",
        re.IGNORECASE | re.DOTALL,
    )
    for m in pattern.finditer(code):
        name = m.group(1).lower()
        raw_params = m.group(2) or ""
        body = m.group(3)
        if ";" not in body:          # SQL-fragment macro — handled elsewhere
            continue
        # Parameter names only (drop any ``=default``)
        params = [p.split("=")[0].strip() for p in raw_params.split(",") if p.strip()]
        defs[name] = (params, body)
    return defs


def _bind_macro_args(params: list[str], argstr: str) -> dict[str, str]:
    """Bind a call's argument string to parameter names (named or positional).

    Trailing dots are stripped from values because in SAS ``&VAR.`` the dot
    is a delimiter, not part of the value.
    """
    mapping: dict[str, str] = {}
    pos = 0
    for arg in _split_csv(argstr):
        if not arg:
            continue
        if "=" in arg:
            key, val = arg.split("=", 1)
            mapping[key.strip().lower()] = val.strip().rstrip(".")
        else:
            if pos < len(params):
                mapping[params[pos].lower()] = arg.strip().rstrip(".")
            pos += 1
    return mapping


def _substitute_macro_params(body: str, mapping: dict[str, str]) -> str:
    """Replace ``&param.`` / ``&param`` references in *body* with bound values."""
    for key, val in mapping.items():
        body = re.sub(rf"&{re.escape(key)}\.", val, body, flags=re.IGNORECASE)
        body = re.sub(rf"&{re.escape(key)}\b\.?", val, body, flags=re.IGNORECASE)
    return body


def _remove_macro_def(code: str, name: str) -> str:
    """Remove a single ``%macro NAME ... %mend;`` block from *code*."""
    pattern = re.compile(
        r"%macro\s+" + re.escape(name) + r"\s*(?:\([^)]*\))?\s*;.*?%mend\b\s*;?",
        re.IGNORECASE | re.DOTALL,
    )
    return pattern.sub("", code, count=1)


def _expand_called_macros(code: str, max_passes: int = 8) -> str:
    """Inline ``%NAME(args)`` invocations of program macros, substituting the
    call's argument values into the macro body.

    This resolves table-name parameters (e.g. ``&PREFIJO_PROGRAMA.``) to the
    literal values supplied at the call site, so tables across different macro
    instances get distinct, correct names (``I_P12_06`` vs ``I_P13_06``)
    instead of all collapsing onto the same ``&PREFIJO_PROGRAMA.`` placeholder.

    Only macros that have BOTH a definition and at least one call are expanded;
    defined-but-never-called macros are left untouched so they still parse into
    ``MacroDefNode`` entries. Expansion runs in bounded passes to resolve nested
    macro calls (a macro body that calls another macro).
    """
    defs = _collect_structural_macros(code)
    if not defs:
        return code

    expandable = {
        name for name in defs
        if re.search(r"%" + re.escape(name) + r"\s*\(", code, re.IGNORECASE)
    }
    if not expandable:
        return code

    # Drop the definitions of expandable macros so they are not also parsed as
    # standalone MacroDefNodes; their bodies live in ``defs`` for inlining.
    driver = code
    for name in expandable:
        driver = _remove_macro_def(driver, name)

    call_pattern = re.compile(r"%(\w+)\s*\(([^()]*)\)\s*;?", re.IGNORECASE | re.DOTALL)

    def _replacer(m: re.Match) -> str:
        name = m.group(1).lower()
        if name not in expandable:
            return m.group(0)        # leave non-program macro calls unchanged
        params, body = defs[name]
        return _substitute_macro_params(body, _bind_macro_args(params, m.group(2)))

    for _ in range(max_passes):
        expanded = call_pattern.sub(_replacer, driver)
        if expanded == driver:
            break
        driver = expanded
    return driver


class _Parser:
    def __init__(self, stmts_with_lines: list[tuple[int, str]]):
        self._s = [s for _, s in stmts_with_lines]
        self._lines = [l for l, _ in stmts_with_lines]
        self._pos = 0
        self._uninterpreted: list[dict] = []

    # ── cursor ────────────────────────────────────────────────────────────────

    def _peek(self) -> str | None:
        return self._s[self._pos] if self._pos < len(self._s) else None

    def _consume(self) -> str:
        s = self._s[self._pos]
        self._pos += 1
        return s

    def _upper(self) -> str:
        p = self._peek()
        return p.upper().strip() if p else ""

    # ── top-level ─────────────────────────────────────────────────────────────

    def parse(self) -> list[AnyNode]:
        nodes: list[AnyNode] = []
        while self._pos < len(self._s):
            result = self._parse_one_top_level()
            if result is not None:
                if isinstance(result, list):
                    nodes.extend(result)
                else:
                    nodes.append(result)
        return nodes

    def _parse_one_top_level(self) -> "AnyNode | list[AnyNode] | None":
        """Parse a single top-level statement (DATA step, PROC, macro, etc.)."""
        u = self._upper()
        if re.match(r"^DATA\b", u) and not re.match(r"^DATA\s*;", u):
            return self._parse_data_step()
        if re.match(r"^PROC\s+\w", u):
            return self._parse_proc()
        if re.match(r"^%MACRO\b", u, re.IGNORECASE):
            return self._parse_macro_def()
        if re.match(r"^%LET\b", u, re.IGNORECASE):
            return self._parse_macro_let()
        if re.match(r"^%IF\b", u, re.IGNORECASE):
            return self._parse_macro_if_block()
        if re.match(r"^%DO\b", u, re.IGNORECASE):
            return self._parse_macro_do_block()
        if re.match(r"^%END\b", u, re.IGNORECASE):
            self._consume()  # orphaned %END
            return None
        if re.match(r"^%MEND\b", u, re.IGNORECASE):
            self._consume()  # orphaned %MEND
            return None
        if re.match(r"^%[A-Z_]\w*\s*\(", u, re.IGNORECASE):
            return self._parse_macro_call()
        if _METADATA_KEYWORDS.match(u):
            self._consume()
            return None
        # Unknown top-level statement — record as uninterpreted
        self._uninterpreted.append({
            "line": self._lines[self._pos],
            "statement": self._s[self._pos][:120],
            "context": "top_level",
        })
        self._consume()
        return None

    # ── INPUT / DATALINES parsing ─────────────────────────────────────────────

    def _parse_input_vars(self, stmt: str) -> list[tuple[str, bool]]:
        """Parse INPUT statement into list of (var_name, is_string)."""
        cleaned = re.sub(r"^INPUT\s+", "", stmt, flags=re.IGNORECASE).strip()
        parts = cleaned.split()
        vars_: list[tuple[str, bool]] = []
        i = 0
        while i < len(parts):
            token = parts[i]
            if token.upper() == "@" and i + 1 < len(parts):
                i += 2  # skip pointer control
                continue
            if token in (":", "&", "~"):
                i += 1
                continue
            # Check if next token is $
            is_str = (i + 1 < len(parts) and parts[i + 1] == "$")
            var = token
            if var.startswith(":"):
                var = var[1:]
            if is_str:
                vars_.append((var, True))
                i += 2
            else:
                vars_.append((var, False))
                i += 1
        return vars_

    def _parse_datalines(self, text: str,
                         vars_: list[tuple[str, bool]]) -> list[dict]:
        """Parse DATALINES text into rows using input vars spec."""
        rows: list[dict] = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) < len(vars_):
                continue
            row: dict[str, Any] = {}
            for j, (var, is_str) in enumerate(vars_):
                if j >= len(tokens):
                    break
                val = tokens[j]
                if val == ".":
                    row[var] = None
                elif is_str:
                    row[var] = val
                else:
                    try:
                        row[var] = float(val)
                    except ValueError:
                        row[var] = None
            rows.append(row)
        return rows

    # ── DATA step ─────────────────────────────────────────────────────────────

    def _parse_data_step(self) -> DataStepNode:
        header = self._consume()
        # Capture potentially multiple output datasets: DATA a b c;
        m = re.match(r"DATA\s+(.*)", header, re.IGNORECASE)
        raw_out = (m.group(1).strip() if m else "work.unknown").split()
        output_ds = raw_out[0] if raw_out else "work.unknown"
        extra_outputs = raw_out[1:] if len(raw_out) > 1 else []

        input_ds = ""
        merge_datasets: list[str] = []
        by_keys: list[str] = []
        body: list[AnyNode] = []
        input_vars: list[tuple[str, bool]] = []
        datalines_data: list[dict] = []

        while self._pos < len(self._s):
            u = self._upper()
            if re.match(r"^(RUN|QUIT)\s*$", u):
                self._consume()
                break
            # SET statement — set input_ds
            if re.match(r"^SET\b", u):
                stmt = self._consume()
                m2 = re.match(r"SET\s+([\w.]+)", stmt, re.IGNORECASE)
                if m2 and not input_ds:
                    input_ds = m2.group(1)
                continue
            # MERGE statement
            if re.match(r"^MERGE\b", u):
                stmt = self._consume()
                raw = re.sub(r"^MERGE\s*", "", stmt, flags=re.IGNORECASE).strip()
                # Extract dataset names and IN= variables
                raw_datasets = re.findall(r"[\w.]+(?:\s*\([^)]*\))?", raw)
                in_vars: list[str] = []
                datasets = []
                for d in raw_datasets:
                    in_m = re.search(r"\(\s*IN\s*=\s*(\w+)", d, re.IGNORECASE)
                    if in_m:
                        in_vars.append(in_m.group(1))
                    clean = re.sub(r"\s*\([^)]*\)", "", d).strip()
                    if clean:
                        datasets.append(clean)
                merge_datasets = datasets
                if not input_ds and datasets:
                    input_ds = datasets[0]
                body.append(MergeNode(datasets=datasets, in_vars=in_vars))
                continue
            # INPUT statement — capture column schema
            if re.match(r"^INPUT\b", u):
                stmt = self._consume()
                input_vars = self._parse_input_vars(stmt)
                continue
            # DATALINES — consume inline data and parse into rows
            if re.match(r"^DATALINES\b", u):
                self._consume()
                data_raw = self._consume() if self._pos < len(self._s) else ""
                if data_raw:
                    datalines_data = self._parse_datalines(data_raw, input_vars)
                continue
            result = self._parse_body_stmt()
            if isinstance(result, list):
                body.extend(result)
            elif isinstance(result, ByNode):
                by_keys = result.keys
                body.append(result)
            elif result is not None:
                body.append(result)

        ds = DataStepNode(output_ds, input_ds, body)
        if extra_outputs:
            ds.output_datasets = extra_outputs
        if merge_datasets:
            ds.merge_datasets = merge_datasets
        if by_keys:
            ds.by_keys = by_keys
        if datalines_data:
            ds.datalines_data = datalines_data
        return ds

    # ── body statement dispatcher ─────────────────────────────────────────────

    def _parse_body_stmt(self) -> "AnyNode | None":
        stmt = self._peek()
        if stmt is None:
            return None
        u = stmt.upper().strip()

        # Control flow
        if re.match(r"^IF\b", u):
            return self._parse_if()
        if re.match(r"^DO\b", u):
            return self._parse_do()
        if re.match(r"^SELECT\b", u):
            return self._parse_select()

        # Row control
        if re.match(r"^OUTPUT\b", u):
            self._consume()
            ds = re.sub(r"^OUTPUT\s*", "", stmt, flags=re.IGNORECASE).strip()
            return OutputNode(dataset=ds)
        if re.match(r"^DELETE\b", u):
            self._consume()
            return DeleteNode()
        if re.match(r"^RETURN\s*$", u):
            self._consume()
            return ReturnNode()
        if re.match(r"^STOP\s*$", u):
            self._consume()
            return StopNode()

        # LINK / GOTO
        if re.match(r"^LINK\b", u):
            self._consume()
            lbl = re.sub(r"^LINK\s*", "", stmt, flags=re.IGNORECASE).strip()
            return LinkNode(label=lbl)
        if re.match(r"^GOTO\b", u):
            self._consume()
            lbl = re.sub(r"^GOTO\s*", "", stmt, flags=re.IGNORECASE).strip()
            return GotoNode(label=lbl)

        # CALL routine
        if re.match(r"^CALL\b", u):
            return self._parse_call()

        # WHERE
        if re.match(r"^WHERE\b", u):
            self._consume()
            cond = re.sub(r"^WHERE\s+", "", stmt, flags=re.IGNORECASE).strip()
            return FilterNode(cond)

        # BY
        if re.match(r"^BY\b", u):
            return self._parse_by()

        # RETAIN
        if re.match(r"^RETAIN\b", u):
            return self._parse_retain()

        # ARRAY
        if re.match(r"^ARRAY\b", u):
            return self._parse_array()

        # Macro statements
        if re.match(r"^%LET\b", u, re.IGNORECASE):
            return self._parse_macro_let()
        if re.match(r"^%IF\b", u, re.IGNORECASE):
            return self._parse_macro_if_block()
        if re.match(r"^%DO\b", u, re.IGNORECASE):
            return self._parse_macro_do_block()
        if re.match(r"^%END\b", u, re.IGNORECASE):
            self._consume()  # orphaned %END
            return None
        if re.match(r"^%MEND\b", u, re.IGNORECASE):
            self._consume()  # orphaned %MEND
            return None
        if re.match(r"^%[A-Z_]\w*\s*\(", u, re.IGNORECASE):
            return self._parse_macro_call()

        # Metadata-only — consume and skip
        if _METADATA_KEYWORDS.match(u):
            self._consume()
            return None

        # Sum statement: var + expr;  (implicit RETAIN + increment)
        if re.match(r"^(\w+)\s*\+\s*(.+)$", u):
            self._consume()
            sm = re.match(r"^(\w+)\s*\+\s*(.+)$", stmt.strip(), re.IGNORECASE)
            if sm:
                var = sm.group(1).strip()
                expr = sm.group(2).strip()
                # Desugar: var = var + (expr)
                return AssignNode(var, f"{var} + ({expr})")
            return None

        # Assignment: var = expr  (also handles array element: arr[i] = expr)
        if re.match(r"^[\w\[\]{}()]+\s*=\s*(?!=)", u):
            self._consume()
            return self._stmt_to_assign(stmt)

        # Unknown statement — record as uninterpreted
        self._uninterpreted.append({
            "line": self._lines[self._pos],
            "statement": self._s[self._pos][:120],
            "context": "body",
        })
        self._consume()
        return None

    # ── IF ────────────────────────────────────────────────────────────────────

    def _parse_if(self) -> IfNode | FilterNode | None:
        stmt = self._consume()
        m = re.match(r"IF\s+(.*?)\s+THEN\s+(.*)", stmt, re.IGNORECASE | re.DOTALL)
        if not m:
            # Subsetting IF: IF condition; (no THEN) — acts as a WHERE filter
            sub_m = re.match(r"IF\s+(.+)", stmt, re.IGNORECASE | re.DOTALL)
            if sub_m:
                return FilterNode(sub_m.group(1).strip())
            return None
        condition = m.group(1).strip()
        then_raw = m.group(2).strip()

        if re.match(r"^DO\s*$", then_raw, re.IGNORECASE):
            then_branch = self._parse_block_until_end()
        else:
            node = self._inline_stmt(then_raw)
            then_branch = [node] if node else []

        else_branch: list[AnyNode] = []
        # Look for ELSE or ELSE IF on the very next statement
        if self._peek() and re.match(r"^ELSE\b", self._upper()):
            else_stmt = self._consume()
            else_body = re.sub(r"^ELSE\s*", "", else_stmt, flags=re.IGNORECASE).strip()
            if re.match(r"^DO\s*$", else_body, re.IGNORECASE):
                else_branch = self._parse_block_until_end()
            elif re.match(r"^IF\b", else_body.upper()):
                # ELSE IF → recurse as inline
                inner_m = re.match(r"IF\s+(.*?)\s+THEN\s+(.*)", else_body, re.IGNORECASE | re.DOTALL)
                if inner_m:
                    inner_then_raw = inner_m.group(2).strip()
                    inner_then = (self._parse_block_until_end()
                                  if re.match(r"^DO\s*$", inner_then_raw, re.IGNORECASE)
                                  else ([self._inline_stmt(inner_then_raw)]
                                        if self._inline_stmt(inner_then_raw) else []))
                    else_branch = [IfNode(inner_m.group(1).strip(), inner_then)]
            elif else_body:
                node = self._inline_stmt(else_body)
                else_branch = [node] if node else []

        return IfNode(condition, then_branch, else_branch)

    # ── DO ────────────────────────────────────────────────────────────────────

    def _parse_do(self) -> DoLoopNode | None:
        stmt = self._consume()
        u = stmt.upper().strip()

        # DO WHILE (cond)
        m = re.match(r"DO\s+WHILE\s*\((.+)\)\s*$", u, re.IGNORECASE)
        if m:
            cond_raw = stmt[m.start(1) - len(stmt) + len(u) - len(u) :]  # preserve original case
            # simpler: extract from original stmt
            m2 = re.match(r"DO\s+WHILE\s*\((.+)\)\s*$", stmt, re.IGNORECASE)
            cond = m2.group(1).strip() if m2 else m.group(1).strip()
            body = self._parse_block_until_end()
            return DoLoopNode(while_cond=cond, body=body)

        # DO UNTIL (cond)
        m = re.match(r"DO\s+UNTIL\s*\((.+)\)\s*$", stmt, re.IGNORECASE)
        if m:
            body = self._parse_block_until_end()
            return DoLoopNode(until_cond=m.group(1).strip(), body=body)

        # DO var = start TO stop [BY step] [WHILE(cond)] [UNTIL(cond)]
        m = re.match(
            r"DO\s+(\w+)\s*=\s*(.+?)\s+TO\s+(.+?)(?:\s+BY\s+(.+?))?(?:\s+WHILE\s*\((.+?)\))?(?:\s+UNTIL\s*\((.+?)\))?\s*$",
            stmt, re.IGNORECASE,
        )
        if m:
            var   = m.group(1).strip()
            start = m.group(2).strip()
            stop  = m.group(3).strip()
            step  = (m.group(4) or "1").strip()
            wcond = (m.group(5) or "").strip()
            ucond = (m.group(6) or "").strip()
            body  = self._parse_block_until_end()
            return DoLoopNode(var=var, start=start, stop=stop, by_step=step,
                              while_cond=wcond, until_cond=ucond, body=body)

        # Bare DO; ... END;  (treat as anonymous block — flatten into inline body)
        body = self._parse_block_until_end()
        return DoLoopNode(body=body)   # no loop — just a group

    # ── SELECT ────────────────────────────────────────────────────────────────

    def _parse_select(self) -> SelectNode | None:
        stmt = self._consume()
        m = re.match(r"SELECT\s*\((.+)\)\s*$", stmt, re.IGNORECASE)
        select_expr = m.group(1).strip() if m else ""

        whens: list[WhenNode] = []
        otherwise: list[AnyNode] = []

        while self._pos < len(self._s):
            u = self._upper()
            if re.match(r"^END\s*$", u):
                self._consume()
                break
            if re.match(r"^OTHERWISE\b", u):
                ow_stmt = self._consume()
                # Inline otherwise: OTHERWISE <stmt>
                inline = re.sub(r"^OTHERWISE\s*", "", ow_stmt, flags=re.IGNORECASE).strip()
                if not inline:
                    # Nothing on same line — check next statement
                    if self._peek() and re.match(r"^DO\b", self._upper()):
                        self._consume()
                        otherwise = self._parse_block_until_end()
                    else:
                        node = self._parse_body_stmt()
                        if node:
                            otherwise = [node]
                elif re.match(r"^DO\s*$", inline, re.IGNORECASE):
                    otherwise = self._parse_block_until_end()
                else:
                    node = self._inline_stmt(inline)
                    if node:
                        otherwise = [node]
            elif re.match(r"^WHEN\b", u):
                when_node = self._parse_when()
                if when_node:
                    whens.append(when_node)
            else:
                self._consume()

        return SelectNode(select_expr=select_expr, whens=whens, otherwise=otherwise)

    def _parse_when(self) -> WhenNode | None:
        stmt = self._consume()
        # WHEN (val1, val2, ...) or WHEN (cond)
        m = re.match(r"WHEN\s*\((.+)\)\s*(.*)", stmt, re.IGNORECASE | re.DOTALL)
        if not m:
            return None
        raw_vals = m.group(1).strip()
        inline = m.group(2).strip()

        # Split values by comma (respecting strings)
        values = [v.strip() for v in _split_csv(raw_vals)]

        if re.match(r"^DO\s*$", inline, re.IGNORECASE) or not inline:
            if not inline:
                body = self._parse_block_until_end()
            else:
                body = self._parse_block_until_end()
        else:
            node = self._inline_stmt(inline)
            body = [node] if node else []

        return WhenNode(values=values, body=body)

    # ── ARRAY ─────────────────────────────────────────────────────────────────

    def _parse_array(self) -> ArrayNode | None:
        stmt = self._consume()
        # ARRAY name {dims} [_TEMPORARY_] [var1 var2 ...] [(init_vals)]
        m = re.match(
            r"ARRAY\s+(\w+)\s*[{(\[]([^})\]]*)[})\]](.*)",
            stmt, re.IGNORECASE,
        )
        if not m:
            # ARRAY without braces: ARRAY name * var1 var2 ...
            m2 = re.match(r"ARRAY\s+(\w+)\s+(\*|\d+)\s+(.*)", stmt, re.IGNORECASE)
            if m2:
                return ArrayNode(
                    name=m2.group(1).strip(),
                    dims=m2.group(2).strip(),
                    vars=[v.strip() for v in m2.group(3).split() if v.strip() and not v.startswith("(")],
                )
            return ArrayNode(name="UNKNOWN", dims="*", vars=[])

        name = m.group(1).strip()
        dims = m.group(2).strip()
        rest = m.group(3).strip()

        temporary = bool(re.search(r"\b_TEMPORARY_\b", rest, re.IGNORECASE))
        rest_clean = re.sub(r"\b_TEMPORARY_\b", "", rest, flags=re.IGNORECASE)

        # Initial values in parens at end
        init_vals: list[str] = []
        init_m = re.search(r"\(([^)]+)\)\s*$", rest_clean)
        if init_m:
            raw_init = init_m.group(1).strip()
            if "," in raw_init:
                init_vals = [v.strip() for v in raw_init.split(",") if v.strip()]
            else:
                init_vals = raw_init.split()
            rest_clean = rest_clean[:init_m.start()].strip()

        var_list = [v.strip() for v in rest_clean.split() if v.strip()]

        return ArrayNode(name=name, dims=dims, vars=var_list,
                         temporary=temporary, initial_values=init_vals)

    # ── RETAIN ────────────────────────────────────────────────────────────────

    def _parse_retain(self) -> RetainNode:
        stmt = self._consume()
        rest = re.sub(r"^RETAIN\s*", "", stmt, flags=re.IGNORECASE).strip()
        # Separate variable list from optional initial value
        parts = rest.split()
        # If last token is numeric literal or string, it's an initial value
        if len(parts) > 1:
            last = parts[-1]
            if re.match(r"^[0-9.\-'\"]+$", last):
                return RetainNode(vars=parts[:-1], initial=last)
        return RetainNode(vars=parts)

    # ── BY ────────────────────────────────────────────────────────────────────

    def _parse_by(self) -> ByNode:
        stmt = self._consume()
        rest = re.sub(r"^BY\s*", "", stmt, flags=re.IGNORECASE).strip()
        keys: list[str] = []
        desc: list[bool] = []
        tokens = rest.split()
        i = 0
        while i < len(tokens):
            if tokens[i].upper() == "DESCENDING" and i + 1 < len(tokens):
                keys.append(tokens[i + 1])
                desc.append(True)
                i += 2
            else:
                keys.append(tokens[i])
                desc.append(False)
                i += 1
        return ByNode(keys=keys, descending=desc)

    # ── CALL ──────────────────────────────────────────────────────────────────

    def _parse_call(self) -> CallNode:
        stmt = self._consume()
        m = re.match(r"CALL\s+(\w+)\s*(?:\((.*)?\))?\s*$", stmt, re.IGNORECASE | re.DOTALL)
        if m:
            return CallNode(routine=m.group(1).upper(), args=(m.group(2) or "").strip())
        return CallNode(routine="UNKNOWN", args="")

    # ── Macro %LET ────────────────────────────────────────────────────────────

    def _parse_macro_let(self) -> MacroLetNode | None:
        stmt = self._consume()
        m = re.match(r"%LET\s+(\w+)\s*=\s*(.*)", stmt, re.IGNORECASE | re.DOTALL)
        if m:
            return MacroLetNode(var=m.group(1).strip(), value=m.group(2).strip())
        return None

    # ── Macro %MACRO/%MEND ────────────────────────────────────────────────────

    def _parse_macro_def(self) -> MacroDefNode | None:
        header = self._consume()
        m = re.match(r"%MACRO\s+(\w+)\s*(?:\(([^)]*)\))?\s*$", header, re.IGNORECASE)
        if not m:
            return None
        name = m.group(1).strip()
        params = (m.group(2) or "").strip()
        body: list[AnyNode] = []
        while self._pos < len(self._s):
            u = self._upper()
            if re.match(r"^%MEND\b", u, re.IGNORECASE):
                self._consume()
                break
            result = self._parse_one_top_level()
            if result is not None:
                if isinstance(result, list):
                    body.extend(result)
                else:
                    body.append(result)
        return MacroDefNode(name=name, params=params, body=body)

    # ── Macro %IF/%THEN/%DO ... %END ────────────────────────────────────────

    def _parse_macro_if_block(self) -> list[AnyNode] | None:
        """Parse %IF ... %THEN %DO ... %END [%ELSE %DO ... %END].

        Returns inner nodes from both branches (flattened) since we cannot
        evaluate macro conditions at parse time.
        """
        stmt = self._consume()
        u = stmt.upper()
        nodes: list[AnyNode] = []

        # %IF ... %THEN %DO on same statement
        if "%THEN" in u and "%DO" in u:
            nodes.extend(self._parse_macro_block_until_pct_end())
        elif "%THEN" in u:
            # %THEN without %DO — single inline statement follows
            result = self._parse_one_top_level()
            if result is not None:
                if isinstance(result, list):
                    nodes.extend(result)
                else:
                    nodes.append(result)
        # else: %IF without %THEN on same line — just consumed, inner stmts
        # will be parsed by the caller naturally

        # Check for %ELSE
        if self._pos < len(self._s):
            nu = self._upper()
            if re.match(r"^%ELSE\b", nu, re.IGNORECASE):
                stmt2 = self._consume()
                u2 = stmt2.upper()
                if "%DO" in u2:
                    nodes.extend(self._parse_macro_block_until_pct_end())
                else:
                    result = self._parse_one_top_level()
                    if result is not None:
                        if isinstance(result, list):
                            nodes.extend(result)
                        else:
                            nodes.append(result)

        return nodes if nodes else None

    # ── Macro %DO ... %END ───────────────────────────────────────────────────

    def _parse_macro_do_block(self) -> list[AnyNode]:
        """Consume %DO ... %END block, recursively parsing inner statements."""
        self._consume()  # consume the %DO statement
        return self._parse_macro_block_until_pct_end()

    def _parse_macro_block_until_pct_end(self) -> list[AnyNode]:
        """Parse statements until matching %END, handling nested %DO/%END."""
        nodes: list[AnyNode] = []
        depth = 0
        while self._pos < len(self._s):
            u = self._upper()
            if re.match(r"^%END\b", u, re.IGNORECASE):
                if depth == 0:
                    self._consume()
                    break
                depth -= 1
                self._consume()
                continue
            # Track nested %DO (standalone or inside %IF...%THEN %DO)
            if re.match(r"^%DO\b", u, re.IGNORECASE):
                depth += 1
            elif re.match(r"^%IF\b", u, re.IGNORECASE) and "%DO" in u:
                depth += 1
            result = self._parse_one_top_level()
            if result is not None:
                if isinstance(result, list):
                    nodes.extend(result)
                else:
                    nodes.append(result)
        return nodes

    # ── Macro call %name(...) ─────────────────────────────────────────────────

    def _parse_macro_call(self) -> MacroCallNode | None:
        stmt = self._consume()
        m = re.match(r"%(\w+)\s*(?:\(([^)]*)\))?\s*$", stmt, re.IGNORECASE)
        if m:
            return MacroCallNode(name=m.group(1), args=(m.group(2) or "").strip())
        return None

    # ── PROC ──────────────────────────────────────────────────────────────────

    def _parse_proc(self) -> ProcNode:
        header = self._consume()
        m = re.match(r"PROC\s+(\w+)(?:\s+DATA\s*=\s*([\w.]+))?", header, re.IGNORECASE)
        kind = m.group(1).upper() if m else "UNKNOWN"
        data = m.group(2) if (m and m.group(2)) else ""
        raw_lines = [header]
        while self._pos < len(self._s):
            u = self._upper()
            if re.match(r"^(RUN|QUIT)\s*$", u):
                self._consume()
                break
            raw_lines.append(self._consume())
        raw = "; ".join(raw_lines)
        node = ProcNode(kind, data, raw)
        if kind == "SQL":
            out_tbl, in_tbls, fields = _parse_proc_sql_fields(raw)
            node.output_table = out_tbl
            node.input_tables = in_tbls
            node.select_fields = fields
            if not node.data and in_tbls:
                node.data = in_tbls[0]
        return node

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _parse_block_until_end(self) -> list[AnyNode]:
        """Parse statements until END; — used by IF/DO/SELECT blocks."""
        nodes: list[AnyNode] = []
        depth = 0
        while self._pos < len(self._s):
            u = self._upper()
            if re.match(r"^END\s*$", u):
                if depth == 0:
                    self._consume()
                    break
                depth -= 1
                self._consume()
                continue
            # Track nested DO/SELECT that also have END
            if re.match(r"^DO\b", u) or re.match(r"^SELECT\b", u):
                depth += 1
            result = self._parse_body_stmt()
            if result is not None:
                nodes.append(result)
        return nodes

    def _stmt_to_assign(self, stmt: str) -> AssignNode | None:
        # Handles plain var = expr AND array element arr[i] = expr
        m = re.match(r"^([\w\[\]{}().\s]+?)\s*=\s*(.+)$", stmt.strip(), re.DOTALL)
        if not m:
            return None
        lhs = m.group(1).strip()
        # Simplify array element refs: arr[i] → arr (for lineage purposes)
        base = re.sub(r"[\[{(].*", "", lhs).strip()
        return AssignNode(base, m.group(2).strip())

    def _inline_stmt(self, stmt: str) -> AnyNode | None:
        u = stmt.upper().strip()
        if re.match(r"^IF\b", u):
            m = re.match(r"IF\s+(.*?)\s+THEN\s+(.*)", stmt, re.IGNORECASE | re.DOTALL)
            if m:
                inner = self._inline_stmt(m.group(2).strip())
                return IfNode(m.group(1).strip(), [inner] if inner else [], [])
        if re.match(r"^OUTPUT\b", u):
            ds = re.sub(r"^OUTPUT\s*", "", stmt, flags=re.IGNORECASE).strip()
            return OutputNode(dataset=ds)
        if re.match(r"^DELETE\b", u):
            return DeleteNode()
        if re.match(r"^RETURN\s*$", u):
            return ReturnNode()
        if re.match(r"^STOP\s*$", u):
            return StopNode()
        if re.match(r"^CALL\b", u):
            m = re.match(r"CALL\s+(\w+)\s*\((.*)\)", stmt, re.IGNORECASE | re.DOTALL)
            if m:
                return CallNode(routine=m.group(1), args=m.group(2).strip())
        if re.match(r"^LINK\b", u):
            lbl = re.sub(r"^LINK\s*", "", stmt, flags=re.IGNORECASE).strip()
            return LinkNode(label=lbl)
        if re.match(r"^GOTO\b", u):
            lbl = re.sub(r"^GOTO?\s*", "", stmt, flags=re.IGNORECASE).strip()
            return GotoNode(label=lbl)
        if re.match(r"^[\w\[\]{}().\s]+\s*=\s*(?!=)", u):
            return self._stmt_to_assign(stmt)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CSV-style splitter (for WHEN value lists)
# ─────────────────────────────────────────────────────────────────────────────

def _split_csv(s: str) -> list[str]:
    """Split on commas while respecting quoted strings."""
    parts, buf, in_q, q_char = [], [], False, ""
    for ch in s:
        if in_q:
            buf.append(ch)
            if ch == q_char:
                in_q = False
        elif ch in ("'", '"'):
            in_q, q_char = True, ch
            buf.append(ch)
        elif ch == ",":
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return parts


def _split_sql_select(body: str) -> list[str]:
    """Split SQL SELECT field list on commas, respecting parens and CASE/END."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0  # parenthesis depth
    case_depth = 0  # CASE ... END depth
    i = 0
    upper = body.upper()
    while i < len(body):
        ch = body[i]
        # Track CASE/END keywords
        if upper[i:i+4] == "CASE" and (i == 0 or not upper[i-1].isalnum()):
            case_depth += 1
            buf.append(body[i:i+4])
            i += 4
            continue
        if upper[i:i+3] == "END" and case_depth > 0 and (i == 0 or not upper[i-1].isalnum()):
            case_depth -= 1
            buf.append(body[i:i+3])
            i += 3
            continue
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == ',' and depth == 0 and case_depth == 0:
            parts.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    if buf:
        parts.append("".join(buf))
    return parts


def _parse_proc_sql_fields(raw: str) -> tuple[str, list[str], list[tuple[str, str]]]:
    """Extract output table, input tables, and SELECT field aliases from PROC SQL."""
    # Output table: CREATE TABLE <name> AS
    output_table = ""
    # Table names may contain MVAR[name] bracket tokens after macro substitution
    ct_m = re.search(r"CREATE\s+TABLE\s+([\w.\[\]]+)\s+AS\b", raw, re.IGNORECASE)
    if ct_m:
        output_table = ct_m.group(1).strip()

    # Input tables: FROM <table> and JOIN <table>
    input_tables: list[str] = []
    skip = {'SELECT', 'WHERE', 'GROUP', 'ORDER', 'HAVING', 'SET', 'UNION'}
    for m in re.finditer(r"(?:FROM|JOIN)\s+([\w.\[\]]+)", raw, re.IGNORECASE):
        tbl = m.group(1).strip()
        if tbl.upper() not in skip:
            input_tables.append(tbl)

    # SELECT fields with AS alias
    select_fields: list[tuple[str, str]] = []
    sel_m = re.search(r"\bSELECT(?:\s+DISTINCT)?\s+(.*?)\bFROM\b", raw, re.IGNORECASE | re.DOTALL)
    if not sel_m:
        return output_table, input_tables, select_fields

    select_body = sel_m.group(1).strip()
    if re.match(r"^\*\s*$", select_body):
        return output_table, input_tables, [("*", "*")]

    # Handle SELECT * , computed_col — strip leading *,
    if select_body.startswith("*"):
        select_body = re.sub(r"^\*\s*,?\s*", "", select_body)
        select_fields.append(("*", "*"))

    fragments = _split_sql_select(select_body)
    for frag in fragments:
        frag = frag.strip()
        if not frag:
            continue
        # Skip macro calls (e.g. %macro_name(...))
        if frag.startswith("%"):
            continue
        # Skip passthrough alias.* fragments — no specific field name available
        if re.match(r"^\w+\.\*$", frag.strip()):
            select_fields.append(("*", frag.strip()))
            continue
        # Look for AS <alias> at the end
        as_m = re.search(r"\bAS\s+(\w+)\s*$", frag, re.IGNORECASE)
        if as_m:
            alias = as_m.group(1).strip()
            expr = frag[:as_m.start()].strip()
            select_fields.append((alias, expr))
        else:
            # No alias — extract last identifier as implicit alias
            parts = re.findall(r'[\w]+', frag)
            if parts:
                select_fields.append((parts[-1], frag))

    return output_table, input_tables, select_fields


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class _Evaluator:
    def __init__(self):
        self.steps: list[TraceStep] = []
        self._current_step: str = ""
        self._step_passes: bool = True
        self.filter_results: list[dict] = []
        self._first_filter_done: bool = False
        self._first_filter_passed: bool = True
        self._row_deleted: bool = False

    def run(self, nodes: list[AnyNode], env: dict[str, Any]) -> None:
        for node in nodes:
            if isinstance(node, DataStepNode):
                self._current_step = node.output_dataset
                self._step_passes = True
                # Set MERGE IN= variables to 1 (assume row is present)
                for body_node in node.body:
                    if isinstance(body_node, MergeNode):
                        for v in body_node.in_vars:
                            env[v] = 1
                self._eval_body(node.body, env)
            elif isinstance(node, ProcNode):
                pass
            elif isinstance(node, (MacroLetNode, MacroDefNode, MacroCallNode)):
                pass  # macro state not simulated
            else:
                self._eval_node(node, env)

    def _eval_body(self, nodes: list[AnyNode], env: dict[str, Any]) -> None:
        for node in nodes:
            if not self._step_passes or self._row_deleted:
                break
            self._eval_node(node, env)

    def _eval_node(self, node: AnyNode, env: dict[str, Any]) -> None:
        if isinstance(node, AssignNode):
            self._eval_assign(node, env)
        elif isinstance(node, IfNode):
            self._eval_if(node, env)
        elif isinstance(node, FilterNode):
            self._eval_filter(node, env)
        elif isinstance(node, DoLoopNode):
            self._eval_do_loop(node, env)
        elif isinstance(node, SelectNode):
            self._eval_select(node, env)
        elif isinstance(node, RetainNode):
            self._eval_retain(node, env)
        elif isinstance(node, OutputNode):
            self.steps.append(TraceStep(kind="output", label=node.dataset,
                                        data_step=self._current_step))
        elif isinstance(node, DeleteNode):
            self._row_deleted = True
            self.steps.append(TraceStep(kind="delete", label="",
                                        data_step=self._current_step))
        elif isinstance(node, CallNode):
            self.steps.append(TraceStep(kind="call",
                                        label=f"CALL {node.routine}({node.args})",
                                        data_step=self._current_step))
        elif isinstance(node, StopNode):
            self.steps.append(TraceStep(kind="stop", label="STOP",
                                        data_step=self._current_step))
        # Other nodes (MergeNode, ByNode, ArrayNode, LinkNode, GotoNode, etc.) — skip

    def _eval_assign(self, node: AssignNode, env: dict[str, Any]) -> None:
        old = env.get(node.var)
        new = _eval_expr(node.expr, env)
        if new is not None:
            env[node.var] = new
        self.steps.append(TraceStep(
            kind="assign", label=f"{node.var} = {node.expr}",
            var=node.var, old_val=old, new_val=env.get(node.var),
            data_step=self._current_step,
        ))

    def _eval_if(self, node: IfNode, env: dict[str, Any]) -> None:
        result = _eval_expr(node.condition, env, is_condition=True)
        taken = bool(result)
        self.steps.append(TraceStep(
            kind="if_taken" if taken else "if_skipped",
            label=node.condition, data_step=self._current_step,
        ))
        self._eval_body(node.then_branch if taken else node.else_branch, env)

    def _eval_filter(self, node: FilterNode, env: dict[str, Any]) -> None:
        condition = " ".join(node.condition.split())
        result = _eval_expr(condition, env, is_condition=True)
        passes = bool(result)
        self.steps.append(TraceStep(
            kind="filter_pass" if passes else "filter_block",
            label=condition, data_step=self._current_step,
        ))
        self.filter_results.append({
            "data_step": self._current_step,
            "condition": condition,
            "passed": passes,
        })
        if not self._first_filter_done:
            self._first_filter_done = True
            self._first_filter_passed = passes
        if not passes:
            self._step_passes = False

    def _eval_do_loop(self, node: DoLoopNode, env: dict[str, Any]) -> None:
        iters = 0
        if node.var:
            # Iterative loop
            start = _eval_expr(node.start, env) or 0
            stop  = _eval_expr(node.stop,  env) or 0
            step  = _eval_expr(node.by_step, env) or 1
            i = float(start)
            while (step > 0 and i <= float(stop)) or (step < 0 and i >= float(stop)):
                if iters >= _MAX_LOOP_ITERS:
                    break
                # Check WHILE/UNTIL if combined
                if node.while_cond and not _eval_expr(node.while_cond, env, True):
                    break
                env[node.var] = i
                self.steps.append(TraceStep(
                    kind="loop_iter", label=f"{node.var} = {i}",
                    data_step=self._current_step,
                ))
                self._eval_body(node.body, env)
                if node.until_cond and _eval_expr(node.until_cond, env, True):
                    break
                i += float(step)
                iters += 1
        elif node.while_cond:
            while iters < _MAX_LOOP_ITERS:
                if not _eval_expr(node.while_cond, env, True):
                    break
                self.steps.append(TraceStep(
                    kind="loop_iter", label=f"WHILE ({node.while_cond})",
                    data_step=self._current_step,
                ))
                self._eval_body(node.body, env)
                iters += 1
        elif node.until_cond:
            while iters < _MAX_LOOP_ITERS:
                self.steps.append(TraceStep(
                    kind="loop_iter", label=f"UNTIL ({node.until_cond})",
                    data_step=self._current_step,
                ))
                self._eval_body(node.body, env)
                iters += 1
                if _eval_expr(node.until_cond, env, True):
                    break
        else:
            # Bare DO group — just execute body once
            self._eval_body(node.body, env)

    def _eval_select(self, node: SelectNode, env: dict[str, Any]) -> None:
        select_val = _eval_expr(node.select_expr, env) if node.select_expr else None

        for when in node.whens:
            matched = False
            if node.select_expr:
                # Value-based SELECT: match select_val against WHEN values
                for v in when.values:
                    v_eval = _eval_expr(v, env)
                    if v_eval == select_val or str(v_eval) == str(select_val):
                        matched = True
                        break
            else:
                # Condition-based SELECT: evaluate each value as a condition
                for v in when.values:
                    if _eval_expr(v, env, is_condition=True):
                        matched = True
                        break
            if matched:
                self.steps.append(TraceStep(
                    kind="select_when",
                    label=", ".join(when.values)[:80],
                    data_step=self._current_step,
                ))
                self._eval_body(when.body, env)
                return  # first match wins

        if node.otherwise:
            self.steps.append(TraceStep(
                kind="select_otherwise", label="",
                data_step=self._current_step,
            ))
            self._eval_body(node.otherwise, env)
        else:
            self.steps.append(TraceStep(
                kind="select_no_match", label="",
                data_step=self._current_step,
            ))

    def _eval_retain(self, node: RetainNode, env: dict[str, Any]) -> None:
        if node.initial:
            init_val = _eval_expr(node.initial, env)
            for v in node.vars:
                # Use original case to match how _eval_assign stores variables.
                if v not in env:
                    env[v] = init_val

    @property
    def row_passes_filter(self) -> bool:
        return self._first_filter_passed and not self._row_deleted


# ─────────────────────────────────────────────────────────────────────────────
# Field lineage extraction
# ─────────────────────────────────────────────────────────────────────────────

_LINEAGE_SKIP = {
    # Language keywords / operators
    'AND', 'OR', 'NOT', 'EQ', 'NE', 'GT', 'LT', 'GE', 'LE', 'IN',
    'IF', 'THEN', 'ELSE', 'DO', 'END', 'WHERE', 'SET', 'DATA',
    'TRUE', 'FALSE', 'NULL', 'MISSING', 'TO', 'BY', 'BETWEEN',
    'SELECT', 'WHEN', 'OTHERWISE', 'WHILE', 'UNTIL', 'LIKE',
    # SELECT/CASE keywords
    'CASE', 'AS', 'DISTINCT', 'CALCULATED',
    # PROC SQL structural keywords
    'PROC', 'SQL', 'QUIT', 'RUN', 'CREATE', 'TABLE', 'DROP',
    'FROM', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 'FULL',
    'ON', 'GROUP', 'ORDER', 'HAVING', 'UNION', 'ALL', 'FORCE',
    # Numeric functions
    'MAX', 'MIN', 'ABS', 'ROUND', 'INT', 'FLOOR', 'CEIL', 'SQRT',
    'LOG', 'LOG2', 'LOG10', 'EXP', 'MOD', 'SIGN',
    'SUM', 'MEAN', 'STD', 'VAR', 'NMISS', 'N', 'RANGE',
    'LAG', 'LAG1', 'LAG2', 'DIF', 'DIF1',
    # String functions
    'LENGTH', 'SUBSTR', 'TRIM', 'STRIP', 'COMPRESS',
    'UPCASE', 'LOWCASE', 'PROPCASE',
    'CAT', 'CATS', 'CATX', 'CATT', 'REPEAT', 'REVERSE',
    'PUT', 'INPUT', 'SCAN', 'INDEX', 'FIND', 'INDEXW', 'TRANWRD',
    # Date functions
    'TODAY', 'DATE', 'DATETIME', 'YEAR', 'MONTH', 'DAY',
    'HOUR', 'MINUTE', 'SECOND', 'MDY', 'YMD',
    'DATEPART', 'TIMEPART', 'INTCK', 'INTNX',
    # Logic / lookup
    'COALESCE', 'COALESCEC', 'IFN', 'IFC',
    'WHICHN', 'WHICHC', 'DIM', 'HBOUND', 'LBOUND',
    # Stats
    'PROBNORM', 'PROBCHI', 'PROBT', 'PROBF', 'QNORM',
    # Dataset/library prefixes
    'WORK', 'MYLIB', 'SASHELP', 'SASUSER',
    # Auto variables
    'N', 'ERROR', 'FIRST', 'LAST', 'NOBS',
    # Common loop vars
    'I', 'J', 'K', 'IDX',
    # ARRAY pseudo-variables
    'TEMPORARY', 'INITIAL',
}

# SAS format-spec pattern: uppercase letters followed by digits (e.g. YYMMN6, BEST12, DATE9)
_FORMAT_SPEC_RE = re.compile(r'^[A-Z]+\d+$')


def _vars_in_expr(expr: str) -> set[str]:
    """Extract SAS variable names from an expression."""
    cleaned = re.sub(r"'[^']*'", " ", expr)
    cleaned = re.sub(r'"[^"]*"', " ", cleaned)
    cleaned = re.sub(r"&&?\w+\.?", " ", cleaned)   # strip macro var refs
    cleaned = re.sub(r"\bMVAR\s*\[[^\]]*\]", " ", cleaned)  # drop macro-var placeholders
    cleaned = re.sub(r"\[\s*\w+\s*\]", " ", cleaned)  # strip subscripts
    # alias.* (SELECT *) carries no field information — remove whole token
    cleaned = re.sub(r"\b\w+\.\*", " ", cleaned)
    # Strip SQL table-alias qualifiers: T1.FIELD → FIELD, alias.FIELD → FIELD
    cleaned = re.sub(r"\b[A-Za-z_]\w*\.(?=[A-Za-z_])", "", cleaned)
    tokens = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', cleaned)
    result: set[str] = set()
    for t in tokens:
        u = t.upper()
        if u in _LINEAGE_SKIP:
            continue
        if _FORMAT_SPEC_RE.match(u):   # SAS format spec like YYMMN6, BEST12
            continue
        if "MVAR[" in u:             # substituted macro variable placeholder
            continue
        result.add(u)
    return result


@dataclass
class FieldLineage:
    nodes: list[dict]      # {id, label, kind, layer, data_steps}
    edges: list[dict]      # {source, target, kind, data_step, expr}
    data_steps: list[str]


class _LineageWalker:
    def __init__(self) -> None:
        self._written: set[str] = set()
        self._read: set[str] = set()
        self._edges: list[dict] = []
        self._seen_edges: set[tuple] = set()
        self._step_order: list[str] = []
        self._field_steps: dict[str, list[str]] = {}
        self._current_step: str = ""
        self._cond_stack: list[set[str]] = []

    def walk(self, nodes: list[AnyNode]) -> FieldLineage:
        for node in nodes:
            if isinstance(node, DataStepNode):
                self._current_step = node.output_dataset
                if node.output_dataset not in self._step_order:
                    self._step_order.append(node.output_dataset)
                for ds in node.merge_datasets:
                    self._read.update()  # no field-level info from merge names
                self._walk_body(node.body)
            elif isinstance(node, ProcNode) and node.output_table and node.select_fields:
                self._current_step = node.output_table
                if node.output_table not in self._step_order:
                    self._step_order.append(node.output_table)
                self._walk_proc_sql(node)
            elif isinstance(node, MacroDefNode):
                self.walk(node.body)
        return self._build()

    def _walk_proc_sql(self, node: ProcNode) -> None:
        """Extract lineage edges from PROC SQL SELECT fields."""
        for alias, expr in node.select_fields:
            if alias == "*":
                # SELECT * propagates all fields already known from input tables
                for f in list(self._written):
                    self._field_steps.setdefault(f, [])
                    if self._current_step not in self._field_steps[f]:
                        self._field_steps[f].append(self._current_step)
                continue
            target = alias.upper()
            sources = _vars_in_expr(expr)
            self._written.add(target)
            # Only count a source as a real read if it is not the field itself
            # (bare passthrough T1.FOO AS FOO resolves to source==target — skip)
            real_sources = sources - {target}
            self._read.update(real_sources)
            self._field_steps.setdefault(target, [])
            if self._current_step not in self._field_steps[target]:
                self._field_steps[target].append(self._current_step)
            for src in real_sources:
                self._add_edge(src, target, "assigns", expr)

    def _walk_body(self, body: list[AnyNode]) -> None:
        for node in body:
            self._walk_node(node)

    def _walk_node(self, node: AnyNode) -> None:
        if isinstance(node, AssignNode):
            self._on_assign(node)
        elif isinstance(node, IfNode):
            self._on_if(node)
        elif isinstance(node, FilterNode):
            self._on_filter(node)
        elif isinstance(node, DoLoopNode):
            self._on_do_loop(node)
        elif isinstance(node, SelectNode):
            self._on_select(node)
        elif isinstance(node, RetainNode):
            for v in node.vars:
                self._written.add(v.upper())
        elif isinstance(node, ArrayNode):
            # Array vars are read/written in the loop body context
            for v in node.vars:
                if v and not v.startswith("_"):
                    self._written.add(v.upper())

    def _on_assign(self, node: AssignNode) -> None:
        target = node.var.upper()
        sources = _vars_in_expr(node.expr)
        self._written.add(target)
        self._read.update(sources)
        self._field_steps.setdefault(target, [])
        if self._current_step not in self._field_steps[target]:
            self._field_steps[target].append(self._current_step)
        for src in sources:
            self._add_edge(src, target, "assigns", node.expr)
        for cond_vars in self._cond_stack:
            for cv in cond_vars:
                if cv != target:
                    self._add_edge(cv, target, "conditions", "")

    def _on_if(self, node: IfNode) -> None:
        cond_vars = _vars_in_expr(node.condition)
        self._read.update(cond_vars)
        self._cond_stack.append(cond_vars)
        self._walk_body(node.then_branch)
        self._walk_body(node.else_branch)
        self._cond_stack.pop()

    def _on_filter(self, node: FilterNode) -> None:
        self._read.update(_vars_in_expr(node.condition))

    def _on_do_loop(self, node: DoLoopNode) -> None:
        # Loop variable is written
        if node.var:
            self._written.add(node.var.upper())
        # Conditions reference vars
        for cond in [node.while_cond, node.until_cond]:
            if cond:
                self._read.update(_vars_in_expr(cond))
        if node.start:
            self._read.update(_vars_in_expr(node.start))
        if node.stop:
            self._read.update(_vars_in_expr(node.stop))
        self._walk_body(node.body)

    def _on_select(self, node: SelectNode) -> None:
        # Collect all vars that guard this SELECT block — the select expression
        # plus any condition variables inside WHEN clauses.  Push them onto the
        # condition stack so that assignments inside WHEN bodies record these as
        # conditioning ancestors (same mechanism as _on_if).
        guard_vars: set[str] = set()
        if node.select_expr:
            guard_vars.update(_vars_in_expr(node.select_expr))
            self._read.update(guard_vars)
        self._cond_stack.append(guard_vars)
        for when in node.whens:
            when_guard: set[str] = set()
            for v in when.values:
                when_guard.update(_vars_in_expr(v))
            self._read.update(when_guard)
            self._cond_stack.append(when_guard)
            self._walk_body(when.body)
            self._cond_stack.pop()
        self._walk_body(node.otherwise)
        self._cond_stack.pop()

    def _add_edge(self, source: str, target: str, kind: str, expr: str) -> None:
        key = (source, target, kind, self._current_step)
        if key in self._seen_edges:
            return
        self._seen_edges.add(key)
        self._edges.append({
            "source": source, "target": target, "kind": kind,
            "data_step": self._current_step,
            "expr": expr[:120] if expr else "",
        })

    def _build(self) -> FieldLineage:
        all_fields = self._written | self._read
        step_idx = {s: i for i, s in enumerate(self._step_order)}

        # Fields that have at least one real incoming edge
        has_real_source: set[str] = {e["target"] for e in self._edges}

        raw_layers: dict[str, int] = {}
        for f in all_fields:
            if f in self._written:
                steps = self._field_steps.get(f, [])
                first_step = steps[0] if steps else ""
                raw_layers[f] = step_idx.get(first_step, 0) + 1
            else:
                raw_layers[f] = 0
        used = sorted(set(raw_layers.values()))
        remap = {old: new for new, old in enumerate(used)}

        nodes = []
        for f in sorted(all_fields):
            if f in self._written:
                if f not in has_real_source:
                    # Written by a step but no real source edges — opaque origin
                    kind = "input"
                elif f in self._read:
                    kind = "modified"
                else:
                    kind = "computed"
                steps = self._field_steps.get(f, [])
            else:
                kind = "input"
                steps = []
            nodes.append({
                "id": f, "label": f, "kind": kind,
                "layer": remap[raw_layers[f]],
                "data_steps": steps,
            })

        return FieldLineage(nodes=nodes, edges=self._edges, data_steps=self._step_order)


def trace_field_ancestors(
    lineage: FieldLineage,
    target: str,
    *,
    max_depth: int | None = None,
) -> dict:
    """Walk the lineage graph *backwards* from ``target``.

    Edges are directed ``source → target`` (source feeds into target).  This
    returns every transitive predecessor of ``target``, grouped by hop
    distance, plus a subgraph containing only those nodes and the edges
    between them.
    """
    target_u = target.upper()
    all_ids = {n["id"].upper() for n in lineage.nodes}

    preds: dict[str, list[dict]] = {}
    for e in lineage.edges:
        t = e["target"].upper()
        preds.setdefault(t, []).append({
            "source": e["source"].upper(),
            "target": t,
            "kind": e.get("kind", "assigns"),
            "data_step": e.get("data_step", ""),
            "expr": e.get("expr", ""),
        })

    # BFS upward: depth 0 = target, depth 1 = direct inputs, …
    depth: dict[str, int] = {target_u: 0}
    frontier = [target_u]
    while frontier:
        nxt: list[str] = []
        for node in frontier:
            for edge in preds.get(node, []):
                src = edge["source"]
                if src in depth:
                    continue
                d = depth[node] + 1
                if max_depth is not None and d > max_depth:
                    continue
                depth[src] = d
                nxt.append(src)
        frontier = nxt

    ancestor_set = {f for f, d in depth.items() if d > 0}
    keep = set(depth.keys())

    filtered_nodes = [n for n in lineage.nodes if n["id"].upper() in keep]
    filtered_edges = [
        e for e in lineage.edges
        if e["source"].upper() in keep and e["target"].upper() in keep
    ]

    layers: list[dict] = []
    by_depth: dict[int, list[str]] = {}
    for f, d in depth.items():
        by_depth.setdefault(d, []).append(f)
    for d in sorted(by_depth):
        layers.append({
            "depth": d,
            "fields": sorted(by_depth[d]),
        })

    direct = preds.get(target_u, [])

    return {
        "target": target_u,
        "found": target_u in all_ids or bool(direct),
        "ancestors": sorted(ancestor_set),
        "ancestor_count": len(ancestor_set),
        "direct_predecessors": direct,
        "depth": {k: v for k, v in sorted(depth.items())},
        "layers": layers,
        "nodes": filtered_nodes,
        "edges": filtered_edges,
        "data_steps": lineage.data_steps,
    }


def _parse_schema_columns(schema_path: Path) -> dict[str, set[str]]:
    """Parse a SQL DDL file and return {TABLE_NAME: {COL1, COL2, ...}}."""
    text = schema_path.read_text(encoding="utf-8")
    result: dict[str, set[str]] = {}
    for block in re.split(r"(?=CREATE TABLE)", text, flags=re.IGNORECASE):
        m = re.match(r"CREATE TABLE\s+(\w+)", block, re.IGNORECASE)
        if not m:
            continue
        tname = m.group(1).upper()
        cols: set[str] = set()
        # Extract column names from lines like "  col_name  TYPE  ..."
        for line in block.split("\n"):
            line = line.strip()
            if line.startswith("--") or line.startswith("/*") or not line:
                continue
            cm = re.match(r"(\w+)\s+(?:VARCHAR|TEXT|NUMERIC|INTEGER|INT|DECIMAL|DATE|TIMESTAMP|BOOLEAN|SERIAL|BIGINT|SMALLINT|REAL|DOUBLE|FLOAT|CHAR)", line, re.IGNORECASE)
            if cm:
                cols.add(cm.group(1).upper())
        if cols:
            result[tname] = cols
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class SASLogicTree:
    """Parse SAS DATA step code into a logic tree and simulate it with example values."""

    def __init__(self) -> None:
        self._uninterpreted: list[dict] = []

    @property
    def uninterpreted(self) -> list[dict]:
        """List of statements the parser did not recognise.

        Each entry has the keys ``line`` (1-indexed line number in the
        pre-processed source), ``statement`` (the statement text, truncated
        to 120 chars), and ``context`` (``"top_level"`` or ``"body"``).
        """
        return list(self._uninterpreted)

    def parse(self, code: str) -> list[AnyNode]:
        clean = _strip_comments(code)
        # Expand SQL-fragment macros (no semicolons in body) before statement
        # splitting so their SELECT-field expressions become visible to the
        # PROC SQL field extractor.
        sql_macros = _collect_sql_fragment_macros(clean)
        clean = _expand_sql_fragment_macros(clean, sql_macros)
        # Inline program-macro calls, resolving table-name parameters (e.g.
        # &PREFIJO_PROGRAMA.) to the literal values passed at each call site.
        clean = _expand_called_macros(clean)
        clean = _strip_macro_refs(clean)
        stmts_with_lines = _split_statements_with_lines(clean)
        parser = _Parser(stmts_with_lines)
        nodes = parser.parse()
        self._uninterpreted = parser._uninterpreted
        return nodes

    def display(self, nodes: list[AnyNode]) -> str:
        return "\n".join(n.display() for n in nodes)

    def to_dict(self, nodes: list[AnyNode]) -> list[dict]:
        return [n.to_dict() for n in nodes]

    def lineage(self, nodes: list[AnyNode]) -> FieldLineage:
        return _LineageWalker().walk(nodes)

    def trace_lineage(self, nodes: list[AnyNode], target: str, *, max_depth: int | None = None) -> dict:
        lg = self.lineage(nodes)
        return trace_field_ancestors(lg, target, max_depth=max_depth)

    def evaluate(self, nodes: list[AnyNode], values: dict[str, Any]) -> EvalTrace:
        env = dict(values)
        ev = _Evaluator()
        ev.run(nodes, env)
        return EvalTrace(
            initial=dict(values), final=env,
            steps=ev.steps,
            row_passes_filter=ev.row_passes_filter,
            filter_results=ev.filter_results,
        )

    def validate(
        self,
        nodes: list[AnyNode],
        schema_path: str | Path | None = None,
    ) -> list[dict[str, str]]:
        """Run static checks on parsed AST and return diagnostics.

        Checks: undefined variables, missing datasets, type heuristics,
        and optionally column validation against a DDL schema file.
        """
        diags: list[dict[str, str]] = []
        lg = self.lineage(nodes)
        node_map = {n["id"]: n for n in lg.nodes}

        # 1. Variables used in expressions that shadow or duplicate names.
        #    Without the input dataset schema we cannot distinguish
        #    legitimate SET inputs from truly undefined variables, so this
        #    check is deferred to schema validation (check 4 below).

        # 2. Missing datasets — input datasets not produced by prior steps
        produced: set[str] = set()
        for n in nodes:
            if isinstance(n, DataStepNode):
                produced.add(n.output_dataset.upper())
            elif isinstance(n, ProcNode) and n.output_table:
                produced.add(n.output_table.upper())
        for n in nodes:
            if isinstance(n, DataStepNode):
                all_inputs = [n.input_dataset] + n.merge_datasets
                for ds in all_inputs:
                    if not ds:
                        continue
                    ds_u = ds.upper()
                    if ds_u not in produced and "MVAR[" not in ds_u:
                        diags.append({
                            "level": "info", "code": "MISSING_DATASET",
                            "dataset": ds,
                            "message": f"Dataset {ds} is read but not created in this code",
                        })

        # 3. Type heuristics — fields in arithmetic AND compared to string
        numeric_ctx: set[str] = set()
        string_ctx: set[str] = set()
        for e in lg.edges:
            expr = e.get("expr", "")
            src = e["source"]
            if any(op in expr for op in ("+", "-", "*", "/", "**")):
                numeric_ctx.add(src)
            if "'" in expr or '"' in expr:
                string_ctx.add(src)
        for f in sorted(numeric_ctx & string_ctx):
            diags.append({
                "level": "warning", "code": "TYPE_MISMATCH", "field": f,
                "message": f"Variable {f} used in both numeric and string contexts",
            })

        # 4. Schema validation (optional)
        if schema_path:
            schema_path = Path(schema_path)
            if schema_path.exists():
                schema_cols = _parse_schema_columns(schema_path)
                for n in nodes:
                    if isinstance(n, ProcNode) and n.kind == "SQL" and n.input_tables:
                        for tbl in n.input_tables:
                            tbl_clean = tbl.split(".")[-1].upper()
                            if tbl_clean in schema_cols:
                                valid_cols = schema_cols[tbl_clean]
                                for alias, expr in n.select_fields:
                                    if alias == "*":
                                        continue
                                    for v in _vars_in_expr(expr):
                                        if v not in valid_cols and v not in produced:
                                            diags.append({
                                                "level": "warning",
                                                "code": "SCHEMA_MISMATCH",
                                                "field": v, "table": tbl_clean,
                                                "message": f"Column {v} not found in schema table {tbl_clean}",
                                            })
        return diags
