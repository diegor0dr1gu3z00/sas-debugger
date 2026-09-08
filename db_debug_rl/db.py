"""SQLite helpers for db-debug-rl (external production databases)."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SQL_DIR = ROOT / "data" / "external" / "sql"


def open_db(name: str, read_only: bool = False) -> sqlite3.Connection:
    path = SQL_DIR / f"{name}.db"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run scripts/build_validation_dbs.py first")
    if read_only:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(path)
    conn.execute("PRAGMA query_only=ON" if read_only else "PRAGMA foreign_keys=OFF")
    return conn


def schema_summary(conn: sqlite3.Connection) -> dict[str, list[str]]:
    schema: dict[str, list[str]] = {}
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'zz\\_%' ESCAPE '\\' "
        "ORDER BY name")]
    for t in tables:
        cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{t}")')]
        schema[t] = cols
    return schema


def schema_text(schema: dict[str, list[str]], max_cols: int = 24) -> str:
    lines = ["TABLES PRESENT (table: columns):"]
    for table in sorted(schema):
        cols = ", ".join(schema[table][:max_cols])
        if len(schema[table]) > max_cols:
            cols += ", ..."
        lines.append(f"  {table}  ({cols})")
    return "\n".join(lines)


def exec_sql(conn: sqlite3.Connection, sql: str, max_rows: int = 20,
             timeout_s: float = 20.0) -> tuple[str, Any]:
    """Run a read-only statement; return ('rows', (cols, rows)) or ('err', msg)."""
    if not sql or not sql.strip():
        return ("err", "empty query")
    body = sql.strip().rstrip(";").strip()
    if not re.match(r"(?is)^(select|with)\b", body):
        return ("err", "only SELECT/WITH statements are allowed")
    if ";" in body:
        return ("err", "one statement at a time (no ';')")
    if not re.search(r"(?is)\blimit\s+\d+\s*$", body):
        body = f"{body}\nLIMIT {max_rows}"

    state = {"n": 0}

    def _progress() -> int:
        state["n"] += 1
        return 1 if state["n"] > timeout_s * 5_000 else 0

    conn.set_progress_handler(_progress, 2_000)
    try:
        cur = conn.execute(body)
        rows = cur.fetchmany(max_rows + 1)
        cols = [d[0] for d in cur.description] if cur.description else []
        truncated = len(rows) > max_rows
        return ("rows", {"cols": cols, "rows": [tuple(r) for r in rows[:max_rows]],
                         "n_rows": len(rows[:max_rows]),
                         "truncated": truncated})
    except Exception as exc:  # noqa: BLE001
        return ("err", f"{type(exc).__name__}: {exc}")
    finally:
        conn.set_progress_handler(None, 0)


def readonly_guard(conn: sqlite3.Connection) -> None:
    """Install an authorizer that denies any write/DDL/attach action."""
    deny = {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE,
            sqlite3.SQLITE_DROP_TABLE, sqlite3.SQLITE_DROP_INDEX,
            sqlite3.SQLITE_DROP_VIEW, sqlite3.SQLITE_DROP_TRIGGER,
            sqlite3.SQLITE_CREATE_TABLE, sqlite3.SQLITE_CREATE_INDEX,
            sqlite3.SQLITE_CREATE_VIEW, sqlite3.SQLITE_CREATE_TRIGGER,
            sqlite3.SQLITE_ALTER_TABLE, sqlite3.SQLITE_REINDEX,
            sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH,
            sqlite3.SQLITE_SAVEPOINT}

    def _auth(action: int, arg1: str | None, arg2: str | None,
              dbname: str | None, source: str | None) -> int:
        return sqlite3.SQLITE_DENY if action in deny else sqlite3.SQLITE_OK

    conn.set_authorizer(_auth)
