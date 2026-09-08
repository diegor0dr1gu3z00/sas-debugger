#!/usr/bin/env python3
"""Build the generalization/validation database set for db-debug-rl.

Downloads 10+ public production-style databases (SQLite, MySQL dumps, TPC
benchmarks) plus real SAS (sas7bdat) files, converts everything to directly
usable SQLite databases under data/external/sql, and writes a manifest +
README with per-DB table/row inventory.

Usage:
    .venv/bin/python scripts/build_validation_dbs.py            # full build
    .venv/bin/python scripts/build_validation_dbs.py world f1   # subset
"""
from __future__ import annotations

import gzip
import json
import re
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "data" / "external"
DL = EXT / "downloads"
SQL = EXT / "sql"
SAS = EXT / "sas" / "sas7bdat"

MYSQL_DUMPS = {
    "world": "https://downloads.mysql.com/docs/world-db.zip",
    "sakila": "https://downloads.mysql.com/docs/sakila-db.zip",
    "menagerie": "https://downloads.mysql.com/docs/menagerie-db.zip",
}
DIRECT_SQLITE = {
    "chinook": "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite",
    "northwind": "https://github.com/jpwhite3/northwind-SQLite3/raw/main/dist/northwind.db",
    "sakila_sqlite": "https://github.com/bradleygrant/sakila-sqlite3/raw/main/sakila_master.db",
}
URL_ZIPS = {
    "classicmodels": "https://www.mysqltutorial.org/wp-content/uploads/2018/03/mysqlsampledatabase.zip",
}
URL_SQLITE_ZIPS = {
    "f1db": "https://github.com/f1db/f1db/releases/download/v2026.13.0/f1db-sqlite.zip",
}
SAS_FILES = [
    "airline.sas7bdat", "cars.sas7bdat", "productsales.sas7bdat",
    "datetime.sas7bdat", "dates_null.sas7bdat", "max_sas_date.sas7bdat",
    "test1.sas7bdat", "test2.sas7bdat", "test3.sas7bdat", "test5.sas7bdat",
    "test8.sas7bdat", "test9.sas7bdat", "test10.sas7bdat", "test11.sas7bdat",
    "test12.sas7bdat", "0x40controlbyte.sas7bdat",
]
SAS_BASE = "https://raw.githubusercontent.com/pandas-dev/pandas/main/pandas/tests/io/sas/data/"
LICENSES = {
    "chinook": "BSD-3 (lerocha/chinook-database)",
    "northwind": "Microsoft sample, MIT-licensed port (jpwhite3/northwind-SQLite3)",
    "sakila_sqlite": "Sakila sample port (bradleygrant/sakila-sqlite3)",
    "world": "MySQL sample data (dev.mysql.com)",
    "sakila": "MySQL sample data (dev.mysql.com)",
    "menagerie": "MySQL sample data (dev.mysql.com)",
    "classicmodels": "mysqltutorial.org sample database",
    "f1db": "F1 racing data (f1db, GPL/CC-derived releases)",
    "openflights": "OpenFlights.org data (ODbL)",
    "imdb": "IMDb datasets (title.basics/ratings/episode subset, non-commercial)",
    "tpch": "TPC-H benchmark, generated locally (fair use, no commercial benchmarking)",
    "tpcds": "TPC-DS benchmark, generated locally (fair use, no commercial benchmarking)",
    "sas7bdat": "SAS sample data files (pandas-dev/pandas test corpus, real-world SAS files)",
}


def sh_curl(url: str, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists() or out.stat().st_size == 0:
        print(f"  curl {url}")
        subprocess.run(["curl", "-sSL", "--retry", "3", "--max-time", "600",
                        "-o", str(out), url], check=True)
    return out


# ── mysqldump → SQLite ───────────────────────────────────────────────────────

_COND_RE = re.compile(r"/\*!\d+[^*]*?\*/\s*;?", re.DOTALL)
_SKIP_STMT = re.compile(
    r"^\s*(SET|LOCK|UNLOCK|START\s+TRANSACTION|COMMIT|BEGIN|USE|DELIMITER|"
    r"DROP\s+(FUNCTION|PROCEDURE|VIEW|TRIGGER)|CREATE\s+(FUNCTION|PROCEDURE|"
    r"VIEW|TRIGGER|DEFINER)|ALTER\s+TABLE|CREATE\s+(UNIQUE\s+)?INDEX|"
    r"RENAME\s+TABLE)\b", re.IGNORECASE)
_GEN_ALWAYS = ("GENERATED ALWAYS", "AS (")
_DROP_DEF = re.compile(
    r"\bAUTO_INCREMENT\b|\bUNSIGNED\b|\bZEROFILL\b|\bCHARACTER SET\s+\w+|"
    r"\bCOLLATE\s+\w+|\bON UPDATE CURRENT_TIMESTAMP(\(\d*\))?|"
    r"\bCOMMENT\s+'(?:[^'\\]|\\.|'')*'", re.IGNORECASE)
_ENUM_RE = re.compile(r"\bENUM\s*\([^)]*\)|\bSET\s*\([^)]*\)", re.IGNORECASE)


def _split_statements(text: str):
    """Quote-aware split on ';'. Handles \\-escapes inside strings."""
    stmt, i, n = [], 0, len(text)
    in_s = in_d = in_b = False
    while i < n:
        c = text[i]
        if in_s:
            if c == "\\" and i + 1 < n:
                stmt.append(c); stmt.append(text[i + 1]); i += 2; continue
            if c == "'":
                if i + 1 < n and text[i + 1] == "'":
                    stmt.append("''"); i += 2; continue
                in_s = False
            stmt.append(c)
        elif in_d:
            if c == "\\" and i + 1 < n:
                stmt.append(c); stmt.append(text[i + 1]); i += 2; continue
            if c == '"':
                in_d = False
            stmt.append(c)
        elif in_b:
            if c == "`":
                in_b = False
            stmt.append(c)
        else:
            if c == "'":
                in_s = True; stmt.append(c)
            elif c == '"':
                in_d = True; stmt.append(c)
            elif c == "`":
                in_b = True; stmt.append(c)
            elif c == ";":
                yield "".join(stmt)
                stmt = []
            else:
                stmt.append(c)
        i += 1
    tail = "".join(stmt).strip()
    if tail:
        yield tail


_UNESCAPE_RE = re.compile(r"\\(.)")
_LEADING = re.compile(r"^(?:/\*.*?\*/|--[^\n]*|\s)+", re.DOTALL)


def _strip_leading(s: str) -> str:
    while True:
        new = _LEADING.sub("", s, count=1)
        if new == s:
            return s.strip()
        s = new


def _unescape(stmt: str) -> str:
    """MySQL dump escapes → SQLite literals (only backslashes exist in dumps)."""
    return _UNESCAPE_RE.sub(
        lambda m: {"'": "''", '"': '"', "\\": "\\", "n": "\n", "r": "\r",
                   "0": "", "Z": "\x1a", "%": "\\%", "_": "\\_"}.get(
                       m.group(1), m.group(1)),
        stmt)


def _rewrite_create(stmt: str) -> str | None:
    stmt = re.sub(r"/\*!\d+.*?\*/", " ", stmt, flags=re.DOTALL)
    stmt = re.sub(r"/\*.*?\*/", " ", stmt, flags=re.DOTALL)
    stmt = re.sub(r"(?m)^\s*--.*$", "", stmt)
    m = re.match(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[\"'`]?([A-Za-z0-9_]+)[\"'`]?\s*\(",
                 stmt, re.IGNORECASE | re.DOTALL)
    if not m:
        return stmt
    name = m.group(1)
    depth, start = 0, m.end() - 1
    for i in range(start, len(stmt)):
        if stmt[i] == "(":
            depth += 1
        elif stmt[i] == ")":
            depth -= 1
            if depth == 0:
                body, _rest = stmt[start + 1:i], stmt[i + 1:]
                break
    else:
        return None
    items, buf, depth, in_s = [], [], 0, False
    i = 0
    while i < len(body):
        c = body[i]
        if in_s:
            if c == "\\" and i + 1 < len(body):
                buf.append(body[i:i + 2]); i += 2; continue
            if c == "'":
                in_s = False
            buf.append(c)
        else:
            if c == "'":
                in_s = True; buf.append(c)
            elif c == "(":
                depth += 1; buf.append(c)
            elif c == ")":
                depth -= 1; buf.append(c)
            elif c == "," and depth == 0:
                items.append("".join(buf)); buf = []
            else:
                buf.append(c)
        i += 1
    items.append("".join(buf))
    kept = []
    for it in items:
        s = it.strip()
        if not s:
            continue
        head = re.match(r"^(PRIMARY KEY|UNIQUE KEY|UNIQUE|KEY|INDEX|FULLTEXT|"
                        r"SPATIAL|CONSTRAINT)\b", s, re.IGNORECASE)
        if head:
            if head.group(1).upper() == "PRIMARY KEY":
                pk_cols = re.search(r"\((.*)\)\s*$", s, re.DOTALL)
                kept.append("PRIMARY KEY (" + pk_cols.group(1) + ")" if pk_cols
                            else "PRIMARY KEY (" + s[11:].strip(" ()") + ")")
            continue
        if any(k in s.upper() for k in _GEN_ALWAYS):
            continue
        s = _ENUM_RE.sub("TEXT", s)
        s = _DROP_DEF.sub("", s)
        s = re.sub(r"\s+", " ", s).strip().rstrip(",").strip()
        if s:
            kept.append(s)
    if not kept:
        return None
    return f'CREATE TABLE "{name}" ({", ".join(kept)})'


def mysql_dump_to_sqlite(dump_text: str, out: Path, table_filter=None) -> dict:
    conn = sqlite3.connect(out)
    conn.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;")
    n_tables = n_inserts = skipped = 0
    for raw in _split_statements(dump_text):
        s = _strip_leading(raw)
        if not s or s.startswith("--"):
            continue
        s = _COND_RE.sub("", s).strip()
        if not s:
            continue
        if _SKIP_STMT.match(s):
            skipped += 1
            continue
        if re.match(r"CREATE\s+TABLE", s, re.IGNORECASE):
            new = _rewrite_create(s)
            if new is None:
                skipped += 1
                continue
            if table_filter and not table_filter(new):
                skipped += 1
                continue
            try:
                conn.execute(new)
                n_tables += 1
            except sqlite3.Error as exc:
                print(f"    ! create failed: {exc}: {new[:120]}")
                skipped += 1
            continue
        if re.match(r"INSERT\s+INTO", s, re.IGNORECASE):
            t = _unescape(s)
            try:
                conn.execute(t)
                n_inserts += 1
            except sqlite3.Error as exc:
                print(f"    ! insert failed: {exc}: {t[:120]}")
                skipped += 1
            continue
        skipped += 1
    conn.commit()
    conn.close()
    return {"tables": n_tables, "inserts": n_inserts, "skipped": skipped}


def load_menagerie_special(sql_dir: Path) -> None:
    """menagerie zip ships CREATE + LOAD-DATA-era INSERT files; handle both."""
    pass  # handled by generic path (its zips contain plain INSERT dumps)


# ── duckdb TPC benchmarks → SQLite ──────────────────────────────────────────

def bench_to_sqlite(bench: str, sf: float, out: Path) -> dict:
    import decimal
    import datetime as _dt
    import duckdb
    sqlite3.register_adapter(decimal.Decimal, lambda d: float(d))
    sqlite3.register_adapter(_dt.date, lambda d: d.isoformat())
    sqlite3.register_adapter(_dt.datetime, lambda d: d.isoformat(sep=" "))
    sqlite3.register_adapter(_dt.time, lambda t: t.isoformat())
    sqlite3.register_adapter(_dt.timedelta, lambda v: str(v))
    out.unlink(missing_ok=True)
    con = duckdb.connect()
    con.execute(f"INSTALL {bench}; LOAD {bench};")
    con.execute(f"CALL {'dbgen' if bench == 'tpch' else 'dsdgen'}(sf={sf});")
    tables = [r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema='main' ORDER BY table_name").fetchall()]
    lite = sqlite3.connect(out)
    lite.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;")
    counts = {}
    for t in tables:
        cols = [d[0] for d in con.execute(f'SELECT * FROM "{t}" LIMIT 0').description]
        lite.execute('CREATE TABLE "{}" ({})'.format(
            t, ", ".join(f'"{c}"' for c in cols)))
        cur = con.execute(f'SELECT * FROM "{t}"')
        rows = 0
        while True:
            batch = cur.fetchmany(50_000)
            if not batch:
                break
            lite.executemany(
                'INSERT INTO "{}" VALUES ({})'.format(t, ",".join("?" * len(cols))),
                [tuple(r) for r in batch])
            rows += len(batch)
        counts[t] = rows
    lite.commit()
    lite.close()
    con.close()
    return {"tables": len(tables), "rows": counts}


# ── CSV / TSV loaders (real-world production datasets) ──────────────────────

OPENFLIGHTS = {
    "url_base": "https://raw.githubusercontent.com/jpatokal/openflights/master/data/",
    "files": {
        "airlines": ("airlines.dat", ["airline_id", "name", "alias", "iata", "icao",
                                      "callsign", "country", "active"]),
        "airports": ("airports.dat", ["airport_id", "name", "city", "country", "iata",
                                      "icao", "latitude", "longitude", "altitude",
                                      "tz_offset", "dst", "tz", "type", "source"]),
        "routes": ("routes.dat", ["airline", "airline_id", "src_airport", "src_airport_id",
                                  "dst_airport", "dst_airport_id", "codeshare", "stops",
                                  "equipment"]),
        "planes": ("planes.dat", ["name", "iata", "icao"]),
        "countries": ("countries.dat", ["name", "iso", "dafif", "wikipedia", "keywords"]),
    },
}
IMDB_BASE = "https://datasets.imdbws.com/"


def csv_to_sqlite(url: str, out: Path, table: str, cols: list[str], delim: str,
                  gz: bool = False, row_filter=None) -> int:
    import csv as _csv
    raw = sh_curl(url, DL / Path(url).name)
    fh = gzip.open(raw, "rt", encoding="utf-8", errors="replace", newline="") if gz \
        else open(raw, encoding="utf-8", errors="replace", newline="")
    conn = sqlite3.connect(out)
    conn.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({", ".join(cols)})')
    n = 0
    reader = _csv.reader(fh, delimiter=delim)
    for row in reader:
        if row_filter and not row_filter(row):
            continue
        try:
            conn.execute(
                f'INSERT INTO "{table}" VALUES ({",".join("?" * len(cols))})', row)
            n += 1
        except (sqlite3.Error, ValueError):
            pass
    fh.close()
    conn.commit()
    conn.close()
    return n


def load_imdb(out: Path) -> dict:
    """Movies/series subset of the official IMDb datasets (join-heavy, real data)."""
    conn = sqlite3.connect(out)
    conn.executescript("""
      CREATE TABLE title_basics (tconst TEXT, title_type TEXT, primary_title TEXT,
        original_title TEXT, is_adult INT, start_year INT, end_year INT,
        runtime_minutes INT, genres TEXT);
      CREATE TABLE title_ratings (tconst TEXT, average_rating REAL, num_votes INT);
      CREATE TABLE title_episode (tconst TEXT, parent_tconst TEXT,
        season_number INT, episode_number INT);
    """)
    keep: set[str] = set()
    raw = sh_curl(IMDB_BASE + "title.basics.tsv.gz", DL / "title.basics.tsv.gz")
    with gzip.open(raw, "rt", encoding="utf-8", newline="") as fh:
        r = _csv_skip(fh)
        n1 = 0
        for row in r:
            if row[1] == "movie" and row[5] >= "2000":
                keep.add(row[0])
                conn.execute('INSERT INTO title_basics VALUES (?,?,?,?,?,?,?,?,?)',
                             [_x(v) for v in row])
                n1 += 1
            elif row[1] == "tvSeries":
                keep.add(row[0])
                conn.execute('INSERT INTO title_basics VALUES (?,?,?,?,?,?,?,?,?)',
                             [_x(v) for v in row])
                n1 += 1
    raw = sh_curl(IMDB_BASE + "title.ratings.tsv.gz", DL / "title.ratings.tsv.gz")
    n2 = 0
    with gzip.open(raw, "rt", encoding="utf-8", newline="") as fh:
        for row in _csv_skip(fh):
            if row[0] in keep:
                conn.execute('INSERT INTO title_ratings VALUES (?,?,?)',
                             [_x(v) for v in row])
                n2 += 1
    raw = sh_curl(IMDB_BASE + "title.episode.tsv.gz", DL / "title.episode.tsv.gz")
    n3 = 0
    with gzip.open(raw, "rt", encoding="utf-8", newline="") as fh:
        for row in _csv_skip(fh):
            if row[1] in keep:
                conn.execute('INSERT INTO title_episode VALUES (?,?,?,?)',
                             [_x(v) for v in row])
                n3 += 1
    conn.commit()
    conn.execute('CREATE INDEX IF NOT EXISTS ix_basics_tconst ON title_basics(tconst);')
    conn.execute('CREATE INDEX IF NOT EXISTS ix_ratings_tconst ON title_ratings(tconst);')
    conn.execute('CREATE INDEX IF NOT EXISTS ix_episode_parent ON title_episode(parent_tconst);')
    conn.execute('CREATE INDEX IF NOT EXISTS ix_episode_tconst ON title_episode(tconst);')
    conn.close()
    return {"titles": n1, "ratings": n2, "episodes": n3}


def _csv_skip(fh):
    import csv as _csv
    return _csv.reader(fh, delimiter="\t")


def _x(v: str):
    if v == "\\N" or v == "":
        return None
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


# ── verification / manifest ─────────────────────────────────────────────────

def inventory(path: Path) -> dict:
    conn = sqlite3.connect(path)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    counts = {}
    for t in tables:
        try:
            counts[t] = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        except sqlite3.Error:
            counts[t] = -1
    conn.close()
    return {"file": str(path.relative_to(EXT)), "size_bytes": path.stat().st_size,
            "tables": counts, "n_tables": len(tables),
            "total_rows": sum(v for v in counts.values() if v >= 0)}


def main() -> None:
    wanted = set(sys.argv[1:]) or None
    for d in (DL, SQL, SAS):
        d.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}

    def want(db: str) -> bool:
        return wanted is None or db in wanted

    # direct SQLite
    for name, url in DIRECT_SQLITE.items():
        if not want(name):
            continue
        print(f"[{name}] direct sqlite")
        p = sh_curl(url, DL / f"{name}.sqlite")
        target = SQL / f"{name}.db"
        shutil.copyfile(p, target)
        inv = inventory(target)
        manifest[name] = {"source": url, "kind": "sqlite", "license": LICENSES[name], **inv}

    # MySQL official zips
    for name, url in MYSQL_DUMPS.items():
        if not want(name):
            continue
        print(f"[{name}] mysql zip → sqlite")
        z = sh_curl(url, DL / f"{name}-db.zip")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(DL / f"{name}_x")
        target = SQL / f"{name}.db"
        target.unlink(missing_ok=True)
        agg = {"tables": 0, "inserts": 0, "skipped": 0}
        files = sorted((DL / f"{name}_x").rglob("*.sql"),
                       key=lambda p: (0 if "schema" in p.name.lower() or
                                      "structure" in p.name.lower() else 1,
                                      p.name))
        for sqlf in files:
            print(f"  converting {sqlf.name}")
            st = mysql_dump_to_sqlite(sqlf.read_text("utf-8", "replace"), target)
            for k in agg:
                agg[k] += st[k]
        # LOAD-DATA-era TSVs shipped alongside the dumps (pet.txt, event.txt)
        MENAGERIE_COLS = {
            "pet": ["name", "owner", "species", "sex", "birth", "death"],
            "event": ["name", "date", "type", "remark"],
        }
        for tsv in sorted((DL / f"{name}_x").rglob("*.txt")):
            table = tsv.stem.lower()
            header = MENAGERIE_COLS.get(table)
            if header is None:
                continue
            with tsv.open(encoding="utf-8", errors="replace") as fh:
                rows = [tuple(_x(v) for v in line.rstrip("\n").split("\t"))
                        for line in fh if line.strip()]
            rows = [r + (None,) * (len(header) - len(r)) for r in rows]
            rows = [r[:len(header)] for r in rows]
            conn = sqlite3.connect(target)
            conn.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({", ".join(header)})')
            conn.executemany(f'INSERT INTO "{table}" VALUES ({",".join("?" * len(header))})',
                             rows)
            conn.commit()
            conn.close()
            print(f"  {tsv.name}: {len(rows)} rows")
        assert target.exists() and agg["tables"], f"{name} conversion failed"
        manifest[name] = {"source": url, "kind": "mysql_dump", "license": LICENSES[name],
                          "convert": agg, **inventory(target)}

    # classicmodels zip (mysqltutorial)
    if want("classicmodels"):
        print("[classicmodels] zip → sqlite")
        z = sh_curl(URL_ZIPS["classicmodels"], DL / "classicmodels.zip")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(DL / "classicmodels_x")
        target = SQL / "classicmodels.db"
        target.unlink(missing_ok=True)
        sqlf = next((DL / "classicmodels_x").rglob("*.sql"))
        st = mysql_dump_to_sqlite(sqlf.read_text("utf-8", "replace"), target)
        assert st["tables"], "classicmodels conversion failed"
        manifest["classicmodels"] = {"source": URL_ZIPS["classicmodels"],
                                     "kind": "mysql_dump", "license": LICENSES["classicmodels"],
                                     "convert": st, **inventory(target)}

    # ready-made SQLite inside a zip (f1db etc.)
    for name, url in URL_SQLITE_ZIPS.items():
        if not want(name):
            continue
        print(f"[{name}] sqlite zip")
        z = sh_curl(url, DL / f"{name}-sqlite.zip")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(DL / f"{name}_x")
        inner = next((p for p in (DL / f"{name}_x").rglob("*")
                      if p.suffix.lower() in {".sqlite", ".db"} and p.is_file()))
        target = SQL / f"{name}.db"
        shutil.copyfile(inner, target)
        manifest[name] = {"source": url, "kind": "sqlite", "license": LICENSES[name],
                          **inventory(target)}

    # TPC benchmarks generated locally with duckdb
    if want("openflights"):
        print("[openflights] CSV → sqlite")
        target = SQL / "openflights.db"
        target.unlink(missing_ok=True)
        for table, (fn, cols) in OPENFLIGHTS["files"].items():
            delim = "\t" if table == "countries" else ","
            n = csv_to_sqlite(OPENFLIGHTS["url_base"] + fn, target, table, cols, delim)
            print(f"  {table}: {n} rows")
        conn = sqlite3.connect(target)
        conn.executescript(
            'CREATE INDEX IF NOT EXISTS ix_routes_airline ON routes(airline);'
            'CREATE INDEX IF NOT EXISTS ix_routes_src ON routes(src_airport_id);'
            'CREATE INDEX IF NOT EXISTS ix_routes_dst ON routes(dst_airport_id);'
            'CREATE INDEX IF NOT EXISTS ix_airports_id ON airports(airport_id);'
            'CREATE INDEX IF NOT EXISTS ix_airlines_id ON airlines(airline_id);')
        conn.commit()
        conn.close()
        manifest["openflights"] = {"source": OPENFLIGHTS["url_base"],
                                   "kind": "csv", "license": LICENSES["openflights"],
                                   **inventory(target)}
    if want("imdb"):
        print("[imdb] TSV subset → sqlite")
        target = SQL / "imdb.db"
        target.unlink(missing_ok=True)
        st = load_imdb(target)
        print(f"  {st}")
        manifest["imdb"] = {"source": IMDB_BASE, "kind": "csv",
                            "license": LICENSES["imdb"], "convert": st,
                            **inventory(target)}
    if want("tpch"):
        print("[tpch] duckdb dbgen sf=0.3 → sqlite")
        st = bench_to_sqlite("tpch", 0.3, SQL / "tpch.db")
        manifest["tpch"] = {"source": "generated (duckdb tpch dbgen)", "kind": "benchmark",
                            "license": LICENSES["tpch"], **inventory(SQL / "tpch.db")}
    if want("tpcds"):
        print("[tpcds] duckdb dsdgen sf=0.02 → sqlite")
        st = bench_to_sqlite("tpcds", 0.02, SQL / "tpcds.db")
        manifest["tpcds"] = {"source": "generated (duckdb tpcds dsdgen)", "kind": "benchmark",
                             "license": LICENSES["tpcds"], **inventory(SQL / "tpcds.db")}

    # real SAS files (sas7bdat)
    if want("sas7bdat"):
        print("[sas7bdat] pandas SAS test corpus")
        import pyreadstat
        n_sas = 0
        for fn in SAS_FILES:
            try:
                p = sh_curl(SAS_BASE + fn, SAS / fn)
                pyreadstat.read_sas7bdat(str(p))
                n_sas += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {fn}: {exc}")
                (SAS / fn).unlink(missing_ok=True)
        assert n_sas > 0, "no sas7bdat files downloaded"
        manifest["sas7bdat"] = {"source": SAS_BASE, "kind": "sas",
                                "license": LICENSES["sas7bdat"],
                                "n_files": n_sas,
                                "files": sorted(p.name for p in SAS.glob("*.sas7bdat"))}

    # integrity gate: every SQL db must have >=2 tables and >0 rows
    for name, m in list(manifest.items()):
        if m.get("kind", "").startswith(("sqlite", "mysql", "benchmark")):
            assert m["n_tables"] >= 2, f"{name}: only {m['n_tables']} tables"
            assert m["total_rows"] > 0, f"{name}: empty"
    (EXT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest)} entries → {EXT/'manifest.json'}")
    for name, m in manifest.items():
        if "n_tables" in m:
            print(f"  {name:14s} {m['n_tables']:3d} tables  {m['total_rows']:>10,d} rows  "
                  f"{m['size_bytes']/1e6:8.1f} MB")
        else:
            print(f"  {name:14s} {m['n_files']} .sas7bdat files")


if __name__ == "__main__":
    main()
