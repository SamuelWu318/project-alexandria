from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Any, Iterable
from utils import SrcPaths, schema   # schema = the scene-record registry (utils/schema.py)

# ---- schema constants (DERIVED from scene_schema.json via schema.py — do not hand-edit) ----
# This module owns HOW the SQLite mirror behaves (upsert / filter / navigate); the field list,
# column types, and indexes all come from the registry so every store stays in lockstep. Edit
# scene_schema.json, then `python -m utils.schema --check` (parity) and `--reconcile` (ripple).
_COLS = schema.SQL_COLS             # upsert column order == registry order
_FILTERABLE = schema.FILTERABLE     # find()/count() WHERE + GROUP BY whitelist (injection guard)
_SCHEMA = schema.create_table_sql()  # CREATE TABLE + indexes, generated from the registry
_INT_COLS = schema.INT_COLS         # INTEGER-affinity cols; _migrate uses them to back-fill


# ---- connection ----

# ** LOCKED **
# Add any _COLS column an older `scenes` table predates (idempotent ALTER; names come from our own whitelist, never caller input).
def _migrate(conn: sqlite3.Connection) -> None:
    have = {r["name"] for r in conn.execute("PRAGMA table_info(scenes)")}
    for col in _COLS:
        if col not in have:
            typ = "INTEGER" if col in _INT_COLS else "TEXT"
            conn.execute(f"ALTER TABLE scenes ADD COLUMN {col} {typ}")


# ** LOCKED **  ** MAIN ** — opened by embed.index_scenes, tests, webtest, schema.sync_db
# Open (creating + migrating) the on-disk scene mirror: self-healing DDL, WAL, dict rows.
def open_db(path: str | Path = SrcPaths.DB_PATH) -> sqlite3.Connection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA)   # CREATE TABLE ... IF NOT EXISTS + indexes
    _migrate(conn)                # ALTER in any column a pre-existing table predates
    return conn


# ---- write ----

# Flatten one record into a column tuple in _COLS order (registry-driven store transforms).
def _to_row(rec: dict) -> tuple:
    return schema.to_row(rec)


# ** MAIN ** — called by embed.index_records and schema.sync_db to mirror every record
# Upsert `records` into the scenes table (INSERT OR REPLACE, one transaction); idempotent on scene_id. Returns rows written.
def sql_upsert(conn: sqlite3.Connection, records: Iterable[dict]) -> int:
    rows = [_to_row(r) for r in records if r.get("scene_id")]   # every record, enriched or not
    if not rows:
        return 0
    placeholders = ",".join("?" * len(_COLS))
    sql = f"INSERT OR REPLACE INTO scenes ({','.join(_COLS)}) VALUES ({placeholders})"
    with conn:                       # commit/rollback as one unit
        conn.executemany(sql, rows)
    return len(rows)


# ---- read (the relational API: exact-match / navigation / aggregate over the scene rows) ----

# Turn a sqlite3.Row into a plain dict, re-inflating the JSON-store columns (descriptors).
def _row(r: sqlite3.Row | None) -> dict | None:
    if r is None:
        return None
    return schema.inflate_row(dict(r))


# ** LOCKED **
# Fetch one scene by id — O(1) PRIMARY KEY lookup, None if absent.
def get(conn: sqlite3.Connection, scene_id: str) -> dict | None:
    cur = conn.execute("SELECT * FROM scenes WHERE scene_id = ?", (scene_id,))
    return _row(cur.fetchone())


# ** LOCKED **
# The window of scenes around `scene_id` in reading order (anchor included) via UNIQUE(book_id, pos).
def neighbors(conn: sqlite3.Connection, scene_id: str,
              before: int = 1, after: int = 1) -> list[dict]:
    anchor = conn.execute(
        "SELECT book_id, pos FROM scenes WHERE scene_id = ?", (scene_id,)
    ).fetchone()
    if anchor is None or anchor["pos"] is None:
        return []
    cur = conn.execute(
        "SELECT * FROM scenes WHERE book_id = ? AND pos BETWEEN ? AND ? ORDER BY pos",
        (anchor["book_id"], anchor["pos"] - before, anchor["pos"] + after),
    )
    return [_row(r) for r in cur.fetchall()]


# ** LOCKED **
# Build a parametrized WHERE from whitelisted kwargs (scalar -> `= ?`, list -> `IN (...)`); unknown key raises.
def _where(filters: dict) -> tuple[str, list]:
    clauses, params = [], []
    for key, val in filters.items():
        if key not in _FILTERABLE:
            raise KeyError(f"not a filterable column: {key!r} (allowed: {sorted(_FILTERABLE)})")
        if isinstance(val, (list, tuple, set)):
            vals = list(val)
            if not vals:                       # empty IN () matches nothing
                clauses.append("0")
                continue
            clauses.append(f"{key} IN ({','.join('?' * len(vals))})")
            params.extend(vals)
        else:
            clauses.append(f"{key} = ?")
            params.append(val)
    return ("WHERE " + " AND ".join(clauses)) if clauses else "", params


# ** LOCKED **  ** MAIN ** — webtest builds each book's reading-order list via relational.find
# Relational scene lookup: AND of equality / IN filters over indexed columns, ordered by reading position.
def find(conn: sqlite3.Connection, limit: int | None = None,
         order: bool = True, **filters: Any) -> list[dict]:
    where, params = _where(filters)
    sql = f"SELECT * FROM scenes {where}"
    if order:
        sql += " ORDER BY book_id, pos"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return [_row(r) for r in conn.execute(sql, params).fetchall()]


# ** LOCKED **
# How many scenes match the filters (no rows fetched) — the relational COUNT.
def count(conn: sqlite3.Connection, **filters: Any) -> int:
    where, params = _where(filters)
    return conn.execute(f"SELECT COUNT(*) FROM scenes {where}", params).fetchone()[0]


# ** LOCKED **
# GROUP BY count over one whitelisted column, e.g. tally(conn, 'dominant_tone') -> {tone: n}.
def tally(conn: sqlite3.Connection, column: str) -> dict[str, int]:
    if column not in _FILTERABLE:
        raise KeyError(f"not a groupable column: {column!r}")
    cur = conn.execute(
        f"SELECT {column} AS k, COUNT(*) AS n FROM scenes GROUP BY {column} ORDER BY n DESC"
    )
    return {r["k"]: r["n"] for r in cur.fetchall()}
