# FOR CLAUDE — Relational read/write path: the SQLite mirror of the scene records.
# -----------------------------------------------------------------------------
# The vector store (search.py / Qdrant) answers "what FEELS like X" (semantic
# nearest-neighbour). It is the wrong tool for exact relational lookups — "give me
# every grief scene by Homer whose previous scene was calm", "the two scenes on
# either side of this one", "count scenes per tone". Those are equality / range /
# navigation queries, and they want an indexed relational engine.
#
# This module IS that engine: a plain SQLite file (`master/scenes.db`) holding one
# row per scene, mirrored from the SAME record list that embed.py indexes into
# Qdrant. The two stores join on `scene_id`:
#     Qdrant   = vector source of truth (summary + descriptors + svos)  -> ORDER BY similarity
#     SQLite   = relational mirror (this file)                    -> WHERE / JOIN / COUNT
#
# By design the DB path + schema live HERE, not in storage.py — exactly as the
# Qdrant contract lives in search.py. storage.py centralizes plain-JSON file IO;
# each store owns its own on-disk contract so read and write sides share one home.
#
# Complexity (the point of this file):
#   * get(scene_id)          -> O(1)          (PRIMARY KEY hash)
#   * neighbors(scene_id)    -> O(1)          (UNIQUE(book_id, pos) navigation)
#   * find(**filters)        -> O(log N) locate + O(k) stream   (B-tree per column)
#   * count / group          -> real SQL, which Qdrant cannot do
#
# Invariants:
#   * sql_upsert is idempotent (INSERT OR REPLACE on the scene_id PK) — re-running
#     the pipeline overwrites rows instead of duplicating them, mirroring the
#     stable-uuid overwrite that point_id() gives the vector store.
#   * `pos` is the dense per-book rank taken from the scene_id suffix, which
#     scenes_to_records assigns as `f"{book}-{i}"`. That is the ONE coupling to the
#     id format, isolated in _pos(); neighbours rely on it being contiguous.
#   * descriptors are stored as a JSON text column for display only — they are the
#     vector's job, so they are deliberately NOT an indexed relational field. (A
#     future `scene_descriptors(scene_id, descriptor)` child table would add exact
#     descriptor filtering + JOIN if ever wanted.)
# -----------------------------------------------------------------------------
from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Any, Iterable
from utils import SrcPaths, schema   # schema = the scene-record registry (utils/schema.py)

# --- schema (DERIVED from scene_schema.json via schema.py — the single source of truth) --- #
# This module owns HOW the relational mirror behaves (upsert, filters, navigation); the
# FIELD LIST, column types, and indexes all come from the registry so every store stays in
# lockstep. Edit scene_schema.json (NOT these) to change the schema, then run
# `python -m schema --check` to confirm parity and `--reconcile` to ripple it.
#   * heavy blobs (text_html, book_metadata) are declared sql.col=false in the registry, so
#     they stay out of SQL — this is a relational index, not a blob store.
#   * descriptors is a sql.store="json" column: a JSON list for display, NOT indexed.
_COLS = schema.SQL_COLS             # upsert column order == registry order
_FILTERABLE = schema.FILTERABLE     # find()/count() WHERE + GROUP BY whitelist (injection guard)
_SCHEMA = schema.create_table_sql()  # CREATE TABLE + indexes, generated from the registry
_INT_COLS = schema.INT_COLS         # INTEGER-affinity cols; _migrate uses them to back-fill


# --- connection --- #

def _migrate(conn: sqlite3.Connection) -> None:
    """Add any _COLS column the existing `scenes` table predates (idempotent ALTER).

    CREATE TABLE IF NOT EXISTS builds a fresh file but leaves an OLDER table untouched —
    so a DB created before a later column (e.g. moments / svos) would be missing it and every
    sql_upsert would fail with "no column named moments". This back-fills each missing column
    with the right type. Column names come from _COLS (our own whitelist), never caller input,
    so interpolating them is safe. These enrichment columns are display/JSON only and
    unindexed, so no index needs rebuilding here.
    """
    have = {r["name"] for r in conn.execute("PRAGMA table_info(scenes)")}
    for col in _COLS:
        if col not in have:
            typ = "INTEGER" if col in _INT_COLS else "TEXT"
            conn.execute(f"ALTER TABLE scenes ADD COLUMN {col} {typ}")


def open_db(path: str | Path = SrcPaths.DB_PATH) -> sqlite3.Connection:
    """Open (creating + migrating if needed) the on-disk scene mirror.

    Self-heals: the schema is CREATE ... IF NOT EXISTS, so a fresh file gets built; then
    _migrate ALTERs in any column a pre-existing table predates, so an old DB is brought
    up to the current _COLS without a manual rebuild. Rows come back as dicts (sqlite3.Row
    factory). WAL lets a reader query while embed.py is upserting.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA)
    _migrate(conn)
    return conn


# --- write --- #

def _to_row(rec: dict) -> tuple:
    """Flatten one scene record into a column tuple in _COLS order (registry-driven,
    incl. the pos/bool/json store transforms). See schema.to_row."""
    return schema.to_row(rec)


def sql_upsert(conn: sqlite3.Connection, records: Iterable[dict]) -> int:
    """Mirror `records` into the scenes table (INSERT OR REPLACE, one transaction).

    Idempotent on scene_id — re-running the pipeline overwrites, never duplicates.
    Unlike the vector index (enriched scenes only), this stores EVERY record passed,
    so relational queries work independently of enrichment. Returns rows written.
    """
    rows = [_to_row(r) for r in records if r.get("scene_id")]
    if not rows:
        return 0
    placeholders = ",".join("?" * len(_COLS))
    sql = f"INSERT OR REPLACE INTO scenes ({','.join(_COLS)}) VALUES ({placeholders})"
    with conn:                       # commit/rollback as one unit
        conn.executemany(sql, rows)
    return len(rows)


# --- read --- #

def _row(r: sqlite3.Row | None) -> dict | None:
    """sqlite3.Row -> plain dict, re-inflating the JSON descriptors column."""
    if r is None:
        return None
    return schema.inflate_row(dict(r))   # re-inflate JSON-store columns (descriptors)


def get(conn: sqlite3.Connection, scene_id: str) -> dict | None:
    """One scene by id — O(1) PRIMARY KEY lookup. None if absent."""
    cur = conn.execute("SELECT * FROM scenes WHERE scene_id = ?", (scene_id,))
    return _row(cur.fetchone())


def neighbors(conn: sqlite3.Connection, scene_id: str,
              before: int = 1, after: int = 1) -> list[dict]:
    """The window of scenes around `scene_id` in reading order (anchor included).

    O(1): resolve the anchor's (book_id, pos), then a covering range scan on the
    UNIQUE(book_id, pos) index. Returns rows ordered by pos; the anchor sits in the
    middle. Empty list if the anchor is unknown or has no parseable pos.
    """
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


def _where(filters: dict) -> tuple[str, list]:
    """Build a parametrized WHERE from whitelisted kwargs.

    Scalar value -> `col = ?`; list/tuple/set -> `col IN (?, ?, ...)`. Unknown keys
    raise (never interpolate a caller string as a column name). Returns ("", []) for
    no filters.
    """
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


def find(conn: sqlite3.Connection, limit: int | None = None,
         order: bool = True, **filters: Any) -> list[dict]:
    """Relational scene lookup: AND of equality / IN filters over indexed columns.

    Each kwarg is a filterable column (see _FILTERABLE); pass a scalar for `=` or a
    list for `IN`. e.g. find(conn, dominant_tone="grief", author="Homer",
    prev_tone="calm") or find(conn, dominant_tone=["grief", "sorrow"]).
    B-tree resolves the match set in O(log N); results stream in O(k). `order` sorts
    by reading position (book_id, pos); `limit` caps the rows returned.
    """
    where, params = _where(filters)
    sql = f"SELECT * FROM scenes {where}"
    if order:
        sql += " ORDER BY book_id, pos"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return [_row(r) for r in conn.execute(sql, params).fetchall()]


def count(conn: sqlite3.Connection, **filters: Any) -> int:
    """How many scenes match the filters (no rows fetched) — the relational COUNT."""
    where, params = _where(filters)
    return conn.execute(f"SELECT COUNT(*) FROM scenes {where}", params).fetchone()[0]


def tally(conn: sqlite3.Connection, column: str) -> dict[str, int]:
    """GROUP BY count over one column, e.g. tally(conn, 'dominant_tone') -> {tone: n}.

    A cheap relational aggregate the vector store cannot do at all. `column` is
    whitelisted for the same injection reason as the filters."""
    if column not in _FILTERABLE:
        raise KeyError(f"not a groupable column: {column!r}")
    cur = conn.execute(
        f"SELECT {column} AS k, COUNT(*) AS n FROM scenes GROUP BY {column} ORDER BY n DESC"
    )
    return {r["k"]: r["n"] for r in cur.fetchall()}
