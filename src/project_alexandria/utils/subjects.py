# FOR CLAUDE — Subject-tree relational store: the SQL home of the Gutenberg subject trie.
# -----------------------------------------------------------------------------
# Sibling to relational.py. Where relational.py mirrors SCENES (one row per scene,
# scene_schema-driven), this owns the BOOK-level subject taxonomy — a different grain
# and a different source (catalog metadata, not scene records), so it gets its own
# module + on-disk contract, exactly as relational.py argues each store should.
#
# A nested in-memory/JSON trie does not scale: at ~50k books / ~400k subjects it is a
# ~100 MB blob reparsed into ~500k objects every startup, rewritten wholesale per book
# added, with book codes duplicated up every ancestor. This module is the SCALE form —
# the trie flattened into indexed rows so retrieval is a slice, not a whole-file load.
#
# ONE row per (book, right-anchored subject prefix). The reversed path
#     "World War, 1914-1918 -- Campaigns -- Italy -- Fiction"
#     reversed -> [Fiction, Italy, Campaigns, World War, 1914-1918]
# emits four rows (depth 1..4), each carrying the SUFFIX (the un-reversed prefix, e.g.
# "Italy -- Fiction") and its PARENT suffix ("Fiction"). suffix is the retrieval key;
# parent_suffix is the browse key. Both are indexed, so the SQL B-trees ARE the tree —
# no trie object is materialised to answer a query.
#
# Complexity:
#   * books_in_branch(path) -> O(log N) locate + O(k) stream   (idx_bsp_suffix)
#   * children(path)        -> O(log N) locate + O(k) stream   (idx_bsp_parent)
#   * upsert(book)          -> O(rows-of-one-book)             (delete-by-book + insert)
# Contrast the JSON form, whose every write is O(whole corpus).
#
# Invariants:
#   * upsert is idempotent AND shrink-safe: it DELETEs the book's rows before re-inserting,
#     so a re-parse that drops a subject leaves no stale prefix behind (INSERT OR REPLACE
#     alone could not, since the row set can shrink).
#   * suffix is the canonical un-reversed prefix joined by " -- "; parent_suffix is that
#     same string with its leading component removed (NULL at depth 1). _suffix()/_parts()
#     are the ONE place the " -- " delimiter and the reversal live.
#   * derivation is pure (subject_rows) and independent of scenes — the table can be built
#     from metadata alone (build_from_recall), before any scene work.
# -----------------------------------------------------------------------------
from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Iterable, Iterator
from utils import SrcPaths, read_json

DELIM = " -- "   # Gutenberg subject component delimiter (broadest term rightmost)

# --- schema --- #

_DDL = """
CREATE TABLE IF NOT EXISTS book_subject_path (
    book_id       TEXT    NOT NULL,   -- Gutenberg text number
    suffix        TEXT    NOT NULL,   -- right-anchored prefix, e.g. "Italy -- Fiction"
    parent_suffix TEXT,               -- suffix with leading term dropped ("Fiction"); NULL at depth 1
    depth         INTEGER NOT NULL,   -- 1 = root term, grows leftward
    PRIMARY KEY (book_id, suffix)     -- one row per (book, prefix); dedups repeats within a book
);
CREATE INDEX IF NOT EXISTS idx_bsp_suffix ON book_subject_path(suffix);         -- retrieval key
CREATE INDEX IF NOT EXISTS idx_bsp_parent ON book_subject_path(parent_suffix);  -- browse key
"""


def ensure_table(conn: sqlite3.Connection) -> None:
    """Create the table + indexes if absent (idempotent). Safe on a shared scenes.db conn."""
    conn.executescript(_DDL)


def open_db(path: str | Path = SrcPaths.DB_PATH) -> sqlite3.Connection:
    """Open the shared relational file and ensure THIS store's table exists.

    Mirrors relational.open_db (same WAL/row_factory setup) but ensures only the subject
    table, so subjects can be built/queried standalone — the two tables coexist in one
    scenes.db and join on book_id.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    ensure_table(conn)
    return conn


# --- derivation (pure) --- #

def _parts(subject: str) -> list[str]:
    """Split a subject on DELIM and REVERSE it into a root-down path.

    "A -- B -- C" -> ["C", "B", "A"]  (C broadest/root). Empty components dropped.
    """
    return list(reversed([p.strip() for p in subject.split(DELIM) if p.strip()]))


def _suffix(path) -> str:
    """A reversed nav list -> the canonical un-reversed suffix key.

    ["Fiction", "Italy"] -> "Italy -- Fiction". This is the exact string stored in `suffix`,
    so a nav path and a stored row meet on one join key.
    """
    return DELIM.join(reversed(list(path)))


def suffixes(subject_strings: Iterable[str]) -> list[str]:
    """Every distinct right-anchored prefix across a book's subjects — the Qdrant payload labels.

    Same expansion as subject_rows, but flattened to the bare suffix strings (no book_id /
    depth), in first-seen order. This is the list stamped onto each scene point as
    `subject_paths`, so one exact keyword match filters a branch at any depth.
        ["Italy -- Fiction", ...] -> matched by MatchValue("Italy -- Fiction")
    """
    out: list[str] = []
    seen: set[str] = set()
    for subj in subject_strings:
        rev = _parts(subj)
        for d in range(1, len(rev) + 1):
            s = DELIM.join(reversed(rev[:d]))
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


def subject_rows(book_id: str, subjects: Iterable[str]) -> Iterator[tuple]:
    """Yield (book_id, suffix, parent_suffix, depth) for every right-anchored prefix.

    One subject of depth d yields d rows (each ancestor prefix). Prefixes shared across a
    book's subjects are emitted once (the PK would dedup anyway; this saves the round-trip).
    """
    seen: set[str] = set()
    for subj in subjects:
        rev = _parts(subj)
        for d in range(1, len(rev) + 1):
            suffix = DELIM.join(reversed(rev[:d]))
            if suffix in seen:
                continue
            seen.add(suffix)
            parent = DELIM.join(reversed(rev[:d - 1])) if d > 1 else None
            yield (book_id, suffix, parent, d)


# --- write --- #

_INSERT = "INSERT OR REPLACE INTO book_subject_path VALUES (?, ?, ?, ?)"


def upsert(conn: sqlite3.Connection, book_id: str, subjects: Iterable[str]) -> int:
    """Replace ONE book's subject rows (delete-by-book, then insert). Idempotent + shrink-safe.

    Re-running the pipeline for a book whose subjects changed leaves no stale prefixes.
    Returns rows written. O(rows-of-one-book) — never touches other books.
    """
    rows = list(subject_rows(book_id, subjects))
    with conn:
        conn.execute("DELETE FROM book_subject_path WHERE book_id = ?", (book_id,))
        if rows:
            conn.executemany(_INSERT, rows)
    return len(rows)


def upsert_many(conn: sqlite3.Connection, metadata: dict) -> int:
    """Mirror every book's subjects from a code -> metadata-dict map. Returns rows written.

    One transaction. Each book is delete-by-book then insert, so the whole call is an
    idempotent rebuild of exactly the books present in `metadata`.
    """
    total = 0
    with conn:
        for code, md in metadata.items():
            rows = list(subject_rows(code, md.get("Subjects") or ()))
            conn.execute("DELETE FROM book_subject_path WHERE book_id = ?", (code,))
            if rows:
                conn.executemany(_INSERT, rows)
                total += len(rows)
    return total


def build_from_recall(conn: sqlite3.Connection,
                      recall_path: str | Path = SrcPaths.RECALL_DIR) -> int:
    """Populate the table straight from recall/metadata.json — no scene work, no zips.

    Subjects are stored as a plain list in the cache, which subject_rows consumes directly,
    so this needs neither the zips nor the parsed Books — just the one JSON.
    """
    md_cache = read_json(Path(recall_path) / "metadata.json", {})
    return upsert_many(conn, md_cache)


def delete_book(conn: sqlite3.Connection, book_id: str) -> int:
    """Drop one book from the tree (e.g. it left the corpus). Returns rows removed."""
    with conn:
        cur = conn.execute("DELETE FROM book_subject_path WHERE book_id = ?", (book_id,))
    return cur.rowcount


# --- read (recall) --- #

def books_in_branch(conn: sqlite3.Connection, path) -> list[str]:
    """Every book whose subject ends in this branch.

    IN : path = reversed nav list, e.g. ["Fiction", "Italy"]
    OUT: list of book_ids (codes) for "Italy -- Fiction". O(log N) index seek + O(k) stream.
    """
    cur = conn.execute(
        "SELECT DISTINCT book_id FROM book_subject_path WHERE suffix = ? ORDER BY book_id",
        (_suffix(path),),
    )
    return [r[0] for r in cur.fetchall()]


def count_branch(conn: sqlite3.Connection, path) -> int:
    """How many distinct books sit in a branch (no rows fetched)."""
    return conn.execute(
        "SELECT COUNT(DISTINCT book_id) FROM book_subject_path WHERE suffix = ?",
        (_suffix(path),),
    ).fetchone()[0]


def children(conn: sqlite3.Connection, path=()) -> list[str]:
    """The next components branching off a path — the browse / facet view.

    IN : path = reversed nav list; [] (or ()) = the roots.
    OUT: sorted child terms. children([]) -> ["Adventure stories", ..., "Fiction", ...];
         children(["Fiction"]) -> ["France", "Italy", ...]. O(log N) seek + O(k) stream.
    """
    if path:
        cur = conn.execute(
            "SELECT DISTINCT suffix FROM book_subject_path WHERE parent_suffix = ?",
            (_suffix(path),),
        )
    else:
        cur = conn.execute(
            "SELECT DISTINCT suffix FROM book_subject_path WHERE depth = 1")
    # a child's suffix is "child -- <path>", so its leading component IS the child term
    return sorted(r[0].split(DELIM, 1)[0] for r in cur.fetchall())


def branch(conn: sqlite3.Connection, path=()) -> dict:
    """One API slice for a client: this branch's child terms + its book ids.

    IN : path = reversed nav list ([] = roots).
    OUT: {"subject": "Italy -- Fiction" | None, "children": [...], "books": [...]}.
         `books` is empty at the root ([]), where only browsing (children) makes sense.
    This is the thin-client contract — a small slice, never the whole tree.
    """
    return {
        "subject": _suffix(path) if path else None,
        "children": children(conn, path),
        "books": books_in_branch(conn, path) if path else [],
    }


def walk(conn: sqlite3.Connection) -> Iterator[tuple[str, str, int]]:
    """Stream every branch (suffix, parent_suffix, book_count) in tree order — audit/export.

    Ordered by suffix so siblings group; book_count is the aggregate per branch.
    """
    cur = conn.execute(
        "SELECT suffix, parent_suffix, COUNT(DISTINCT book_id) AS n "
        "FROM book_subject_path GROUP BY suffix, parent_suffix ORDER BY suffix")
    for r in cur.fetchall():
        yield r["suffix"], r["parent_suffix"], r["n"]
