from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Iterable, Iterator
from utils import SrcPaths, read_json

# ---- subject-tree store: the SQL home of the Gutenberg subject trie (book-level grain) ----
# Sibling to relational.py, but a different grain + source: the BOOK subject taxonomy from catalog
# metadata, flattened to indexed rows so retrieval is a slice, not a whole-file load. ONE row per
# (book, right-anchored subject prefix): a reversed path "...Italy -- Fiction" stores each ancestor
# suffix ("Italy -- Fiction") + its parent ("Fiction"); both indexed, so the SQL B-trees ARE the tree.

DELIM = " -- "   # Gutenberg subject component delimiter (broadest term rightmost)

# ---- schema ----

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


# ** LOCKED **  ** MAIN ** — tests + webtest ensure the table on a shared scenes.db conn
# Create the table + indexes if absent (idempotent).
def ensure_table(conn: sqlite3.Connection) -> None:
    conn.executescript(_DDL)


# ** LOCKED **  ** MAIN ** — tests.subject_sql_test opens the tree standalone
# Open the shared relational file (WAL, dict rows) and ensure only THIS store's table.
def open_db(path: str | Path = SrcPaths.DB_PATH) -> sqlite3.Connection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    ensure_table(conn)   # sibling subject table in the same scenes.db
    return conn


# ---- derivation (pure · LOCKED — the ONE place the " -- " delimiter and reversal live) ----

# ** LOCKED **
# Split a subject on DELIM and REVERSE it into a root-down path ("A -- B -- C" -> ["C","B","A"]).
def _parts(subject: str) -> list[str]:
    return list(reversed([p.strip() for p in subject.split(DELIM) if p.strip()]))


# ** LOCKED **
# A reversed nav list -> the canonical un-reversed suffix key (["Fiction","Italy"] -> "Italy -- Fiction").
def _suffix(path) -> str:
    return DELIM.join(reversed(list(path)))


# ** LOCKED **  ** MAIN ** — embed.index_records + tests.backfill stamp these as the Qdrant `subject_paths` label
# Every distinct right-anchored prefix across a book's subjects, first-seen order (branch filter labels).
def suffixes(subject_strings: Iterable[str]) -> list[str]:
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


# ** LOCKED **
# Yield (book_id, suffix, parent_suffix, depth) for every right-anchored prefix of a book's subjects.
def subject_rows(book_id: str, subjects: Iterable[str]) -> Iterator[tuple]:
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


# ---- write (delete-by-book then insert: idempotent AND shrink-safe) ----

_INSERT = "INSERT OR REPLACE INTO book_subject_path VALUES (?, ?, ?, ?)"


# ** LOCKED **
# Replace ONE book's subject rows (delete-by-book, then insert). Returns rows written.
def upsert(conn: sqlite3.Connection, book_id: str, subjects: Iterable[str]) -> int:
    rows = list(subject_rows(book_id, subjects))   # expand to per-prefix rows
    with conn:
        conn.execute("DELETE FROM book_subject_path WHERE book_id = ?", (book_id,))
        if rows:
            conn.executemany(_INSERT, rows)
    return len(rows)


# ** LOCKED **  ** MAIN ** — webtest._build_subject_tree + build_from_recall rebuild the whole tree here
# Mirror every book's subjects from a code -> metadata-dict map, one transaction. Returns rows written.
def upsert_many(conn: sqlite3.Connection, metadata: dict) -> int:
    total = 0
    with conn:
        for code, md in metadata.items():
            rows = list(subject_rows(code, md.get("Subjects") or ()))   # per-book prefixes
            conn.execute("DELETE FROM book_subject_path WHERE book_id = ?", (code,))
            if rows:
                conn.executemany(_INSERT, rows)
                total += len(rows)
    return total


# ** MAIN ** — tests + embed_test build the tree straight from the metadata cache
# Populate the table from recall/metadata.json alone (no zips, no scene work). Returns rows written.
def build_from_recall(conn: sqlite3.Connection,
                      recall_path: str | Path = SrcPaths.RECALL_DIR) -> int:
    md_cache = read_json(Path(recall_path) / "metadata.json", {})   # code -> metadata dict (Subjects as list)
    return upsert_many(conn, md_cache)


# ** LOCKED **
# Drop one book from the tree (e.g. it left the corpus). Returns rows removed.
def delete_book(conn: sqlite3.Connection, book_id: str) -> int:
    with conn:
        cur = conn.execute("DELETE FROM book_subject_path WHERE book_id = ?", (book_id,))
    return cur.rowcount


# ---- read (recall) ----

# ** LOCKED **  ** MAIN ** — webtest lists a branch's books; O(log N) seek + O(k) stream
# Every book_id whose subject ends in this branch (reversed nav list -> suffix key).
def books_in_branch(conn: sqlite3.Connection, path) -> list[str]:
    cur = conn.execute(
        "SELECT DISTINCT book_id FROM book_subject_path WHERE suffix = ? ORDER BY book_id",
        (_suffix(path),),
    )
    return [r[0] for r in cur.fetchall()]


# ** LOCKED **  ** MAIN ** — webtest reads a branch's book count to pick the exact-vs-walk strategy
# How many distinct books sit in a branch (no rows fetched).
def count_branch(conn: sqlite3.Connection, path) -> int:
    return conn.execute(
        "SELECT COUNT(DISTINCT book_id) FROM book_subject_path WHERE suffix = ?",
        (_suffix(path),),
    ).fetchone()[0]


# ** LOCKED **  ** MAIN ** — tests + webtest browse the tree one level at a time
# The next components branching off a path — the browse / facet view ([] = the roots).
def children(conn: sqlite3.Connection, path=()) -> list[str]:
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


# ** MAIN ** — tests.subject_sql_test reads one branch slice (the thin-client contract)
# One API slice: this branch's child terms + its book ids ({subject, children, books}).
def branch(conn: sqlite3.Connection, path=()) -> dict:
    return {
        "subject": _suffix(path) if path else None,
        "children": children(conn, path),                       # sub-folders
        "books": books_in_branch(conn, path) if path else [],   # files (empty at the root)
    }


# ** LOCKED **
# Stream every branch (suffix, parent_suffix, book_count) in tree order — audit/export.
def walk(conn: sqlite3.Connection) -> Iterator[tuple[str, str, int]]:
    cur = conn.execute(
        "SELECT suffix, parent_suffix, COUNT(DISTINCT book_id) AS n "
        "FROM book_subject_path GROUP BY suffix, parent_suffix ORDER BY suffix")
    for r in cur.fetchall():
        yield r["suffix"], r["parent_suffix"], r["n"]
