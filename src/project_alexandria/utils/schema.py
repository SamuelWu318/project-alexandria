# FOR CLAUDE — Scene-record schema: the SINGLE SOURCE OF TRUTH.
# -----------------------------------------------------------------------------
# Loads scene_schema.json (the editable master registry) and DERIVES every store's
# field contract from it, so the schema is defined ONCE:
#   * process.py       -> blank_record()        (the null template a new scene starts as)
#   * relational.py    -> create_table_sql() / SQL_COLS / to_row() / FILTERABLE / INT_COLS
#   * search.py        -> VECTOR_NAMES / DEFAULT_WEIGHTS
#   * embed.py         -> LLM_FIELDS / QUERY_FIELDS (import-time drift asserts)
# and the reconcile tool ripples an edit of the master across scenes JSON + SQLite +
# Qdrant payloads.
#
# Field CLASSES:
#   core       — structural (ids, positions, prose, provenance). Protected: reconcile
#                never strips or nulls a core field.
#   enrichment — the experimental surface (tones, frame, descriptors, summary). Reconcile
#                nulls a missing one and strips any record key not in the master.
#
# This module stays stdlib-light on purpose (json + pathlib): the JSON/DB reconcile path
# must run without pulling in the LLM client. tags-enum + relational + qdrant imports are
# all LAZY, inside the functions that need them.
#
# Workflow:  edit scene_schema.json  ->  python -m utils.schema --check     (parity vs stores)
#                                    ->  python -m utils.schema --reconcile  (dry-run diff)
#                                    ->  python -m utils.schema --reconcile --apply --db --qdrant
# -----------------------------------------------------------------------------
from __future__ import annotations
import json, os, tempfile
from collections import Counter
from pathlib import Path
from utils import SrcPaths

# a hard floor: the master MUST declare these, so a bad edit can never make reconcile
# strip the columns everything joins on.
_REQUIRED = ("scene_id", "book_id")


# --- load + derive (at import) --- #

def _load() -> dict:
    reg = json.loads(SrcPaths.SCHEMA_PATH.read_text(encoding="utf-8"))
    fields = reg.get("fields") or {}
    for r in _REQUIRED:
        if r not in fields:
            raise ValueError(f"scene_schema.json missing required field {r!r}")
    return reg


_REG = _load()
FIELDS: dict[str, dict] = _REG["fields"]                     # ordered: order == SQL col order
SCHEMA_VERSION: int = _REG["schema_version"]
PRIMARY_KEY: str = _REG["primary_key"]
INDEXES: list[dict] = _REG["indexes"]

FIELD_NAMES = tuple(FIELDS)                                  # every declared field
JSON_FIELDS = tuple(f for f in FIELDS if FIELDS[f]["json"])  # keys that live in a record
CORE = frozenset(f for f in FIELDS if FIELDS[f]["kind"] == "core")
ENRICHMENT = frozenset(f for f in FIELDS if FIELDS[f]["kind"] == "enrichment")
LLM_FIELDS = frozenset(f for f in FIELDS if FIELDS[f].get("source") == "llm")
QUERY_FIELDS = frozenset(f for f in FIELDS if FIELDS[f].get("vector"))  # == distiller frame

# vector store (search.py)
VECTOR_NAMES = tuple(f for f in FIELDS if FIELDS[f].get("vector"))
DEFAULT_WEIGHTS = {f: FIELDS[f]["weight"] for f in VECTOR_NAMES}

# relational store (relational.py)
SQL_COLS = tuple(f for f in FIELDS if FIELDS[f]["sql"].get("col"))
SQL_TYPES = {f: FIELDS[f]["sql"]["type"] for f in SQL_COLS}
INT_COLS = frozenset(f for f in SQL_COLS if SQL_TYPES[f] == "INTEGER")
JSON_STORE_COLS = tuple(f for f in SQL_COLS if FIELDS[f]["sql"].get("store") == "json")
FILTERABLE = frozenset(f for f in FIELDS if FIELDS[f].get("filterable"))


# --- record helpers --- #

def _default(name: str):
    """Declared default for a field ('$schema_version' resolves to the registry version)."""
    d = FIELDS[name]["default"]
    return SCHEMA_VERSION if d == "$schema_version" else d


def blank_record() -> dict:
    """A fresh scene record: every JSON field at its default (the process.py null template)."""
    return {name: _default(name) for name in JSON_FIELDS}


def pos_of(scene_id: str) -> int | None:
    """Dense per-book rank = the integer suffix of the scene_id (`book-i`)."""
    try:
        return int(str(scene_id).rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return None


def to_row(rec: dict) -> tuple:
    """Flatten a record into a SQL column tuple in SQL_COLS order, applying store transforms."""
    out = []
    for name in SQL_COLS:
        store = FIELDS[name]["sql"].get("store", "direct")
        if store == "pos":
            out.append(pos_of(rec.get("scene_id", "")))
        elif store == "bool_int":
            out.append(1 if rec.get(name) else 0)
        elif store == "json":
            out.append(json.dumps(rec.get(name) or [], ensure_ascii=False))
        else:
            out.append(rec.get(name))
    return tuple(out)


def inflate_row(d: dict) -> dict:
    """Re-inflate a SQL row dict: JSON-store columns back into lists (in place, returned)."""
    for col in JSON_STORE_COLS:
        if d.get(col):
            d[col] = json.loads(d[col])
    return d


def create_table_sql() -> str:
    """Derive the full CREATE TABLE + index DDL from the registry."""
    lines = []
    for name in SQL_COLS:
        col = f"    {name:<22} {SQL_TYPES[name]}"
        if name == PRIMARY_KEY:
            col += " PRIMARY KEY"
        lines.append(col)
    ddl = "CREATE TABLE IF NOT EXISTS scenes (\n" + ",\n".join(lines) + "\n);\n"
    for ix in INDEXES:
        uniq = "UNIQUE " if ix.get("unique") else ""
        ddl += f"CREATE {uniq}INDEX IF NOT EXISTS {ix['name']} ON scenes({', '.join(ix['cols'])});\n"
    return ddl


# --- reconcile: make a record match the master --- #

def reconcile(record: dict) -> tuple[dict, dict]:
    """Make one record's keys exactly the master's JSON field set (in place).

    Adds a missing field at its default; strips any key the master no longer declares.
    NEVER overwrites an existing value, and flags (does not silently null) a missing CORE
    field — those are structural and a missing one means an upstream bug, not a tag edit.
    Returns (record, {"added", "removed", "warnings"}).
    """
    added, removed, warnings = [], [], []
    for name in JSON_FIELDS:
        if name not in record:
            record[name] = _default(name)
            added.append(name)
            if FIELDS[name]["kind"] == "core":
                warnings.append(f"core field {name!r} was missing (filled default)")
    for key in list(record):
        if key not in FIELDS or not FIELDS[key]["json"]:
            del record[key]
            removed.append(key)
    return record, {"added": added, "removed": removed, "warnings": warnings}


def _atomic_write_json(path: Path, obj) -> None:
    """Write JSON to `path` atomically (tmp in same dir, then os.replace)."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def reconcile_file(path: str | Path, *, apply: bool = False, backup: bool = True) -> tuple[Counter, set, bool]:
    """Reconcile every record in one scenes json. Dry-run unless apply=True.

    Returns (per-field add/remove counts, set of warnings, changed?). On apply, writes a
    `<path>.bak` sibling first (unless backup=False) then rewrites atomically.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    recs = json.loads(raw)
    stats: Counter = Counter()
    warns: set = set()
    changed = False
    for r in recs:
        _, ch = reconcile(r)
        if ch["added"] or ch["removed"]:
            changed = True
        for a in ch["added"]:
            stats[f"+{a}"] += 1
        for rm in ch["removed"]:
            stats[f"-{rm}"] += 1
        warns.update(ch["warnings"])
    if changed and apply:
        if backup:
            (path.parent / (path.name + ".bak")).write_text(raw, encoding="utf-8")
        _atomic_write_json(path, recs)
    return stats, warns, changed


# --- ripple: SQLite + Qdrant (lazy heavy imports) --- #

def _scene_files(scenes_dir: Path | None = None) -> list[Path]:
    from utils import SrcPaths
    d = Path(scenes_dir) if scenes_dir else Path(SrcPaths.SCENES_DIR)
    return sorted(d.glob("pg*-s.json"))


def sync_db() -> int:
    """Rebuild the SQLite scenes table from the (reconciled) scenes json — DROP + recreate
    from the current DDL so stripped columns actually disappear, then re-upsert every record.
    """
    from utils import relational
    conn = relational.open_db()
    try:
        conn.executescript("DROP TABLE IF EXISTS scenes;")
        conn.executescript(create_table_sql())
        total = 0
        for f in _scene_files():
            recs = json.loads(f.read_text(encoding="utf-8"))
            total += relational.sql_upsert(conn, recs)
        return total
    finally:
        conn.close()


def sync_qdrant() -> int:
    """Overwrite every existing point's payload from the reconciled scenes json, so stripped
    tags vanish from Qdrant too. Payload-only: vectors are untouched. Adding/removing a
    VECTOR field still needs a re-index (embed.index_records) — this does not re-embed.
    """
    import search
    client = search.open_client()
    try:
        n = 0
        for f in _scene_files():
            recs = json.loads(f.read_text(encoding="utf-8"))
            for r in recs:
                if not r.get("summary"):        # only enriched scenes are indexed as points
                    continue
                try:
                    client.overwrite_payload(
                        search.COLLECTION, payload=r,
                        points=[search.point_id(r["scene_id"])])
                    n += 1
                except Exception:
                    pass                         # point not indexed yet — skip
        return n
    finally:
        client.close()


# --- CLI --- #

def _check() -> int:
    """Assert the derived contract matches the store constants + pydantic models. Exit code."""
    import search, embed
    from utils import relational
    problems = []

    def eq(label, a, b):
        if a != b:
            problems.append(f"{label}: derived {a!r} != store {b!r}")

    eq("SQL_COLS", list(SQL_COLS), list(relational._COLS))
    eq("INT_COLS", set(INT_COLS), set(relational._INT_COLS))
    eq("FILTERABLE", set(FILTERABLE), set(relational._FILTERABLE))
    eq("VECTOR_NAMES", set(VECTOR_NAMES), set(search.VECTOR_NAMES))
    eq("DEFAULT_WEIGHTS", DEFAULT_WEIGHTS, dict(search.DEFAULT_FIELD_WEIGHTS))
    eq("SCHEMA_VERSION", SCHEMA_VERSION, __import__("utils").SCHEMA_VERSION)
    llm = {n for n in embed.SceneEnrichment.model_fields if n != "index"}
    eq("LLM_FIELDS", set(LLM_FIELDS), llm)
    eq("QUERY_FIELDS", set(QUERY_FIELDS), set(embed.QueryFrame.model_fields))

    if problems:
        print("SCHEMA PARITY FAIL:")
        for p in problems:
            print("  -", p)
        return 1
    print(f"schema parity OK — {len(SQL_COLS)} SQL cols, {len(VECTOR_NAMES)} vectors, "
          f"{len(CORE)} core / {len(ENRICHMENT)} enrichment fields")
    return 0


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Scene schema: single source of truth + reconcile tool.")
    ap.add_argument("--check", action="store_true", help="assert derived contract == store constants")
    ap.add_argument("--reconcile", action="store_true", help="reconcile scenes json to the master")
    ap.add_argument("--apply", action="store_true", help="write changes (default is dry-run)")
    ap.add_argument("--db", action="store_true", help="also rebuild the SQLite mirror")
    ap.add_argument("--qdrant", action="store_true", help="also overwrite Qdrant payloads")
    ap.add_argument("--no-backup", action="store_true", help="skip the .bak sidecar on apply")
    args = ap.parse_args()

    if args.check:
        raise SystemExit(_check())

    if args.reconcile:
        files = _scene_files()
        total: Counter = Counter()
        warns: set = set()
        any_change = False
        print(f"{'APPLY' if args.apply else 'DRY-RUN'} reconcile over {len(files)} scene files:")
        for f in files:
            stats, w, changed = reconcile_file(f, apply=args.apply, backup=not args.no_backup)
            warns.update(w)
            if changed:
                any_change = True
                delta = " ".join(f"{k}={v}" for k, v in sorted(stats.items()))
                print(f"  {f.name}: {delta}")
        if not any_change:
            print("  (all records already match the master)")
        for w in sorted(warns):
            print(f"  WARN: {w}")
        if args.apply and args.db:
            print(f"rebuilt SQLite mirror: {sync_db()} rows")
        if args.apply and args.qdrant:
            print(f"synced Qdrant payloads: {sync_qdrant()} points")
        if not args.apply and (args.db or args.qdrant):
            print("  (--db/--qdrant skipped: dry-run; add --apply)")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
