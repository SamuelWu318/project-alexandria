from __future__ import annotations
import json, os, tempfile
from collections import Counter
from pathlib import Path
from utils import SrcPaths

# ---- scene-record schema: the SINGLE SOURCE OF TRUTH ----
# Loads scene_schema.json (the editable master registry) and DERIVES every store's field contract
# from it, so the schema is defined ONCE. Stays stdlib-light on purpose (json + pathlib) — the
# reconcile path must run without pulling in the LLM client; tags/relational/qdrant imports are LAZY.
# Workflow: edit scene_schema.json -> `python -m utils.schema --check` (parity) / `--reconcile` (ripple).

# a hard floor: the master MUST declare these, so a bad edit can never strip the join columns.
_REQUIRED = ("scene_id", "book_id")


# ---- load + derive (at import) ----

# Load + validate scene_schema.json (raises if a required field is missing).
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
# multivector fields (svos): a LIST of per-item vectors scored by MAX_SIM (max-pooling).
MULTIVECTOR_NAMES = tuple(f for f in VECTOR_NAMES if FIELDS[f].get("multivector"))

# relational store (relational.py)
SQL_COLS = tuple(f for f in FIELDS if FIELDS[f]["sql"].get("col"))
SQL_TYPES = {f: FIELDS[f]["sql"]["type"] for f in SQL_COLS}
INT_COLS = frozenset(f for f in SQL_COLS if SQL_TYPES[f] == "INTEGER")
JSON_STORE_COLS = tuple(f for f in SQL_COLS if FIELDS[f]["sql"].get("store") == "json")
FILTERABLE = frozenset(f for f in FIELDS if FIELDS[f].get("filterable"))


# ---- record helpers ----

# The declared default for a field ('$schema_version' resolves to the registry version).
def _default(name: str):
    d = FIELDS[name]["default"]
    return SCHEMA_VERSION if d == "$schema_version" else d


# ** MAIN ** — process.scenes_to_records starts every new scene from this null template
# A fresh scene record: every JSON field at its declared default.
def blank_record() -> dict:
    return {name: _default(name) for name in JSON_FIELDS}


# ** LOCKED **
# Dense per-book rank = the integer suffix of the scene_id (`book-i`); None if unparseable.
def pos_of(scene_id: str) -> int | None:
    try:
        return int(str(scene_id).rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return None


# ** MAIN ** — relational._to_row flattens every record for SQL through here
# Flatten a record into a SQL column tuple in SQL_COLS order, applying each column's store transform.
def to_row(rec: dict) -> tuple:
    out = []
    for name in SQL_COLS:
        store = FIELDS[name]["sql"].get("store", "direct")
        if store == "pos":
            out.append(pos_of(rec.get("scene_id", "")))   # id suffix -> dense rank
        elif store == "bool_int":
            out.append(1 if rec.get(name) else 0)
        elif store == "json":
            out.append(json.dumps(rec.get(name) or [], ensure_ascii=False))
        else:
            out.append(rec.get(name))
    return tuple(out)


# ** MAIN ** — relational._row rehydrates every SQL row through here
# Re-inflate a SQL row dict: JSON-store columns back into lists (in place, returned).
def inflate_row(d: dict) -> dict:
    for col in JSON_STORE_COLS:
        if d.get(col):
            d[col] = json.loads(d[col])
    return d


# ** MAIN ** — relational.open_db builds the table from this DDL
# Derive the full CREATE TABLE + index DDL from the registry.
def create_table_sql() -> str:
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


# ---- reconcile: make a record's keys match the master (structure only, never a value) ----

# Sync one record's keys to the master JSON field set: add missing at default, strip unknown, flag a missing CORE field.
def reconcile(record: dict) -> tuple[dict, dict]:
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


# ** LOCKED **
# Write JSON to `path` atomically (tmp in same dir, then os.replace).
def _atomic_write_json(path: Path, obj) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# Reconcile every record in one scenes json (dry-run unless apply); writes a .bak then rewrites atomically.
def reconcile_file(path: str | Path, *, apply: bool = False, backup: bool = True) -> tuple[Counter, set, bool]:
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    recs = json.loads(raw)
    stats: Counter = Counter()
    warns: set = set()
    changed = False
    for r in recs:
        _, ch = reconcile(r)                     # sync one record's structure
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
        _atomic_write_json(path, recs)           # crash-safe rewrite
    return stats, warns, changed


# ---- clear: reset enrichment values to prepare a re-enrichment (RESETS values, unlike reconcile) ----

# Reset each named field to default; if anything changed, mark the record un-enriched so it reprocesses. Returns changed?.
def _clear_record(rec: dict, cols: tuple) -> bool:
    changed = False
    for c in cols:
        d = _default(c)
        if rec.get(c) != d:
            rec[c] = d
            changed = True
    if changed and rec.get("enriched"):
        rec["enriched"] = False
        rec["enrich_model"] = _default("enrich_model")
    return changed


# Clear the named enrichment fields in every record of one scenes json (dry-run unless apply). Returns (recs changed, changed?).
def clear_file(path: str | Path, cols: tuple, *, apply: bool = False, backup: bool = True) -> tuple[int, bool]:
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    recs = json.loads(raw)
    n = sum(1 for r in recs if _clear_record(r, cols))   # reset the fields per record
    if n and apply:
        if backup:
            (path.parent / (path.name + ".bak")).write_text(raw, encoding="utf-8")
        _atomic_write_json(path, recs)
    return n, bool(n)


# Wipe the enrichment RESUME caches (per-scene checkpoints + book status flags) so cleared scenes actually re-run.
def reset_enrich_state() -> None:
    import shutil
    from utils import SrcPaths
    ck = Path(SrcPaths.ENRICH_CKPT_DIR)
    if ck.exists():
        shutil.rmtree(ck, ignore_errors=True)   # Checkpoint() recreates it on next save
    st = Path(SrcPaths.STATUS_PATH)
    if st.exists():
        st.unlink()


# ---- ripple: SQLite + Qdrant (lazy heavy imports) ----

# Every pg*-s.json under the scenes dir, sorted.
def _scene_files(scenes_dir: Path | None = None) -> list[Path]:
    from utils import SrcPaths
    d = Path(scenes_dir) if scenes_dir else Path(SrcPaths.SCENES_DIR)
    return sorted(d.glob("pg*-s.json"))


# Rebuild the SQLite scenes table from the reconciled jsons (DROP + recreate from current DDL, re-upsert all). Returns rows.
def sync_db() -> int:
    from utils import relational
    conn = relational.open_db()
    try:
        conn.executescript("DROP TABLE IF EXISTS scenes;")
        conn.executescript(create_table_sql())              # current DDL, stripped cols gone
        total = 0
        for f in _scene_files():
            recs = json.loads(f.read_text(encoding="utf-8"))
            total += relational.sql_upsert(conn, recs)       # mirror each book's records
        return total
    finally:
        conn.close()


# Overwrite every indexed point's payload from the reconciled jsons (payload-only; vectors untouched). Returns points.
def sync_qdrant() -> int:
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


# ---- CLI ----

# Assert the derived contract matches every store's constants + pydantic models. Returns an exit code.
def _check() -> int:
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
    eq("MULTIVECTOR_NAMES", set(MULTIVECTOR_NAMES), set(search.MULTIVECTOR_NAMES))
    eq("DEFAULT_WEIGHTS", DEFAULT_WEIGHTS, dict(search.DEFAULT_FIELD_WEIGHTS))
    eq("SCHEMA_VERSION", SCHEMA_VERSION, __import__("utils").SCHEMA_VERSION)
    llm = {n for n in embed.SceneEnrichment.model_fields if n != "index"}
    eq("LLM_FIELDS", set(LLM_FIELDS), llm)
    # QUERY_FIELDS <-> QueryFrame parity is SUSPENDED during the svos transition (query input is
    # manual now). Restore when QueryFrame is rewritten for moments — see the matching note in embed.py.
    # eq("QUERY_FIELDS", set(QUERY_FIELDS), set(embed.QueryFrame.model_fields))

    if problems:
        print("SCHEMA PARITY FAIL:")
        for p in problems:
            print("  -", p)
        return 1
    print(f"schema parity OK — {len(SQL_COLS)} SQL cols, {len(VECTOR_NAMES)} vectors, "
          f"{len(CORE)} core / {len(ENRICHMENT)} enrichment fields")
    return 0


# CLI entry: dispatch --check / --reconcile / --clear over the scene jsons (+ optional --db / --qdrant ripple).
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Scene schema: single source of truth + reconcile tool.")
    ap.add_argument("--check", action="store_true", help="assert derived contract == store constants")
    ap.add_argument("--reconcile", action="store_true", help="reconcile scenes json to the master")
    ap.add_argument("--clear", metavar="COLS",
                    help="reset these enrichment fields (comma-separated) to default + mark "
                         "scenes un-enriched, to prepare a re-enrichment (e.g. subject,verb,object)")
    ap.add_argument("--apply", action="store_true", help="write changes (default is dry-run)")
    ap.add_argument("--db", action="store_true", help="also rebuild the SQLite mirror")
    ap.add_argument("--qdrant", action="store_true", help="also overwrite Qdrant payloads")
    ap.add_argument("--no-backup", action="store_true", help="skip the .bak sidecar on apply")
    args = ap.parse_args()

    if args.check:
        raise SystemExit(_check())                # parity gate

    if args.clear:
        cols = tuple(c.strip() for c in args.clear.split(",") if c.strip())
        bad = [c for c in cols if c not in ENRICHMENT]
        if bad:
            raise SystemExit(f"--clear: {bad} are not clearable enrichment fields "
                             f"(enrichment fields: {sorted(ENRICHMENT)})")
        files = _scene_files()
        print(f"{'APPLY' if args.apply else 'DRY-RUN'} clear {list(cols)} over {len(files)} scene files:")
        total = 0
        for f in files:
            n, changed = clear_file(f, cols, apply=args.apply, backup=not args.no_backup)   # reset per file
            if changed:
                total += n
                print(f"  {f.name}: cleared on {n} recs (enriched -> False)")
        if not total:
            print("  (nothing to clear — already at default)")
        if args.apply:
            reset_enrich_state()                  # wipe resume caches so scenes re-run
            print("  reset enrichment resume state: checkpoints wiped + status.json removed")
            if args.db:
                print(f"  rebuilt SQLite mirror: {sync_db()} rows")
            print("  Qdrant: re-index rebuilds the collection (multivector config) — no --qdrant here")
        else:
            print("  (dry-run; add --apply to write, optionally --db to rebuild SQLite)")
        return

    if args.reconcile:
        files = _scene_files()
        total: Counter = Counter()
        warns: set = set()
        any_change = False
        print(f"{'APPLY' if args.apply else 'DRY-RUN'} reconcile over {len(files)} scene files:")
        for f in files:
            stats, w, changed = reconcile_file(f, apply=args.apply, backup=not args.no_backup)   # sync per file
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
