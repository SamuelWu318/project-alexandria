# The Scene Schema — Single Source of Truth

The scene-record shape is defined **once** and every store derives its contract from it, so
the schema can't drift across the JSON files, the SQLite mirror, and the Qdrant index.

- **Registry (edit this):** [`utils/schema/scene_schema.json`](../utils/schema/scene_schema.json)
- **Loader / deriver / reconcile tool:** [`utils/schema.py`](../utils/schema.py)
- **Path:** `SrcPaths.SCHEMA_PATH` (in `utils/storage.py`) — a code asset shipped with the
  package, not master data.
- Stores import it as `from utils import schema`.

## What derives from the registry

| Consumer | Derives |
|---|---|
| `process.py` | `schema.blank_record()` — the null template a new scene starts as |
| `utils/relational.py` | `SQL_COLS`, `create_table_sql()`, `to_row()`, `inflate_row()`, `FILTERABLE`, `INT_COLS` |
| `search.py` | `VECTOR_NAMES`, `DEFAULT_WEIGHTS` |
| `embed.py` | `LLM_FIELDS`, `QUERY_FIELDS` (import-time drift asserts on the pydantic models) |

Because everything derives, you never hand-edit those lists. Change the registry, run
`--check`, then `--reconcile`.

## Field entry anatomy

Each field in `scene_schema.json` declares:

- `kind` — `core` (structural; protected from reconcile) or `enrichment` (experimental).
- `source` — `segment` | `llm` | `derived` | `pipeline` (who fills it; `llm` fields must match
  `SceneEnrichment`).
- `json` — is it a key in the scene record (vs SQL-only, like `pos`).
- `sql` — `{col, type, store}` where `store` ∈ `direct` | `pos` | `bool_int` | `json`
  (the flatten transform); `col: false` keeps a heavy blob (`text_html`, `book_metadata`)
  out of SQLite.
- `filterable` — included in the relational WHERE / GROUP BY whitelist (injection guard).
- `vector` + `weight` — is it a Qdrant named vector, and its default fusion weight.
- `enum` — controlled vocabulary (`Tone` / `Intensity` / `Arc`, from `utils/tags.py`).
- `default` — value in `blank_record()` (`"$schema_version"` resolves to the registry version).

**Field order in the registry = SQLite column order.** Top-level `indexes` defines the DDL
indexes; `primary_key` is `scene_id`.

## Editing the schema

From `src/project_alexandria/` with the venv active:

```bash
# 1. Edit utils/schema/scene_schema.json (add / rename / drop a field).

# 2. Confirm the derived contract still matches the stores + pydantic models:
../../.venv/bin/python -m utils.schema --check

# 3. Dry-run the reconcile — see the per-file add/remove diff, no writes:
../../.venv/bin/python -m utils.schema --reconcile

# 4. Apply: rewrite scenes JSON (with .bak), rebuild SQLite, overwrite Qdrant payloads:
../../.venv/bin/python -m utils.schema --reconcile --apply --db --qdrant
```

**What `reconcile` does to each record:** adds any missing field at its default, strips any
key the registry no longer declares, and **never** strips or overwrites a `core` field (a
missing core field is flagged, not nulled). Existing values are preserved.

**Important:** reconcile ripples *structure* (JSON / SQLite / Qdrant payloads) but does **not
re-embed**. Adding, removing, or renaming a `vector: true` field also needs a re-index via
`embed.index_records` (re-run Stage 3). The Qdrant steps need the webtest server stopped
(local Qdrant is single-process on-disk).

## Guard rails

- `python -m utils.schema --check` asserts the derived `SQL_COLS` / `INT_COLS` / `FILTERABLE`
  / `VECTOR_NAMES` / `DEFAULT_WEIGHTS` match the store constants, `SCHEMA_VERSION` matches
  `utils.SCHEMA_VERSION`, and the `SceneEnrichment` / `QueryFrame` field sets match the
  registry. Run it after any schema edit.
- `embed.py` runs the model asserts at **import**, so a desync fails loudly the moment the
  package loads, not silently at write time.

## Related: evaluating a schema/search change

[`evals.py`](../evals.py) grades the read path against the gold query set
(`webtest/gold/test_queries.json`), scoring **book-match** MRR / Hit@k / P@k by query
sharpness, and A/B-compares two configs. Use it to check whether a schema/vector/weight
change actually helps before committing it:

```bash
../../.venv/bin/python -m evals --a none --b zscore   # raw vs z-scored fusion
../../.venv/bin/python -m evals --tune                # coordinate-ascent field weights
```

Weight tuning on the full gold set **overfits** (in-sample MRR ~0.98) — validate held-out
before committing new `weight` values into the registry.
