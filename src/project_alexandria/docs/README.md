# Project Alexandria — Developer Docs

A **flavor search engine for fiction**: turn Project Gutenberg books into short,
tonally-pure scenes and retrieve them by emotional *flavor* (vibe / tone / decomposed
frame), not plot — so a writer can find and study how a kind of scene is built.

## Start here

- **[PIPELINE.md](PIPELINE.md)** — the main walkthrough: how one book becomes searchable
  scenes, stage by stage, file by file, method by method, and how the record's shape changes
  along the way. Ends with `tests.py`, the harness that drives everything.
- **[SCHEMA.md](SCHEMA.md)** — the scene-record single source of truth
  (`utils/schema/scene_schema.json`) and the `reconcile` tool for rippling schema edits
  across every store.

## The four stages (see PIPELINE.md for detail)

| Stage | File | In → Out |
|---|---|---|
| 1 · Acquire & Parse | `data.py` | `pg{code}-h.zip` → `Book → Chunk → Paragraph` tree (+ recall cache) |
| 2 · Segment | `process.py` | chunk of paragraphs → `pg{code}-s.json` scene records (enrichment null) |
| 3 · Enrich & Index | `embed.py` | null records → enriched records + Qdrant vectors + SQLite rows |
| 4 · Search & Read | `search.py`, `utils/relational.py` | query → ranked scenes + relational queries |

## Running things

All commands run from `src/project_alexandria/` with the venv active
(`../../.venv/bin/python`). The active data root is `SrcPaths.MASTER_DIR`
(`logs/test/`); books, scenes, checkpoints, and the two databases live under it.

```bash
# Full build over the 15 canonical books (download → segment → enrich → index):
../../.venv/bin/python tests.py

# Verify the schema contract after editing scene_schema.json:
../../.venv/bin/python -m utils.schema --check

# A/B the read path against the gold set:
../../.venv/bin/python -m evals --a none --b zscore

# Local browser UI for the read path (stdlib server, port 8765):
../../.venv/bin/python -m webtest.server
```

> The on-disk Qdrant is **single-process**. Stop the webtest server before running anything
> else that opens the index (`evals`, `--reconcile --qdrant`), or it will fail to acquire the
> lock.

## Ownership

The **prompts and model-call tuning** (`BATCH_SYSTEM_PROMPT`, `QUERY_SYSTEM_PROMPT`,
`SYSTEM_PROMPT`, retry/temperature policy, `MODEL`, effort) are the maintainer's surface —
don't edit unless asked. Plumbing, assembly, schema wiring, and docs are fair game.
