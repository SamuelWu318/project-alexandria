# Project Alexandria — Pipeline Walkthrough

A contributor's map of **how one book becomes searchable scenes**, stage by stage:
which file owns each stage, what each method does to the data as it passes through, and
how the record's shape changes along the way. It ends with `tests.py`, the harness that
drives the whole thing end to end.

> Alexandria is a **flavor search engine for fiction**: it fetches scenes by emotional
> *flavor* (tone / vibe), not plot, so a writer can find "a tense stairwell climb toward an
> unseen presence" and study how it's built. Everything below serves that goal.

---

## The data flow at a glance

```mermaid
flowchart LR
    A[pg{code}-h.zip<br/>raw Gutenberg HTML] -->|data.py| B[Book → Chunks → Paragraphs<br/>parse tree + recall cache]
    B -->|process.py| C[pg{code}-s.json<br/>flat scene records<br/>enrichment fields = null]
    C -->|embed.py| D[enriched records<br/>+ Qdrant vectors<br/>+ SQLite rows]
    D -->|search.py / relational.py| E[ranked scenes<br/>+ relational queries]
```

Four stages, four main files, plus one registry that defines the record shape and a few
shared utilities. The unit that flows through everything is the **scene record**.

---

## The thing that flows: the scene record

Every stage reads or writes the same dict — one record per scene, one record per future
Qdrant point. Its shape is defined **once** in
[`utils/schema/scene_schema.json`](../utils/schema/scene_schema.json) (the editable
registry) and loaded by [`utils/schema.py`](../utils/schema.py). See
[SCHEMA.md](SCHEMA.md) for the source-of-truth mechanics.

Fields fall into two classes:

- **core** (18) — structural, set by segmentation, never touched by enrichment:
  `scene_id`, `book_id`, `pos`*, `prev_scene_id`, `next_scene_id`, `scene_title`,
  `chapter_title`, `stitch_status`, `start_paragraph_index`, `end_paragraph_index`,
  `word_count`, `author`, `language`, `book_metadata`, `text_html`, `schema_version`,
  `enriched`, `enrich_model`.  (*`pos` is derived and lives only in SQLite.)
- **enrichment** (11) — the experimental surface the LLM fills:
  `dominant_tone`, `intensity`, `arc`, `descriptors`, `subject`, `verb`, `object`,
  `setting`, `prev_tone`, `next_tone`, `summary`.

**How the record changes across stages:**

| after stage | core fields | enrichment fields |
|---|---|---|
| 2 — segment | filled | all `null` (`enriched: false`) |
| 3 — enrich | filled | filled (`enriched: true`) + denormalized `prev/next_tone` |
| 3 — index | → SQLite row (via `to_row`) + Qdrant payload (full record) + 7 named vectors | |

---

## Stage 1 — Acquire & Parse → `data.py`

**In:** a Gutenberg text number (e.g. `1727`) and its `pg{code}-h.zip`.
**Out:** an in-memory `Book → Chunk → Paragraph` tree, cached to `recall/`.

The zip is downloaded by `tests.step_one_retrieval` (a `wget` per id); `data.py` turns the
HTML into a chunked parse tree the segmenter can consume.

**`MetadataParser`** — one book's catalog metadata.
- `_load_catalog` — reads `pg_catalog.csv` into a `Text# → row` map (once).
- `feed(file_code)` — builds the metadata dict (title, author, translator, subjects set,
  date, language) from that book's catalog row.
- `_parse_name` — flips `"Last, First"` → `"First Last"`, dropping `[notes]`/dates.
- `to_dict` / `from_dict` — JSON round-trip (the `Subjects` set ⇄ sorted list) for the cache.

**`SceneParser`** — HTML → chunked paragraphs.
- `parse_file` — opens the `-h.zip`, sniffs the encoding, returns the raw HTML string.
- `parse_html` — BeautifulSoup parse; strips comments + the boilerplate selectors
  (`#pg-header`, `#pg-footer`, license); returns the `<body>`.
- `get_segments` — cuts the body into ordered `(heading, paragraphs)` segments, preferring
  `div.chapter`, then `<h2>` headings, then the whole document. Loose `<p>` outside every
  chapter are swept into a leading **"Front Matter"** segment so nothing is dropped.
- `_clean_html` — keeps inline tags (`i`, `b`, `em`, …), unwraps the rest, collapses
  whitespace — the paragraph's stored render form.
- `_heading` — first heading text inside a tag, normalized.
- `_pack` — greedily groups a chapter's paragraphs into parts ≤ `TARGET_CHARS` (~6k tokens),
  never splitting a paragraph.
- `parse_book` — the assembler: drop segments under `MIN_SEGMENT_CHARS`, assign **global**
  book-wide `Paragraph.index` values (never reset per chapter — later stages rely on this
  contiguity), then pack into `Chunk`s. Each chunk carries `OVERLAP_PARAGRAPHS` of read-only
  lookback `context` (reaching across chapter boundaries) so a scene split across chunks can
  be caught during segmentation.
- `parse` — convenience: zip → stripped body → chunked `Book`.

**Data model** (`Book` / `Chunk` / `Paragraph` dataclasses):
- `Paragraph{index, text}` — the atom; `to_dict`/`from_dict` for the cache.
- `Chunk` — a section sent to the LLM. `scene_payload()` is the **exact JSON the segmenter
  sees**: `{chapter_title, section_within_chunk, read_only_context_paragraphs,
  number_of_indexed_paragraphs, indexed_paragraphs}`. `payload()` is a fuller inspection
  view; `to_dict`/`from_dict` for the cache.
- `Book.to_json` dumps chunk payloads for eyeballing; `to_dict`/`from_dict` for the cache.

**Module-level:**
- `parse_rights(file_code)` — reads `<meta name="dc.rights">` straight from the zip,
  independent of the body parse, so the public-domain gate works even for books we never
  segment.
- `build_library(data_path, recall_path)` — the entry point. Backed by the **recall cache**
  (`recall/metadata.json` + `recall/books.json`): parse each `.zip` once, reload from JSON
  forever after. Returns two live dicts, `metadata[code]` and `books[code]`.

---

## Stage 2 — Segment into scenes → `process.py`

**In:** one `Chunk.scene_payload()` (a section of indexed paragraphs + lookback context).
**Out:** `pg{code}-s.json` — a flat list of scene records with **all enrichment fields
`null`**.

The goal: cut each chunk into **flavor-pure scenes** — one dominant tone each, ~300–600
words — and drop non-story "noise".

**Gates (which books to skip):**
- `presegmentation_gate(code, md, …)` — two gates: **US public-domain only** (reads
  `dc.rights` via `parse_rights`) and **non-prose exclusion** (poetry/plays/drama by
  Gutenberg subject). Returns a reason string (already logged) or `None` to proceed.
- `_log_exclusion` — appends the rejected book to `excluded-books.json` for audit.

**LLM segmentation:**
- `SceneData` / `MultiSceneData` (pydantic) — the forced tool-call schema: per segment,
  `start/end_paragraph_index`, `paragraph_type` (`scene` | `noise`), `content_form`
  (`prose` | `other` | `noise`), `open_start_index` / `open_end_index` (cross-section flags),
  `title`.
- `SceneBreaker.break_chunk` — sends one section, forces an `output_scenes` tool call, and
  **retries until coverage passes**. Every retry is a *fresh* conversation (no chat history);
  the paragraphs missed last time are replayed as a system note. Transient API errors back
  off; temperature climbs then freezes; only a fatal API error raises.
- `_expected_indices` / `_validate_coverage` — the coverage check: every indexed paragraph
  covered exactly once (no gaps, dupes, or out-of-input indices), run *before* noise is
  dropped.
- `_retry_note` — builds the segmenter's corrective reminder text; the shared
  `utils.llm.inject_retry_notes` slots it into the system prompt (thread-safe; never mutates
  the module prompt). Enrichment has its own `_retry_note` and reuses the same splice.
- `segment_book(book, checkpoint_base)` — orchestrates one whole book: one `break_chunk` call
  per chunk, run in parallel (`SEGMENT_WORKERS`), each chunk **checkpointed** so a crash
  resumes instead of re-paying. `ex.map` preserves reading order. Returns the flat, ordered
  list of scene objects (noise included — the caller filters).

**Flatten to records:**
- `scenes_to_records(file_code, scenes, book, metadata)` — the shape change. It **stitches**
  scenes that were cut across chunk boundaries (an `open_start` head joins the previous
  chunk's `open_end` tail → `status: "stitched"`; an unmatched head → `"broken_stitch"`),
  rebuilds each scene's `text_html` from its paragraph range, computes `word_count`, and
  emits one flat record per scene. Each record starts from **`schema.blank_record()`** (all
  enrichment fields `null`, `enriched: false`, `schema_version` stamped) and fills only the
  fields segmentation knows. Enrichment fields stay `null` for Stage 3.

> Note: noise removal and the book-level "too much poetry/plays" gate happen in the
> **caller** (`tests.segment_test`), not in `process.py` — see the finale.

---

## Stage 3 — Enrich & Index → `embed.py`

**In:** `pg{code}-s.json` (null enrichment fields).
**Out:** the same file **enriched in place**, plus a Qdrant collection and a SQLite mirror.

### 3a. Enrichment (fill the flavor + frame + summary)

- `SceneEnrichment` / `BatchEnrichment` (pydantic) — the LLM output schema for a *batch*:
  per scene, `dominant_tone`, `intensity`, `arc`, `descriptors` (3–5), the decomposed frame
  `subject`/`verb`/`object`/`setting`, and a general `summary`. Validators clean each field
  (lowercase descriptors, capital-and-period summary, whitespace-collapsed frame).
- `EMBED_PROMPT` (in `utils/llm.py`) / `BATCH_TOOL` — the enrichment instructions + forced tool.
  **This is the user's tuning surface** (do not edit unless asked).
- `_plain` — strips markup for the LLM input (not stored).
- `_batches` — packs scenes into prompt-sized batches by text length (`BATCH_CHAR_LIMIT`),
  never splitting a scene.
- `_run_tool` — one forced, validated tool call reusing the segmenter's retry policy (fresh
  convo per retry, misses replayed via `utils.llm.inject_retry_notes`, temperature climb-then-
  freeze). Generic over `(system_prompt, tool, model_cls, note_fn)`.
- `_enrich_batch` — one LLM call per batch → per-scene `{tags, frame, summary}`, coverage-
  validated (one item per scene, in order).
- `_apply(rec, enriched)` — writes the returned tags/frame/summary onto the record and sets
  `enriched: true`, `enrich_model`.
- `enrich_file(path)` — the driver: resume from `Checkpoint` (skip already-enriched scenes),
  batch the rest through the LLM in parallel (`ENRICH_WORKERS`), checkpoint each scene before
  mutating, then **denormalize neighbor tones** (`prev_tone`/`next_tone` from adjacent
  scenes, for transition search) and rewrite the json in place.

### 3b. Indexing (into both stores, in lockstep)

- `_ensure_collection` — creates the named-vector Qdrant collection (rebuilds if the vector
  set is stale) using `search.VECTOR_NAMES`.
- `index_records(client, records, conn)` — the shape change into the stores:
  1. **SQLite mirror first** (`relational.sql_upsert`) — *every* record, independent of
     enrichment, so relational queries work even before summaries exist.
  2. **Vectors** — embeds `summary`, the `descriptors` vibe string, and the four frame
     fields as **named vectors** (a null frame field falls back to the summary text, so every
     point carries every vector), then upserts one `PointStruct{id, vectors, payload=full
     record}`. `id` is a stable uuid5 of `scene_id`, so re-runs overwrite.

> **Schema drift guard:** at import, `embed.py` asserts `SceneEnrichment`'s field set equals the
> registry's (`schema.LLM_FIELDS`) — so editing the schema can't silently desync what the LLM
> returns from what the index stores.

> The old LLM **query distiller** that used to live here (`QueryFrame` / `distill_query` /
> `QUERY_SYSTEM_PROMPT`) is **deleted**. The read side is now `query.py` (Stage 4), which builds the
> query object with no LLM call.

---

## Stage 4 — Search & Read → `query.py` + `search.py` + `utils/relational.py`

**In:** a writer's query. `query.py` is the front door: `to_query_object(summary)` wraps the raw text
as a **single beat** (`{summary, moments:[{sentence: summary}]}`, no splitting, no facet extraction);
`run()` hands it to `search.search`. Descriptor/flavor and filter args can be passed straight through.
**Out:** ranked `ScoredPoints` (payload = the full scene record) and relational query results.

Two stores answer two different questions, joined on `scene_id`: **Qdrant** ranks by
similarity, **SQLite** answers exact-match / navigation / counts.

### Vector search — `search.py`

Config/primitives shared with the write path: `COLLECTION`, `EMBED_MODEL`, `QUERY_PREFIX`,
`SUBJECT_PATHS_FIELD`, `point_id`, `open_client`, `book_filter`, `embed`. `VECTOR_NAMES` and
`DEFAULT_FIELD_WEIGHTS` are **derived from the schema registry** (the per-field `weight` is
legacy — kept for the drift check; it does not drive the live blend).

- `search` — the **one entry** (imported by `query.py`, tests, evals, webtest). Runs the active
  retrievers over one ANDed filter (`book_id`, `subject_branch`, `tone`/`intensity`/`arc`) and
  merges them by **weighted RRF**. Two knobs: `method_weights` (the scenes-vs-flavor RRF balance,
  default `{"scenes":0.7,"flavor":0.3}`) and `normalize` (z-norm inside the what-happens blend).
- **What-happens + frame** — `search_scenes`: fuse the `summary`, `svos`, and
  `subject`/`verb`/`object`/`setting` channels by a per-channel z-normalized weighted blend, so no
  channel overpowers by raw cosine scale. Split so `evals.py` can re-blend cheaply while tuning:
  - `_channel_queries` / `_moment_sentences` / `_frame_query_terms` — turn the manual query
    (`summary`, `moments`, optional `frame`) into a query vector per active channel (summary → one
    vector; svos + facets → MAX_SIM matrices).
  - `score_channels` — prefetch each channel, then score every channel over one **union** candidate
    pool and `_normalize_pool` (z-score / min-max) it.
  - `blend_channels` — the pure-math half: per-channel z-score × weight, combined by weighted **sum**
    (blend) or **max** (greatest single match); sort.
- **Flavor** — `search_weighted_descriptors`: a weighted descriptor centroid, with optional
  **anti-descriptors** subtracted to tilt away from a flavor. `_unit` / `_check_weights` /
  `weighted_vector` build the centroid (each term L2-normalized first, so none dominates by magnitude).
- `search_frame` — query ONE multivector facet on its own as a MAX_SIM matrix (a scene scores by its
  best-matching stored facet term).
- `_rrf` — weighted reciprocal-rank fusion: reconcile the scenes vs flavor rankings without a shared
  score scale.

### Relational store — `utils/relational.py`

The SQLite mirror; its columns, DDL, filters, and row codec are all **derived from the
schema** (`schema.SQL_COLS`, `create_table_sql`, `to_row`, `inflate_row`, `FILTERABLE`,
`INT_COLS`).

- `open_db` / `_migrate` — open (creating + back-filling any missing column) the on-disk
  mirror; WAL so a reader can query while `embed.py` upserts.
- `sql_upsert` — mirror records (INSERT OR REPLACE on `scene_id`, idempotent).
- `get` — one scene by id, O(1) primary-key lookup.
- `neighbors` — the reading-order window around a scene, O(1) via `UNIQUE(book_id, pos)` —
  the "small-to-big" context fetch after a vector hit.
- `find` / `count` / `tally` — AND-of-equality/`IN` filtering, counting, and `GROUP BY` over
  the whitelisted indexed columns (what Qdrant can't do).

---

## Cross-cutting: the schema registry → `utils/schema.py`

One source of truth for the record shape feeds every stage above:
`process.blank_record`, `relational`'s SQL contract, `search`'s named vectors + weights, and
`embed`'s drift asserts. It also carries the **reconcile tool** that ripples a schema edit
across scenes JSON + SQLite + Qdrant payloads. Full details in [SCHEMA.md](SCHEMA.md).

## Supporting utilities → `utils/`

- `storage.py` — `SrcPaths` (every path constant, incl. `SCHEMA_PATH`) and the atomic-IO
  glue. Importing it loads `.env` once for the whole app.
- `read_write.py` — atomic `read_json`/`write_json`/`read_text`/`write_text` (tmp +
  `os.replace`).
- `checkpoint.py` — `Checkpoint`: the resume cache both LLM stages use (per-item json under a
  per-book dir; `None` on missing/corrupt → recompute; `clear()` on completion).
- `llm.py` — the shared OpenRouter `CLIENT`, `MODEL`, `SCHEMA_VERSION`, `classify_llm_error`
  (transient vs fatal), and `inject_retry_notes` (the retry-note prompt splice both stages reuse)
  — the retry policy's backbone.
- `tags.py` — the controlled `Tone` / `Intensity` / `Arc` vocabularies (the allowed values
  for those fields).

---

## The conductor → `tests.py`

`tests.py` is the hand-run harness (not pytest) that ties every stage together. `main()`
runs the three build steps over `FILE_IDS` (10 active books; more are commented out in the list);
the read-path helpers are
run by hand.

**Build path** (`main`):
1. `step_one_retrieval(FILE_IDS)` — `wget` each `pg{code}-h.zip` into `DATA_DIR`, in parallel,
   waiting for all so Stage 2 never reads a half-finished download.
2. `step_two_processing(FILE_IDS)` — `build_library` (Stage 1), then `segment_test` per book:
   - skip if a scenes json already exists;
   - `presegmentation_gate` (public-domain + non-prose);
   - `segment_book` (Stage 2 LLM segmentation);
   - **drop noise** scenes and tally `content_form == "other"`;
   - **book-level gate**: skip embedding if "other" (poetry/plays) exceeds `OTHER_SKIP_RATIO`
     (70%) of non-noise text;
   - `scenes_to_records` → write `pg{code}-s.json`.
3. `step_three_embedding(FILE_IDS)` — `embed_test`: for each scenes json, `enrich_file`
   (Stage 3a) then `index_records` (Stage 3b) into the local Qdrant + SQLite, and mark the
   book `"completed"` in `status.json` so reruns skip it.

Both long steps run inside `stay_awake()` — a context manager that keeps macOS awake
(`pmset` + `caffeinate`) so a multi-hour segment/embed run survives a closed lid, and
restores normal sleep on exit (even on Ctrl-C).

**Read path** (run by hand):
- `search_test(book_id, limit)` — canned queries against the live index: summary-only
  (`TEST_QUERIES`), summary+weighted-descriptors (`COMBINED_QUERIES`), and pure-descriptor
  (`DESCRIPTOR_QUERIES`); `_show` prints each hit's score, tags, title, summary, descriptors.
- `manual_search(summary, moments, descriptors, ...)` — the hand-driven read path: run the unified
  `search.search()` over a manual summary / moments / descriptors query and print the hits.
- `payload_dump_test()` — dumps every book's chunk payloads for eyeballing (Stage 1 output).

**Related:** [`evals.py`](../evals.py) grades and A/B-tests the read path against the gold
query set — see [SCHEMA.md](SCHEMA.md) and the search-tuning notes.
