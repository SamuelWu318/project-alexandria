# PLAN.md — Project Alexandria restructure — Phase 10

**Branch:** `restructure`. Phases **0–8.5 are DONE** — their reference is the built code + the `CLAUDE.md`
invariants + memory `project-restructure-plan`; this file no longer re-describes them. It now drives **only
Phase 10**: full rebuild → verify → β-tune → HyDE.

> ## ▶ START HERE (new session)
> 1. **House style = `CLAUDE.md` → "Code principles"** (readability; comment framework; arrows-down;
>    one-import-one-method; single-responsibility). Binding on every edit.
> 2. **Do §2 in order: 10.1 rebuild → 10.2 verify → 10.3 β-sweep → 10.4 HyDE.** The rebuild is the GATE —
>    nothing soft / order-aware / HyDE bites until the new-schema stores exist.
> 3. Author to the house style, run the step's **Checks**, meet **Done**, update the affected `CLAUDE.md`
>    lines (diagram / invariants / ownership / run-ref) so the always-loaded map never lies, flip ✅, commit.
> 4. Running modules on this machine (PYTHONPATH + venv, old-store caveat): memory
>    `reference-restructure-run-verify`.

**Status:** design frozen; D1–D5 + the 0b ablation resolved (§5). `import query/evals/tests/search` GREEN.
`import webtest.server` still fails on the OLD on-disk `scenes.db` (`no such column: pov`) until 10.1.

---

## 1. Phase-10 touchpoints (frozen; full map = `CLAUDE.md`)

- **Stores:** ONE Qdrant collection `scenes`, **7 named vectors** — `summary` + `descriptors` (single, LLM);
  `svos` + `subject`/`verb`/`object`/`setting` (multivector, MAX-SIM, from `moments[]`). `svos` is
  **order-aware** (386-d: 384 semantic + 2 positional, built by `vectorstore.svos_beat_vectors`); the four
  facets stay order-free (384-d). SQLite mirror (`relational.py`) beside the vectors, joined on `scene_id`.
- **Read path:** one `search.search()` (hard filter → semantic pool → soft re-rank → slice), default blend
  `combine="sum"`, fed by the `query.run` front door (SINGLE-BEAT; soft **word→coord** lives in `query.py`;
  `search` stays pure-numeric).
- **HyDE seam — already built:** `search` / `search_scenes` / `score_channels` / `_channel_queries` accept
  `channel_vectors={name: vec}`; a supplied vector overrides the text-derived one (default `None` = unchanged).
  Phase 10.4 fills this seam; no search-side plumbing remains.
- **Module doors** (arrow = "imports one door from"; no feature file imports another's internals):
  ```
  data ──> (utils)                         search  ──> vectorstore , utils.schema
  segment ──> data.gate_facts , utils      query   ──> search.search , utils.tags
  enrich  ──> utils.llm/schema             train/  ──> search channel_vectors seam   (NEW, 10.4)
  derive  ──> utils.tags , utils.schema    tests/evals/webtest ──> the stage doors + search.search + query.run
  index   ──> vectorstore , relational/subjects
  ```

---

## 2. Phase 10 — the work (in order)

### ☐ 10.1 Full rebuild  *(the gate)*
One clean, from-scratch rebuild on the frozen v4 schema with the 8.5 positional `svos` in place.
- **Prep:** move the old `logs/test/` stores aside — `scenes/`, `checkpoints/` (holds `status.json`),
  `databases/` (old `scenes.db` + `qdrant_db`). `data/` (zips) + `recall/` (parse cache) are schema-agnostic:
  keep them to skip re-download/reparse, or move them too for a pure cold start.
- **Run** (`tests.main()` = `step_one_retrieval` → `step_two_processing` → `step_three_embedding` over the 10
  uncommented `FILE_IDS`; each long step self-wraps `stay_awake()`):
  ```
  cd src/project_alexandria && source ../../.venv/bin/activate && python -m tests
  ```
- **Unblocks:** `import webtest.server`; the deferred live webtest UI (10.2); `evals --mode soft`;
  `autolabel_scenes` soft gold.
- **Done:** the 10 books re-segmented + re-enriched + indexed (7 vectors, `svos` 386-d) into fresh
  `scenes.db` + `qdrant_db`; `import webtest.server` loads.

### ☐ 10.2 Verify on the new index  *(§4 eval plan)*
- **webtest live UI** (the Phase-9 deferral): prose/dialogue **sliders** + a 3-beat **tone-curve** + pov/tense
  dropdowns route through `query.py`. `python -m webtest.server` → http://localhost:8765/.
- **`evals --mode soft --axis prose|dialogue|tones`**: hold semantic fixed, move one slider, confirm the
  intended reorder; confirm unset sliders are inert.
- **Tone curve:** synthetic "rising" vs "falling" queries retrieve the right arc.
- **`autolabel_scenes`:** stamp each gold query's target `scene_id` + `pov`/`tense`/`prose_register`/
  `dialogue_ratio`/`vdi_curve`.

### ☐ 10.3 β gold-sweep — order-aware `svos`
The mechanism landed in 8.5 (index + query both build `svos` through `vectorstore.svos_beat_vectors`;
`cos = (1-β)·sem + β·pos`). Only the tune remains, on the rebuilt gold:
- Sweep **β ∈ {0, .1, .2, .3}** (and, if needed, the `p(u)` frequency). Ship the **smallest β** that makes an
  in-order query out-rank the same beats shuffled **without** a `book@1`/`scene@1` regression vs β=0. β=0 is
  the safe fallback.
- **Knob:** `SVOS_POS_BETA` in `utils/vectorstore.py`. A β change alone = a re-index of `svos` only
  (`_ensure_collection` rebuilds on the width/size drift).

### ☐ 10.4 HyDE — learned adapter (`train/`)
Generation-free `g(beat) → channel query vectors`, self-labelled from the corpus, feeding the
`channel_vectors` seam. **Do it AFTER 10.1–10.3** so the adapter learns the final manifold. Adds an
adapter-weights dir to `storage.py`. Four files, one responsibility each:
- **`build_dataset.py`** — self-label from the corpus: each scene's `summary` = the COMPLEX query variant,
  `moments[].sentence` = MEDIUM, `subject`+`verb`+`object` = SPARSE; all three target that scene's stored
  `summary` vector; same-book scenes = hard negatives.
- **`fit_adapter.py`** — v0 numpy ridge (closed-form, no deps).
- **`query_adapter.py`** — apply the learned map at query time → `channel_vectors`.
- **`train_adapter.py`** — v1 torch InfoNCE, **gated on a measured v0 lift**.
- **Eval:** per-sharpness lift (`evals.by_sharpness`); no regression on specific queries.

---

## 3. Invariants Phase 10 must not break  *(full set = `CLAUDE.md` "Load-bearing invariants")*

- `scene_schema.json` is the ONLY place the field set is defined; `SCHEMA_VERSION` lives only there; never
  hand-edit the derived lists.
- `EMBED_MODEL` must match the index; `point_id = uuid5(scene_id)`; bge is asymmetric (summary/`svos`
  **queries** get `QUERY_PREFIX`; indexed passages + descriptor queries stay raw); multivector fields are
  queried with a matrix; **`svos` index + query MUST share `svos_beat_vectors`**.
- The soft **word→coord** lives in `query.py` (`tags`); `search` stays pure-numeric. Per-field `weight` is
  retired (tune with `method_weights` + the fixed soft-rank knobs).
- **Rebuilds are explicit** — importing any pipeline module has no build side effects.

---

## 4. Eval plan  *(reuse the 100-query gold `webtest/gold/test_queries.json`)*

- **Soft facets:** hold semantic fixed, move one slider (`evals --mode soft`), confirm the intended reorder;
  confirm unset sliders are inert.
- **Tone curve:** synthetic "rising" vs "falling" retrieve the right arc.
- **Order-aware `svos`:** in-order query out-ranks the shuffled beats; β=0 reproduces today; no
  `book@1`/`scene@1` regression at the chosen β.
- **HyDE:** per-sharpness lift (`evals.by_sharpness`); no regression on specific queries.
- Harden with more gold queries **only** once a change shows a small-but-real signal the current gold can't
  resolve. (`autolabel_scenes` stamps each query's target scene + its soft facets once a new-schema index exists.)

---

## 5. Decisions already baked (do not relitigate)

- **D1** per-moment `tone` + separate `intensity` words → `(v,d,i)` via `tags.py`.
- **D2** no `prev_tone`/`next_tone` (two-scene spanning is a future summary-first read-path feature).
- **D3** the per-field `weight` and the whole `evals` weight-tuning stack are retired.
- **D4** moment cap = 6.
- **D5** soft-rank constants fixed in `search.py`: `λ=.5`; `w_tone/w_prose/w_dia = .5/.3/.2`;
  `wI/wV/wD = .5/.3/.2`.
- **Blend:** default `combine="sum"` (weight-free) — `max` regressed the 0b gold (`book@1` .86 vs .99),
  kept only as an A/B alt.
- **Ablation (0b):** committed set = **7 named vectors** — adding the 4 facet vectors lifted `book@1` to
  1.000 (from .920, human labels) and improved every sharpness bucket.
- Keep the redundant **idempotent `derive` call inside `index`** (cheap insurance).
