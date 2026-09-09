# PLAN.md — Project Alexandria restructure

**Branch:** `restructure` · **Execution manual** — a fresh session can pick it up cold.

> ## ▶ START HERE (new session)
> 1. Binding house style = `CLAUDE.md` → "Code principles" (readability; comment framework; arrows-down;
>    one-import-one-method; single-responsibility). It governs every edit. §1 below is a pointer to it.
> 2. Current phase = first §6 checklist entry not ✅ (**now Phase 7 — `index.py`**).
> 3. Read that phase's **§5 stage design** + its **Appendix A** block (the keep/change/move/drop target).
> 4. Phase loop: **author the new file FROM SCRATCH** to the house style — the old file is a *behavior*
>    reference only (via Appendix A), NEVER a copy-paste port → delete the old file → run the phase's
>    **Checks** → meet **Done-criteria** → update the affected `CLAUDE.md` lines.
> 5. Flip the phase to ✅ in §6, commit. **One phase per session** unless told otherwise; then stop.

**STATUS:** design frozen · D1–D5 resolved (§8) · **Phases 0–6 DONE** (0b kept the 4 facet vectors → 7-vec
set; schema v4 + tags + `vectorstore.py`; `data.gate_facts`; `segment.py` sparse labelling, `process.py`
deleted; `enrich.py` comprehend-before-judge; `derive.py` mechanical word→number pass — `embed.py` kept
until Phase 7). Schema wave: `--check` GREEN via `import enrich`, but `import embed`/`import tests` stay RED
on embed's own stale assert until Phase 7 deletes it. The always-current status table lives in `CLAUDE.md`;
this file drives the *remaining* work (Phases 7–10). **▶ NEXT: Phase 7 — `index.py`** (§5.5, §6, Appendix A
embed-index block).

**What this document is:** the target product + data model (§2–§5) and the ordered file-by-file restructure
that lands it (§6, checklisted against Appendix A). Done phases are one line each; the detail that matters
now is the forward specs.

### Contents
1. Design principles — pointer to `CLAUDE.md`
2. Usage model — the bounty-hunter north star
3. Data model / schema — the target record + the three lanes
4. Architecture — module map + dependency graph
5. Stage designs (5.1–5.4 done; **5.5 index → 5.8 HyDE** are the live specs)
6. Migration plan — Phases 0→10 (0–6 done)
7. Invariants
8. Decisions D1–D5 + ablation
9. Eval plan
- **Appendix A** — per-file keep/change/move/drop (done files collapsed; pending files full)

---

## 1. Design principles (binding)

Live in `CLAUDE.md` → "Code principles" — read them there. In one line each: **readability is the
product**; **keep the comment framework**; **arrows point downward** (helpers above callers, entries last);
**one import = one method** (push composition to the owner; foundation/contract modules exempt); **single
responsibility per file**. They override convenience; if one ever fights genuine best practice, best
practice wins and you note the exception inline.

---

## 2. Product / usage model

North star (grounds every schema/segmentation choice):

> A bounty hunter shifts from patient expert hunting into a quick messy chase = **two scenes**. For the first:
> **Free text:** "an expert hunter analytically waits in hiding, before shooting his bow and barely missing."
> **Controls:** 3rd person · past · low dialogue · "low intensity rising to the bow shot" · "a logical feel."

A request splits into **three lanes**:
- **Semantic (rank by meaning):** the free text — a multi-clause summary containing an ordered beat
  sequence ("waits" → "shoots" → "misses") plus feeling words ("analytically").
- **Hard facets (exclude):** POV, tense, book. Never softened.
- **Soft facets (tilt, never exclude):** prose register, dialogue level, and a **tone curve** ("low rising
  to a peak at the shot") — an affective trajectory, not a scalar.

The unit returned is a **scene** (a dramatic unit with an internal arc) the writer studies to reuse its
feel. Beat-level matching happens *inside* the scene via the moment multivector.

---

## 3. Data model / schema (the target — built in Phase 1; semantics drive Phases 8/9)

### 3.1 Semantic vectors (Qdrant named vectors)

| field | shape | source | notes |
|---|---|---|---|
| `summary` | single | LLM | one richer, multi-clause sentence (request register) |
| `svos` | **multivector** (MAX-SIM) | derived from `moments[].sentence` | the beats |
| `descriptors` | single | LLM | open-vocab vibe (holds non-emotions like "analytical") |
| `subject`/`verb`/`object`/`setting` | **multivector** each | derived from `moments[]` | the 4 facets |

**Committed set = 7 named vectors** (kept per the 0b ablation, §8). **MAX-SIM is order-independent** (all
multivectors): scene = matrix (one vector per moment/term), query = matrix (one per beat), score =
`Σ_qbeat max_scenebeat cos(q, s)`. Do **not** encode sequence into these vectors — beat **order** is carried
by the ordered `vdi_curve` (§3.3), matched positionally in the stage-3 re-rank (§5.6); the `moments[]`
payload keeps reading order for display.

### 3.2 Hard facets (payload filter, categorical)

`book_id` · `pov` (enum, LLM) · `tense` (enum, LLM). Reuse the `facet_filter` payload machinery.
`tone`/`intensity`/`arc` **leave** hard filtering.

### 3.3 Soft facets (payload scalars, stage-3 re-rank)

| field | shape | source | notes |
|---|---|---|---|
| `prose_register` | float 0–1 | LLM **word** → coord | clipped/telegraphic ↔ grand/metaphorical |
| `dialogue_ratio` | float 0–1 | **derived** (quote ratio in `text_html`) | no LLM cost |
| `vdi_curve` | list of `[v,d,i]` | **derived** from `moments[]` words | the affective arc; matched by curve distance (§5.6) |

### 3.4 Moments carry the arc (the load-bearing change)

```
moment = { sentence, subject, verb, object, setting,   # what happens
           tone, intensity }                           # affect, as LLM-picked WORDS
```
- `tone` word → **(valence, dominance)** via a `tags.py` table (circumplex/VAD; dominance separates
  *terror* (low, victim) from *menace* (high, threat)).
- `intensity` word → **i** = presence/tension (faint wash ↔ dominates every line). Kept **separate** from
  the emotion's inherent arousal so a scene can read *low-intensity focused* → *rising tension* while the
  emotion stays analytical. This is the "low rising to the shot."
- `vdi_curve` = the ordered `[(v,d,i)]` over the moments = the arc.

**D1 (resolved):** tone and intensity are **separate words per moment** — two words, three axes;
`vdi_curve[k] = (v_k, d_k, i_k)`.

### 3.5 Two rules that make this cheap

- **Store the word, derive the number.** Records hold the LLM's *word*; the numeric coord is looked up from
  a `tags.py` table and denormalized into the payload (`vdi_curve`, `prose_register`) at **derive** time.
  Re-tuning a slider position = a **payload refresh (no re-enrichment)**; only changing the *vocabulary*
  costs a re-enrich.
- **Derived, not guessed.** `arc`, scene-level intensity, `dialogue_ratio`, `vdi_curve` are all derived. The
  LLM only picks words per beat; the scene shape falls out mechanically and stays consistent with the beats.

### 3.6 Dropped / demoted

- `dominant_tone`, scene `intensity`, `arc` — no longer LLM-authored and no longer hard filters. Keep `arc`
  only as an optional derived display label.
- **D2 (resolved): drop `prev_tone`/`next_tone`.** The two-scene / spanning feature is a future,
  **summary-first** read-path feature: match scene 1 by `summary`, follow `next_scene_id`, test scene 2's
  `summary` against a threshold, only then weigh affect. Needs no denormalized neighbour field —
  `next_scene_id` + the indexed `summary` vectors compose it at read time. So no neighbour-tone denorm.

### 3.7 Schema plumbing — **done in Phase 1**

`float` field type → SQLite `REAL` + codec; derived lists regenerated from `scene_schema.json`; per-field
`weight` retired (D3 — `DEFAULT_WEIGHTS` + parity check gone from `schema.py`; the whole per-field-weight
tuning stack removed from `evals.py`). Search tunes with `method_weights` + the soft-rank knobs (§5.6).

---

## 4. Target architecture (module map)

### 4.1 Foundation — `utils/`

| file | responsibility | change |
|---|---|---|
| `storage.py` | all paths (`SrcPaths`) | add new dirs (adapter weights) |
| `read_write.py` / `checkpoint.py` / `log.py` | IO / resumable cache / logging | none |
| `llm.py` | client / model / prompts / error policy / `inject_retry_notes` | prompts rewritten (owner's surface) |
| `schema.py` | single source of truth (loads `scene_schema.json`, reconcile) | ✅ new `float`/`REAL`, new field set |
| `tags.py` | enums + word→coord tables (tone→VD, intensity→i, prose→register) + `POV`/`Tense` | ✅ expanded |
| `relational.py` | SQLite scene mirror | columns follow schema |
| `subjects.py` | subject-path trie | none |
| `vectorstore.py` | **✅ NEW.** Qdrant contract (`COLLECTION`, named-vector config, `EMBED_MODEL`, `QUERY_PREFIX`, `embed`, `point_id`, `open_client`, filter builders, `SUBJECT_PATHS_FIELD`) | extracted from `search.py` |

`vectorstore.py` removed the old `index → search` (write → read) coupling: both sides import the contract
from the foundation module; neither imports the other.

### 4.2 Pipeline — `src/project_alexandria/`

| file | replaces | one door | responsibility | status |
|---|---|---|---|---|
| `data.py` | `data.py` (kept) | `build_library()`, `ensure_book()`, `gate_facts()` | Stage 1 parse + recall + pre-gate | ✅ |
| `segment.py` | `process.py` (deleted) | `segment_book(book, md) -> records` | Stage 2 boundary-classification → dramatic-unit scenes | ✅ |
| `enrich.py` | `embed.py` enrich half | `enrich_file(path)` | Stage 3a LLM enrichment | ✅ |
| `derive.py` | `embed.py` derive half | `derive_file(path)`, `derive_records(records)` | Stage 3b `svos`/facets/`vdi_curve`/`dialogue_ratio`/`arc`/`prose_register` | ✅ |
| `index.py` | `embed.py` index half | `index_scenes(file_ids)` | Stage 3c build Qdrant + SQLite + subject trie | **Phase 7** |
| `search.py` | `search.py` (rewrite) | `search(...)` | Stage 4: hard filter → semantic rank → soft re-rank | Phase 8 |
| `query.py` | `query.py` (extend) | `run(client, request)` | read front door / normalizer + HyDE hook | Phase 9 |

`embed.py` still exists (enrich/derive/index all still live in it) until **Phase 7 deletes it**.

### 4.3 HyDE — `train/` (Phase 10)

`build_dataset.py` · `fit_adapter.py` (v0 numpy ridge) · `query_adapter.py` · `train_adapter.py` (v1 torch,
gated). Feeds `search` via the existing `channel_vectors` seam. See §5.8.

### 4.4 Dependency graph (arrows = "imports one door from")

```
data ──> (utils)
segment ──> data.gate_facts (one door) , utils.llm , utils.schema
enrich  ──> utils.llm , utils.schema
derive  ──> utils.tags (word→coord) , utils.schema
index   ──> vectorstore , utils.relational , utils.subjects , utils.schema
search  ──> vectorstore , utils.schema
query   ──> search.search (one door) , utils.tags
tests   ──> data.build_library , segment.segment_book , enrich.enrich_file ,
            derive.derive_file , index.index_scenes , search.search   (one door each)
webtest ──> search.search , query.run , utils.subjects
```
No feature file imports another feature file's internals; every cross-feature edge is a single door or a
foundation module.

---

## 5. Stage designs

### 5.1–5.4 — DONE (data / segment / enrich / derive). Code + `CLAUDE.md` invariants are the reference.

- **5.1 `data.py`** — parse/recall unchanged (§7 invariants held). `gate_facts(file_code, md, data_path)`
  is segment's single pre-gate door (folds `parse_rights` + `MetadataParser.to_dict`; policy stays in segment).
- **5.2 `segment.py`** — **sparse boundary labelling**: the model labels ONLY boundary paragraphs —
  `SCENE_START`, ≤1 trailing `SCENE_CONTINUE` (the section's final still-open scene), `NOISE`; every
  unlabelled paragraph is implicit continuation. Reconstruction walks all indices, so scenes + cross-chunk
  stitch + noise-drop fall out of the merged global stream. Book-parallel, chunks **sequential**, threading a
  real continue-flag (`PROCESS_CONTINUE_NOTE`) into the next chunk; the driver runs up to `WORKERS` books at
  once. Soft cap `SOFT_MAX_WORDS=1500`; model targets ~200–1200-word scenes. `scene_title` and the within-book
  non-prose gate removed. One door `segment_book(book, md) -> records` (folds `data.gate_facts`).
- **5.3 `enrich.py`** — comprehend-before-judge `SceneEnrichment`: `summary` (richer, multi-clause) →
  `moments[]` (each `{sentence, subject, verb, object, setting, tone, intensity}`, sentence FIRST, words per
  D1) → `descriptors` (3–5) → `pov` → `tense` → `prose_word`. Moment cap **6** (D4). Import-time drift guard:
  field set == `schema.LLM_FIELDS` = `{summary, descriptors, moments, pov, tense, prose_word}`. Writes LLM
  fields only; `svos`/facets/`vdi_curve`/`dialogue_ratio`/`arc`/`prose_register` are Phase-6 derived. Batched
  (`12000`/`4`) + checkpointed; `_run_tool` splats `MODEL_PARAMS` (tool name already == its `tool_choice`).
  One door `enrich_file(path)`.
- **5.4 `derive.py`** — mechanical no-LLM pass filling every `source:"derived"` field: `svos` (moment
  sentences), the four S/V/O/S facet lists (dedup case-insensitive, order-preserving, `[]`→`None`),
  `vdi_curve` (per-moment tone+intensity WORDS → `[v,d,i]` via `tags.moment_vdi`), `prose_register`
  (`tags.prose_coord`), `dialogue_ratio` (matched straight+smart quote chars ÷ stripped-prose chars, `[0,1]`),
  `arc` (i-axis shape, `ARC_FLAT_BAND=0.15` deadband: mid-curve peak/valley→turn, else net rise/fall→rising/
  falling, else steady; `<2`→steady). Word→number lives ONLY in `tags.py` (retune = derive-only refresh, no
  re-enrich, LLM fields untouched — §3.5); **no** neighbour tones (D2). Two doors: `derive_records(records)`
  (in-place, **IDEMPOTENT** — `index.py`'s pre-embed safety net; field order svos→frame→vdi_curve→
  prose_register→dialogue_ratio→arc, arc last as it reads the curve) and `derive_file(path)` (read → derive →
  write). `import derive` clean (only `utils.tags`/`read_write`/`log`).

### 5.5 Stage 3c — indexing (`index.py`) — **NEXT (Phase 7)**

`index_scenes(file_ids=None)` (single door; `None` = every `pg*-s.json`). Imports the Qdrant contract from
`utils.vectorstore` (`COLLECTION`, `VECTOR_NAMES`, `MULTIVECTOR_NAMES`, `SUBJECT_PATHS_FIELD`, `embed`,
`point_id`, `open_client`, `_as_terms`) and `utils.relational` + `utils.subjects` for SQLite; **imports no
feature file**. Rebuild is explicit — no import-time side effects.

Per-book flow (one client + one SQLite conn for the whole run):
1. **`derive.derive_records(records)`** first — the idempotent pre-embed safety net (§8), so no scene is
   embedded with an un-derived frame/curve. Then persist the derived json back.
2. **SQLite mirror** via `relational.sql_upsert` — **every** record (enriched or not), unlike the vectors.
3. **Vectors** — only scenes with a `summary` become points. Embed the **7 named vectors**: `summary` +
   `descriptors` (single; descriptors joined to one vibe string, summary fallback) and the **5 multivector
   matrices** `svos`/`subject`/`verb`/`object`/`setting` (one vector per term via `_as_terms`, a 1-row
   summary matrix when a field is empty). `_ensure_collection` (re)builds the collection if the vector-name
   set or any field's multivector-ness drifts from the registry; `_ensure_subject_index` adds the
   `subject_paths` keyword index (inert local, live on server).
4. **Payload = the full record**, stamped with `subject_paths` (`subjects.suffixes` over the book's
   `Subjects`) and carrying the soft/hard fields `pov`, `tense`, `prose_register`, `dialogue_ratio`,
   `vdi_curve`. `point_id = uuid5(scene_id)` so a re-run overwrites the point (no dupes).

Then **delete `embed.py`** and repoint its callers (see §6 Phase 7 for the tests.py scope).

### 5.6 Stage 4 — search (`search.py`), the read path — Phase 8

Four stages, only the last is new. Semantic stage + the `channel_vectors` HyDE seam already exist.

```
search(request):
  1  HARD PRE-FILTER  flt = and(book_filter, facet_filter("pov"), facet_filter("tense"))
  2  SEMANTIC RANK    pool = rank(summary, svos, frame, descriptors, flt, limit=PREFETCH)   # PREFETCH >> limit
                       #  what-happens = summary + svos + subject/verb/object/setting, z-norm+blend;
                       #  flavor = descriptors ; the two merged by RRF (channel_vectors= feeds HyDE)
  3  SOFT RE-RANK     if any slider set:
                         U = { m: resample(request.tones, m) for m in MOMENT_RANGE }   # per length
                         for c in pool:
                            pen = w_prose*|c.prose_register - request.prose|
                                + w_dia  *|c.dialogue_ratio - request.dialogue|
                                + w_tone * curve_dist(U[len(c.vdi_curve)], c.vdi_curve)
                            c.score = zpool(c.score) - λ*pen        # tilt, never exclude
                         pool.sort(desc)
  4  return pool[:limit]
```

**Tone-curve math (the novel piece):**
```
resample(tones[k], m):                 # tones[i] = (v,d,i) at position i/(k-1)
   if k == 1: return [tones[0]] * m               # one tone → flat line of length m
   for j in 0..m-1:  u = j/(m-1)                   # linear interp per axis at u
   → m×3 curve

curve_dist(U, S):                      # both length m, values in [0,1]
   mean_j sqrt( wV(ΔV)² + wD(ΔD)² + wI(ΔI)² )  / sqrt(wV+wD+wI)     → bounded [0,1]
```
- User enters **1–5 tone words** → a k-point VDI curve; resampled to every scene length (memoized by `m`,
  ~5 lengths) and compared to each candidate's `vdi_curve` at its own length.
- Soft facets re-rank the semantic pool **via the payload** — no extra vector work — so `PREFETCH` must be
  generous (≈100–200). Orthogonal to HyDE (which only changes stage-2 vectors).
- **Pure-browse mode:** no semantic text → stage 2 becomes a Qdrant `scroll` over the hard filter, stage 3
  ranks that set (same code, different pool source).

**Knobs (all in code, no re-enrichment; D5 defaults — retune on the gold):**
- soft-axis mix `w_tone=0.5, w_prose=0.3, w_dia=0.2`.
- affect-axis mix inside tone distance `wI=0.5, wV=0.3, wD=0.2` ("rising" is mostly an intensity claim).
- global soft strength `λ=0.5` — semantic score is z-normed (~±2), soft penalty bounded [0,1]; a worst-case
  miss costs ~0.5 z, enough to reorder near-ties, not to override a clear semantic winner. `λ` stays fixed in
  code for now (a UI "how strict" control is a later UX decision).

**Refinements (later):** shape term (cosine of mean-subtracted curves) for "rising regardless of baseline";
semantic-aligned tone comparison. Ship absolute + positional first.

### 5.7 Read front door / normalizer (`query.py`) — Phase 9

`run(client, request)` (single door). Turns a writer's request into `search()` inputs:
- **Semantic:** free text → `summary`; pick apart into beats → `moments`; strip *feeling words* →
  `descriptors`. (This is query-side extraction — the one place it is warranted: semantic routing, not the
  SVOS-facet extraction that was declined.)
- **Hard:** `pov`, `tense`, `book_id` pass straight through.
- **Soft:** `prose`/`dialogue` slider values; `tones` = the 1–5 tone words → VDI list.

Single-beat stays the floor; the pick-apart upgrades it. The HyDE adapter plugs in here by emitting
`channel_vectors` instead of (or beside) the text.

### 5.8 HyDE — learned adapter (`train/`, Phase 10)

Per the agreed plan (memory `project-query-normalizer`): generation-free `g(beat) → channel query vectors`,
self-labelled from the corpus, same-book hard negatives; v0 = numpy ridge (no deps), v1 = torch InfoNCE
(gated on v0 lift). Consumes the `channel_vectors` seam already in `search`. **Do HyDE after the schema +
rebuild are frozen**, so the adapter learns the final manifold.

---

## 6. Migration plan (ordered; one step at a time on `restructure`)

**Progress checklist** (current phase = first not ✅):

- ✅ 0  scaffolding — PLAN written; ablation 0b (**KEEP the 4 facet vectors → 7-vec set**, numbers in §8);
  D1–D5 resolved (§8).
- ✅ 1  schema + tags (`utils/`) — schema v4 (7 vec, `pov`/`tense` hard, `prose_register`/`dialogue_ratio`/
  `vdi_curve` soft via float→REAL), tags word→coord tables, `weight` retired (D3).
- ✅ 2  `utils/vectorstore.py` — Qdrant contract extracted from `search`; `search`/`embed`/`schema`
  repointed; `import search` clean.
- ✅ 3  `data.py` — `gate_facts(file_code, md, data_path)` one-door pre-gate added (folds `parse_rights` +
  `MetadataParser.to_dict`); additive, §7 invariants re-confirmed.
- ✅ 4  `segment.py` (deleted `process.py`) — sparse boundary labelling; book-parallel, chunks sequential
  with a real `PROCESS_CONTINUE_NOTE` flag; `SOFT_MAX_WORDS=1500`; `scene_title` + within-book non-prose gate
  removed; one door `segment_book`; `PROCESS_PROMPT` rewritten.
- ✅ 5  `enrich.py` — comprehend-before-judge `SceneEnrichment` (LLM fields only; `pov`/`tense`/`prose_word`,
  per-moment tone/intensity words, cap 6); `EMBED_PROMPT` rewritten + live-confirmed. Detail in §5.3.
- ✅ 6  `derive.py` — mechanical no-LLM pass filling every `source:"derived"` field (svos + facets +
  `vdi_curve`/`prose_register`/`dialogue_ratio`/`arc`); word→number only in `utils.tags`; two idempotent doors.
  Detail in §5.4. (5 + 6: `embed.py` kept until Phase 7 — `--check` GREEN, `import embed`/`tests` RED till then.)
- ☐ **7  `index.py`** ← **NEXT** (delete `embed.py`)
- ☐ 8  `search.py` rewrite
- ☐ 9  `query.py` + harness + `webtest/`
- ☐ 10 full rebuild → HyDE

Each phase: **author the new file(s) FROM SCRATCH** to the new design + house style, **delete** the old file,
run **checks**, meet **done-criteria**, and **update `CLAUDE.md`'s affected lines** (pipeline diagram,
invariants, ownership, run-reference) so the always-loaded map never lies. Diff every phase against
Appendix A — a behavior tagged `[KEEP]` there must still work after the phase. Prompt rewrites (owner's
surface) land after the matching stage code so they target the real tool schemas. The schema wave
(Phases 1–2 + 5–7) is coupled and may be transiently red until Phase 7 closes — expected on this branch.

> A corpus rebuild is pending anyway; this restructure forces it. Sequence: land Phases 1–9, then one clean
> full rebuild on the final schema, then Phase 10 (HyDE).

### Phase 7 — `index.py` (delete `embed.py`) ← NEXT
- **Author `index.py` FROM SCRATCH** to §5.5 (behavior ref = `embed.py`'s index half via Appendix A —
  `_vec_params`/`_ensure_collection`/`_ensure_subject_index`/`_multivector_field`/`index_records`/`index_scenes`
  — do NOT copy-port). Payload gains the soft fields; call `derive.derive_records` as the pre-embed safety net.
- **Delete `embed.py`** — it still holds all three halves (enrich + derive + index) but enrich/derive now live
  in their own files, so nothing of value is lost.
- **Minimal `tests.py` repoint (scope-limited — the FULL tests rewrite is Phase 9):** only what makes the
  imports resolve + the schema wave close. Today tests.py does `from embed import enrich_file, index_records`,
  `import embed`, and calls `embed.index_scenes()` + `embed._ensure_subject_index`/`_point_id`/`COLLECTION`/
  `SUBJECT_PATHS_FIELD` (in `backfill_subject_paths`). Repoint: `enrich_file`→`enrich`, `index_records`/
  `index_scenes`/`_ensure_subject_index`→`index`, the contract symbols (`point_id`/`COLLECTION`/
  `SUBJECT_PATHS_FIELD`)→`utils.vectorstore`. Leave `embed_test`/query shapes/search inputs for Phase 9.
- **Checks:** `python -m utils.schema --check` **green** (the schema wave CLOSES here — `import embed` is gone,
  `import tests` goes GREEN once repointed); `import index` clean; one NEW-schema book indexes; a point's
  payload carries `pov`/`tense`/`prose_register`/`dialogue_ratio`/`vdi_curve`/`subject_paths`; SQLite has every
  record while only enriched scenes are points; a re-run overwrites (no dupes).
- **Run/verify (this machine):** `PYTHONPATH=src/project_alexandria .venv/bin/python` (from repo root).
  **Caveat:** the existing `logs/.../scenes/pg*-s.json` are OLD-schema (moments lack `tone`/`intensity`), so
  exercise end-to-end on a NEW-schema book — `segment` → `enrich.enrich_file` → `derive.derive_file` →
  `index.index_scenes` — or hand-build `schema.blank_record()` records with new-schema `moments`.

### Phase 8 — `search.py` (rewrite read path)
- Hard filters: add `pov`/`tense`, retire `tone`/`intensity`/`arc`. Keep semantic + `channel_vectors`.
- Add stage-3 soft re-rank + `resample`/`curve_dist` (§5.6). Bigger `PREFETCH`.
- **Checks:** hard filters exclude correctly; a tone-curve query reorders sensibly; unset sliders = no effect;
  vector path still == text path (the seam invariant).

### Phase 9 — `query.py` + harness + `webtest/`
- `query.run` normalizer (§5.7). Update `tests.py` (one door per stage), `evals.py` (per-sharpness +
  soft-facet A/B), `webtest/` (slider UI instead of tone/intensity/arc dropdowns).
- **Checks:** end-to-end request → `run` → ranked scenes; gold A/B executes.

### Phase 10 — full rebuild, then HyDE
- One clean rebuild on the frozen schema (re-segment → re-enrich → re-derive → re-index).
- Then `train/` per §5.8, feeding the `channel_vectors` seam.

---

## 7. Invariants to preserve

- **data.py:** global contiguous `Paragraph.index`; lossless extraction + recall round-trip; `_pack` caps;
  sharded lazy recall.
- **segment.py:** every input paragraph accounted for (unlabelled = continuation of the open scene).
- **schema.py:** `scene_schema.json` is the **only** place the field set is defined; every store derives from
  it; never hand-edit the derived lists. `SCHEMA_VERSION` lives only here.
- **vectorstore/search:** `EMBED_MODEL` must match the index; `point_id = uuid5(scene_id)`; bge is asymmetric
  (summary/svos **queries** get `QUERY_PREFIX`; indexed passages + descriptor queries stay raw); multivector
  fields are queried with a matrix.
- **Per-field `weight` is retired** — search tunes with `method_weights` + the soft-rank knobs.
- **Rebuilds are explicit** — importing any pipeline module has no build side effects.

## 8. Decisions (D1–D5 resolved; ablation resolved)

- **D1:** tone word → (valence, dominance) + a **separate** intensity word → i, per moment (§3.4).
- **D2:** drop `prev_tone`/`next_tone`; two-scene spanning is a future summary-first read-path feature over
  `next_scene_id` + the `summary` vectors (§3.6).
- **D3:** retire the per-field `weight` and the entire `evals.py` weight-tuning stack (§3.7).
- **D4:** moment cap = 6 (§5.3).
- **D5:** soft-rank defaults `w_tone/w_prose/w_dia = .5/.3/.2`, `wI/wV/wD = .5/.3/.2`, `λ=.5` fixed in code (§5.6).
- **Also decided:** keep the redundant idempotent `derive` call inside `index` — cheap insurance a scene is
  never indexed with an un-derived frame/curve.
- **Ablation (0b, resolved 2026-09-07): committed set = 7 named vectors.** Adding the 4 facet vectors lifted
  **book@1 to 1.000** (from .920, bias-free human labels), **scene@1 +.19** (.66→.85), scene_mrr +.156, and
  improved *every* one of the 5 sharpness buckets. Head-to-head on top1: ON wins 26, OFF 3, 71 ties. book@1
  uses human ground truth, so "keep" doesn't depend on the scene-label provenance. Driver:
  `scratchpad/ablation_0b.py`.

## 9. Eval plan

- **Segmentation:** spot-check dramatic-unit boundaries vs old flavor-pure cuts on 1–2 books.
- **Vector set (0b):** per-sharpness scene@1, `use_frame` on/off. (Done — §8.)
- **Soft facets:** hold semantic fixed, move one slider, confirm the intended reorder; confirm unset sliders
  are inert.
- **Tone curve:** synthetic "rising" vs "falling" queries retrieve the right arc.
- **HyDE:** per-sharpness lift, no regression on specific queries (`evals.by_sharpness`).
- Reuse the 100-query gold; harden with more queries only once a change shows a small-but-real signal the
  current gold can't resolve.

---

## Appendix A — behavior inventory (keep / change / move / drop)

The diff target for each phase. Done files are collapsed (code is now the reference); **pending files
(index-half, search, query, evals, tests, webtest) are kept in full** — the Phase 7–9 targets.
Legend: **[KEEP]** · **[CHANGE]** · **[MOVE→x]** · **[DROP]** · **[NEW]**.

### Done — collapsed
- **`data.py`** [KEEP, light edits] — parse/recall dataclasses + lossless round-trips + `SceneParser` + lazy
  recall unchanged; added `gate_facts` one door. Details in §5.1 / §6 Phase 3.
- **`process.py` → `segment.py`** [DONE Phase 4] — sparse `ChunkLabels` (`SCENE_START`/≤1 `SCENE_CONTINUE`/
  `NOISE`; unlabelled = continuation), forced `output_labels`; reconstruction + stitch from the merged stream;
  `SceneBreaker.break_chunk` retry loop; `presegmentation_gate` folded into `segment_book`; soft word-cap.
  `process.py` deleted.
- **`embed.py` enrich half → `enrich.py`** [DONE Phase 5] — `Moment` (sentence-first + per-beat tone/intensity
  words) + `SceneEnrichment` (comprehend-before-judge, `pov`/`tense`/`prose_word`, drift guard);
  `BatchEnrichment`/`_run_tool`/`_plain`/`_batches`/`_enrich_batch`/`_apply`/`enrich_file` kept, new field set;
  neighbour-tone denorm dropped (D2).
- **`embed.py` derive half → `derive.py`** [DONE Phase 6] — the old `_derive_frame`/`derive_frame*` facet roll-up
  **[CHANGE]** became the general derive pass authored from scratch: `svos` + the four facet lists, `vdi_curve`
  (tone/intensity words→coords), `prose_register` (word→coord), `dialogue_ratio` (quote ratio), `arc` (curve
  shape). No neighbour tones (D2). Two idempotent doors `derive_records`/`derive_file` (§5.4). `embed.py`'s
  own `derive_frame` still lives until Phase 7 (its callers repoint then).

### `embed.py` index half → `index.py` (via `utils/vectorstore.py`) — **Phase 7 target**
- `_vec_params`/`_ensure_collection` (named-vector config; drop+rebuild if stale). **[MOVE→index.py]**.
- `_ensure_subject_index` (`subject_paths` keyword index). **[KEEP→index.py]**.
- `_multivector_field` (embed a multivector field's per-term matrix, summary fallback — now all 5:
  svos/subject/verb/object/setting). **[KEEP→index.py]**.
- `index_records` (SQLite mirror FIRST for every record, then vectors for enriched-only; payload = full record;
  stamp `subject_paths`; stable `point_id`). **[KEEP→index.py]**; payload gains the soft fields; 7-vector set.
  **Keep** the idempotent `derive_records` call before embedding (safety net, §8).
- `index_scenes` (rebuild driver; one client/conn). **[KEEP→index.py]**.
- Contract imports from `search` (`COLLECTION`, `VECTOR_NAMES`, `MULTIVECTOR_NAMES`, `SUBJECT_PATHS_FIELD`,
  `embed`, `point_id`, `_as_terms`). **[MOVE→utils/vectorstore.py]** (✅ already done Phase 2).

### `search.py` — Stage 4 read → **rewrite in place (Phase 8)**
- Qdrant contract (`COLLECTION`, `EMBED_MODEL`, `QUERY_PREFIX`, `NAMESPACE`, `point_id`, `_embedder`, `embed`,
  `open_client`, `_search_params`, `SUBJECT_PATHS_FIELD`, `facet_filter`, `book_filter`, `subject_filter`).
  **[MOVE→utils/vectorstore.py]** (✅ done Phase 2).
- Flavor: `_unit`/`_check_weights`/`weighted_vector`/`search_weighted_descriptors` (+ anti-descriptors). **[KEEP]**.
- Per-field weights: `DEFAULT_FIELD_WEIGHTS`/`active_field_weights`/`_resolve_field_weights` + `TUNED_WEIGHTS_PATH`.
  **[DROP]** per D3.
- `_as_terms` **[MOVE→vectorstore]**, `_normalize_pool` **[KEEP]**.
- `_moment_sentences`/`_frame_query_terms`/`_channel_queries` (+ `channel_vectors` HyDE seam). **[KEEP]**.
- `score_channels`/`blend_channels`/`search_scenes` (z-norm per channel, weighted blend). **[CHANGE]** blend
  simplifies with weights retired.
- `search_frame` **[KEEP]** (facet vectors kept); `_rrf` **[KEEP]**.
- `tone_filter`/`intensity_filter`/`arc_filter`. **[DROP]** as hard filters.
- `search` (ANDed hard filters + scenes/flavor RRF + `channel_vectors`). **[CHANGE]** swap in `pov`/`tense`,
  retire tone/intensity/arc; **[NEW]** stage-3 soft re-rank + `resample` + `curve_dist` (§5.6).

### `query.py` — read front door → **extend (Phase 9)**
- `to_query_object` (single-beat: whole text → summary + one svos moment). **[CHANGE]** pick text apart into
  beats; strip feeling words → `descriptors`; carry sliders + tone-word list.
- `normalize`/`run` (one door). **[KEEP entry, extend]**; **[NEW]** HyDE hook via `channel_vectors`.

### `evals.py` — A/B + tuning → **update (Phase 9)**
- `load_gold`/`_target`/`_target_scene`. **[KEEP]** (gold gains pov/tense/sliders/tone-curve).
- `score_run` (rank-1 book+scene, MRR/Hit@k), `by_sharpness`, `compare_runs`, `format_comparison`. **[KEEP]**.
- `run_search` (channel-isolation flags), `_gold_frame`/`_gold_moments`. **[CHANGE]** flags for the new lanes +
  a soft-facet A/B path.
- `autolabel_scenes`. **[KEEP]**.
- `collect_vector_channels`/`blend_run`/`coordinate_ascent`/`save_tuned_weights`/`reset_tuned_weights`.
  **[DROP]** with per-field weights (D3); tuning moves to the soft-rank knobs.
- `main` CLI. **[UPDATE]**.

### `tests.py` — build + smoke harness → **update, one door per stage (Phase 9)**
- `FILE_IDS` (10 active). **[KEEP]**. `OTHER_SKIP_RATIO` **[DROP — done Phase 4]**.
- `TEST_QUERIES`/`COMBINED_QUERIES`/`MOMENTS_QUERIES`/`DESCRIPTOR_QUERIES`. **[CHANGE]** new query shapes.
- `subject_sql_test`/`backfill_subject_paths`/`payload_dump_test`. **[KEEP]**.
- `segment_test` **[DONE Phase 4]** — rewired to `segment.segment_book`.
- `_load_status`/`_mark_status`. **[KEEP]**.
- `embed_test`. **[CHANGE]** → enrich + derive + index (the split).
- `_show`/`search_test`/`manual_search`. **[CHANGE]** new search inputs.
- `stay_awake`/`_pmset_disablesleep`. **[KEEP]**.
- `step_one_retrieval` (wget). **[KEEP]**. `step_two_processing`/`step_three_embedding`. **[CHANGE]** to the new
  stage doors. `main`. **[UPDATE]**.

### `main.py` — thin entry → **verify/update** to the new stage doors. **[KEEP/UPDATE]**.

### `webtest/server.py` — local read-path UI (port 8765) → **update (Phase 9)**
- Qdrant lock eviction (`_lock_holders`/`_evict`/`_open_qdrant`). **[KEEP]**.
- `_load_scenes`/`_build_subject_tree`/`_strip`/`_preview`/`_card`/`_pos`. **[KEEP]** (cards may surface new fields).
- `_run_query` (search dispatch + tuning). **[CHANGE]** new search inputs + soft sliders.
- `Handler` routes / `_ctype` / `main`. **[KEEP]**; UI swaps tone/intensity/arc dropdowns for **sliders**.
- imports `utils.tags` for dropdown vocab. **[CHANGE]** → word→coord tables driving sliders.

### `utils/` foundation
- `schema.py` **[✅ done Phase 1]** — new field set, `float`/`REAL` + codec, `weight` retired; reconcile
  machinery + `SCHEMA_VERSION` single-source kept.
- `tags.py` **[✅ done Phase 1]** — `POV`/`Tense` added; word→coord tables (tone→VD, intensity→i, prose→register);
  `Tone` is a word→coord lookup, not a hard-filter enum.
- `vectorstore.py` **[✅ NEW, Phase 2]** — the Qdrant contract, imported by `index.py` (write) + `search.py` (read).
- `relational.py` — mechanics **[KEEP]**; columns follow the new schema.
- `subjects.py` / `checkpoint.py` / `log.py` / `read_write.py` **[KEEP]**.
- `storage.py` **[KEEP]**; add new dirs (adapter weights) in Phase 10.
- `llm.py` — plumbing **[KEEP]**; both prompts rewritten (owner's surface) — `PROCESS_PROMPT` (Phase 4) +
  `EMBED_PROMPT` (Phase 5) done.
