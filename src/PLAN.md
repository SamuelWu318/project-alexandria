# PLAN.md — Project Alexandria restructure

**Branch:** `restructure` · **This is an execution manual** — a fresh session can pick it up cold.

> ## ▶ START HERE (for a new session)
> 1. Read **§1 Design principles** — binding on every edit you make.
> 2. Find the **current phase**: the first entry in the §6 Progress checklist not marked ✅.
> 3. For that phase read its **§5 stage design** *and* its **Appendix A** block (the keep/change/move/drop
>    checklist for the file it touches).
> 4. Run the phase loop: **create** the new file → **port** survivors (rewritten to §1 — comment
>    framework, downward-only method order, one-door imports) → **delete** the old file → run the phase's
>    **Checks** → meet its **Done-criteria**.
> 5. Flip that phase to ✅ in the §6 Progress checklist, update the **Status** line just below, commit.
> 6. **One phase per session** unless told otherwise; then report and stop.
>
> **STATUS (single source of truth):** design frozen · **D1–D5 resolved (§8)** · **0b: committed set =
> 7 named vectors** · **Phase 1 DONE (2026-09-07): `scene_schema.json` v4 + `utils/schema.py` +
> `utils/tags.py` rewritten to the target** — 7 vectors, `pov`/`tense` hard facets, `prose_register`/
> `dialogue_ratio`/`vdi_curve` soft facets (float→REAL codec), per-moment tone/intensity words + `tags.py`
> word→coord tables, per-field `weight` retired (D3).
> **▶ NEXT ACTION: Phase 2 — extract `utils/vectorstore.py`** (Qdrant contract out of `search.py`).
> Note: the schema wave is RED (`import search` / `import embed` break on the retired weight; `--check`
> reports the lag) until Phase 7 closes it — expected on this branch.

**What this document is:** the one reference for (a) the redesigned product + data model (§2–§5) and
(b) the exact, ordered, file-by-file restructure that lands it (§6, checklisted against Appendix A).
Design is frozen; the work now is §6, top to bottom.

### Contents
1. **Design principles (binding)** — read first, every session
2. Usage model — why the design is shaped this way (the bounty-hunter example)
3. Data model / schema — the target record + the three lanes
4. Architecture — module map + the dependency graph after the move
5. Stage designs — the detailed spec for each file you will write (§5.1 → §5.8)
6. **Migration plan** — the ordered phases you execute (Phase 0 → 10)
7. Invariants — must survive the move
8. Decisions — D1–D5 (resolved) + the one open ablation
9. Eval plan — how each risky change is checked
- **Appendix A** — per-file behavior inventory (keep / change / move / drop): the diff target for each phase

---

## 1. Design principles (binding)

These override convenience. If one ever fights genuine best practice, best practice wins — and you note
the exception inline.

1. **Readability is the product.** An extra line that makes intent obvious beats a clever one-liner.
   No massive shortcuts. Names read like the surrounding code.
2. **Keep the comment framework** (see `CLAUDE.md` → "Comment framework"). Every method gets one
   matter-of-fact `#` summary line above it; stack `# ** LOCKED **`, `# ** MAIN ** — <who calls it>`,
   `# ** ENTRY **`, and `# ---- segment ----` markers as defined there. `MAIN` methods annotate their
   cross-module calls inline so the call graph is legible without opening the callee.
3. **Arrows point downward.** Within a file, order methods so a reader falls straight through:
   helpers above the method that uses them, entry points last (or clearly grouped). No ping-pong where
   you must scroll up-and-down to follow one path.
4. **One import = one method (the low-dependency rule).** When file B needs behaviour from file A, B
   should import **one** method from A that encompasses everything B needs — not five helpers. If B
   finds itself composing several of A's functions, that composition belongs **in A** as the single
   method B calls. Push the work to the owner; the consumer takes one door.
   - *Exception:* genuine **foundation/contract** modules (schema registry, the vector-store contract,
     IO, logging, tag tables) are legitimately multi-symbol — they are shared vocabulary, not feature
     dependencies. The rule targets feature-to-feature coupling.
5. **Single responsibility per file.** A file is one stage or one concern. `embed.py` currently does
   enrichment *and* derivation *and* indexing — three concerns — and is split below.

---

## 2. Product / usage model (what a user actually does)

Grounding example (the north star for every schema/segmentation choice):

> Writing a scene: a bounty hunter shifts from patient, expert hunting into a quick, messy chase.
> That is **two scenes**. For the first:
> **Free text:** "a scene where an expert hunter analytically waits in hiding, before shooting his bow
> and barely missing his target."
> **Controls:** 3rd person · past tense · low dialogue · "low intensity rising up to the bow shot" ·
> "a more logical feel."

What that tells us about a request — it splits into **three lanes**:

- **Semantic (rank by meaning):** the free text. It is a *multi-clause summary* that also contains an
  ordered *beat sequence* ("waits" → "shoots" → "misses"), plus *feeling words* ("analytically").
- **Hard facets (exclude):** POV, tense (and book). Never softened.
- **Soft facets (tilt, never exclude):** prose register ("logical/grand"), dialogue level, and a
  **tone curve** ("low rising to a peak at the shot") — an affective trajectory, not a scalar.

The unit returned is a **scene** (a dramatic unit with an internal arc) that the writer studies to
reuse its feel. Beat-level matching (the "shoots his bow" clause hitting the scene's shot beat) happens
*inside* that scene via the moment multivector.

---

## 3. Data model / schema (the target)

Three lanes, each stored and queried differently.

### 3.1 Semantic vectors (Qdrant named vectors)

| field | shape | source | notes |
|---|---|---|---|
| `summary` | single vector | LLM | one **richer, multi-clause** sentence (closer to a real request's register) |
| `svos` | **multivector** (MAX-SIM) | derived from `moments[].sentence` | the beats; a query beat hits its best-matching scene beat |
| `descriptors` | single vector | LLM | open-vocabulary vibe (holds non-emotions like "analytical" that the tone axes cannot) |

**MAX-SIM is order-independent** (applies to `svos` and all four facet multivectors): a scene stores a
matrix (one vector per moment/term), the query is a matrix (one per beat), and the score is
`Σ_qbeat max_scenebeat cos(q, s)` — each query beat greedily grabs its single best scene beat regardless
of position, and query/scene matrix lengths need not match. Do **not** try to encode sequence into these
vectors. Beat **order** is carried separately by the ordered `vdi_curve` (§3.3), matched positionally in
the stage-3 re-rank (§5.6); the `moments[]` payload keeps reading order for display.

`subject`/`verb`/`object`/`setting` as **separate** multivectors were **on probation**. **Ablation gate
(Phase 0b, §6) RESOLVED 2026-09-07: KEEP them.** On the current stores, adding the 4 facet vectors lifted
**book@1 to a perfect 1.000** (from .920 — bias-free, book labels are human ground truth) and **scene@1
by +.19** (.66→.85), improving *every* one of the 5 sharpness buckets. The committed set is therefore
**7 named vectors** (the three above + the four facets), not three. (Phase 8 search must retain the frame
what-happens channel accordingly.)

### 3.2 Hard facets (payload filter, categorical)

`book_id` (core) · `pov` (enum, LLM) · `tense` (enum, LLM). These reuse the existing
`facet_filter` payload machinery. `tone`/`intensity`/`arc` **leave** hard filtering.

### 3.3 Soft facets (payload scalars, stage-3 re-rank)

| field | shape | source | notes |
|---|---|---|---|
| `prose_register` | float 0–1 | LLM **word** → coord | clipped/telegraphic ↔ grand/metaphorical |
| `dialogue_ratio` | float 0–1 | **derived** (quote ratio in `text_html`) | no LLM cost |
| `vdi_curve` | list of `[v,d,i]` | **derived** from `moments[]` words | the affective arc; matched by curve distance (§5.6) |

### 3.4 Moments carry the arc (the load-bearing change)

Each moment becomes both a semantic beat and an affective sample:

```
moment = { sentence, subject, verb, object, setting,   # what happens
           tone, intensity }                           # affect, as LLM-picked WORDS
```

- `tone` word → **(valence, dominance)** via a `tags.py` table (the emotion's position on the circumplex/
  VAD map — dominance is what separates *terror* (low, victim) from *menace* (high, threat)).
- `intensity` word → **i** = *presence/tension* (faint wash ↔ dominates every line). Kept **separate**
  from the emotion's inherent arousal so the hunting scene can read *low-intensity focused* → *rising
  tension* while the emotion stays analytical. This is the "low rising to the shot."
- The scene's `vdi_curve` = the ordered `[(v,d,i)]` over its moments = the arc.

**D1 — RESOLVED: tone and intensity are separate words per moment.** The `tone` word picks a point on a
**2-D valence×dominance grid** → (v, d); the `intensity` word → i (presence/tension). Two words per
moment, three axes; `vdi_curve[k] = (v_k, d_k, i_k)`. (UI later exposes the tone grid + an intensity
slider — not this plan's concern.)

### 3.5 Two schema rules that make this cheap

- **Store the word, derive the number.** Records hold the LLM's *word*; the numeric coord is looked up
  from a `tags.py` table and **denormalized into the payload** (`vdi_curve`, `prose_register`) at derive
  time — like `prev_tone`/`next_tone` today. Re-tuning a slider position is then a **payload refresh
  (no re-enrichment)**; only changing the *vocabulary* costs a re-enrich.
- **Derived, not guessed.** `arc`, scene-level intensity, `dialogue_ratio`, `vdi_curve` are all derived.
  The LLM's job shrinks to picking words per beat; the scene-level shape falls out mechanically and
  therefore stays consistent with the beats.

### 3.6 Fields dropped / demoted

- `dominant_tone`, scene `intensity`, `arc` — no longer LLM-authored **and** no longer hard filters.
  Keep `arc` only as an optional derived display label.
- **D2 — RESOLVED: drop `prev_tone`/`next_tone`.** The two-scene / spanning feature (a request whose
  extended summary crosses a scene boundary) is a **future read-path feature**, and it is **summary-first,
  not tone-first**: match scene 1 by `summary`, follow its `next_scene_id`, test whether scene 2's
  `summary` clears a similarity threshold, and only then weigh tone/affect. That needs no denormalized
  neighbour field — `next_scene_id` (already stored) + the indexed `summary` vectors compose it at read
  time. So the neighbour-tone denormalization in `enrich_file` is removed.

### 3.7 Schema plumbing this requires (`utils/schema.py`)

- A new `float` field type → SQLite `REAL`, with a store codec.
- Regenerate the derived lists (`VECTOR_NAMES`, `SQL_COLS`, `FILTERABLE`, …) from the new
  `scene_schema.json`.
- **D3 — RESOLVED: retire the per-field `weight`.** Drop `DEFAULT_WEIGHTS` + its parity check from
  `schema.py`, and the whole per-field-weight tuning stack from `evals.py` (`collect_vector_channels`,
  `blend_run`, `coordinate_ascent`, `save_tuned_weights`, `reset_tuned_weights`, `TUNED_WEIGHTS_PATH`).
  Search tunes with `method_weights` + the soft-rank knobs (§5.6), not per-vector weights.

---

## 4. Target architecture (module map)

### 4.1 Foundation — `utils/` (edited in place; mostly stable)

| file | responsibility | change |
|---|---|---|
| `storage.py` | all paths (`SrcPaths`) | add any new dirs (adapter weights, etc.) |
| `read_write.py` | atomic JSON/text IO | none |
| `checkpoint.py` | resumable per-item cache | none |
| `log.py` | logging | none |
| `llm.py` | client / model / prompts / error policy / `inject_retry_notes` | prompts rewritten (owner's surface) |
| `schema.py` | **single source of truth** (loads `scene_schema.json`, reconcile) | new `float`/`REAL`; new field set |
| `tags.py` | enums **+ word→coord tables** (tone→VD, intensity→i, prose→register) + `POV`, `Tense` | expanded |
| `relational.py` | SQLite scene mirror | payload/column set follows schema |
| `subjects.py` | subject-path trie (folder browse) | none (keep separate — distinct concern) |
| `vectorstore.py` | **NEW.** Qdrant contract: `COLLECTION`, named-vector config, `EMBED_MODEL`, `QUERY_PREFIX`, `embed`, `point_id`, `open_client`, payload-filter builders, `SUBJECT_PATHS_FIELD` | extracted from `search.py` |

**Why `vectorstore.py`:** today `index` (write) imports 7 symbols out of `search` (read) — a
feature→feature dependency in the wrong direction. Extracting the contract into a foundation module both
sides import removes that coupling entirely (principle #4). `search.py` becomes read-only logic;
`index.py` becomes write-only logic; neither imports the other.

### 4.2 Pipeline — `src/project_alexandria/` (new files; port; delete old)

| new file | replaces | single entry (the "one door") | responsibility |
|---|---|---|---|
| `data.py` | `data.py` (kept, light edits) | `build_library()`, `ensure_book()` | Stage 1: parse zip → Book/Chunk/Paragraph + recall cache |
| `segment.py` | `process.py` (delete) | `segment_book(book) -> records` | Stage 2: **boundary-classification** segmentation → dramatic-unit scenes → flat records |
| `enrich.py` | `embed.py` enrichment half | `enrich_file(path)` | Stage 3a: LLM enrichment (summary, moments+tone/intensity words, descriptors, pov, tense, prose word) |
| `derive.py` | `embed.py` derive half | `derive_file(path)` | Stage 3b: `svos`, `vdi_curve`, `dialogue_ratio`, `arc`, neighbour tones — all word→number / text→number |
| `index.py` | `embed.py` index half | `index_scenes(file_ids)` | Stage 3c: build Qdrant + SQLite + subject trie |
| `search.py` | `search.py` (rewrite) | `search(...)` | Stage 4 read: hard filter → semantic rank → **soft re-rank** |
| `query.py` | `query.py` (extend) | `run(client, request)` | read front door / normalizer (§5.7) + HyDE hook |

`main.py`, `tests.py`, `evals.py`, `webtest/` are updated to consume the new one-door entries; details
in §6.

### 4.3 HyDE — `train/` (new package, later; see the HyDE plan in memory + §5.8)

`build_dataset.py` · `fit_adapter.py` (v0, numpy ridge) · `query_adapter.py` (runtime) ·
`train_adapter.py` (v1, torch — gated). Feeds `search` via the existing `channel_vectors` seam.

### 4.4 Dependency graph after restructure (arrows = "imports one door from")

```
data ──> (utils)
segment ──> data.parse_rights/metadata (one door) , utils.llm , utils.schema
enrich  ──> utils.llm , utils.schema
derive  ──> utils.tags (word→coord) , utils.schema
index   ──> vectorstore , utils.relational , utils.subjects , utils.schema
search  ──> vectorstore , utils.schema
query   ──> search.search (one door) , utils.tags (word→coord for the user curve)
tests   ──> data.build_library , segment.segment_book , enrich.enrich_file ,
            derive.derive_file , index.index_scenes , search.search   (one door each)
webtest ──> search.search , query.run , utils.subjects
```

No feature file imports another feature file's internals; every cross-feature edge is a single entry
or a foundation module.

---

## 5. Stage designs (detail)

### 5.1 Stage 1 — parse (`data.py`)

Largely unchanged. Preserve the invariants (§7): global `Paragraph.index`, lossless extraction, `_pack`
caps, lossless recall round-trip, sharded lazy recall. The new segmenter is **per-paragraph**, so the
one thing to guarantee is that a chunk payload cleanly exposes its paragraphs in `index` order with
stable indices. `segment` should need exactly one door from `data` for its pre-gate (public-domain +
metadata); if it currently reaches for `MetadataParser` *and* `parse_rights`, fold both into a single
`data`-side helper the gate calls (principle #4).

### 5.2 Stage 2 — segmentation (`segment.py`, replaces `process.py`)

**New formulation: per-paragraph boundary classification.** For each paragraph in a chunk the LLM emits
one label — the model reads the whole chunk (global input) but answers locally (per-paragraph output):

- `SCENE_START` — a new dramatic unit begins here (place / time / POV / goal shift)
- `CONTINUE` — same scene as the previous paragraph
- `NOISE` — apparatus / editorial → dropped

Why this beats span-emission: **coverage is automatic** (exactly one label per paragraph — no gaps,
overlaps, or out-of-range indices), so the whole coverage-retry apparatus (`_expected_indices` /
`_validate_coverage`) shrinks to a trivial "every paragraph labelled" check. The retry loop stays only
for transient API errors + malformed output.

Scene reconstruction + cross-chunk stitching **fall out of the labels**:
- Group runs of `CONTINUE` after a `SCENE_START` into one scene.
- If a chunk's first paragraph is `CONTINUE`, that scene is a **continuation** of the prior chunk's tail
  (stitch). If the chunk ends mid-scene, that scene is **open-ended** (await the next chunk).
- `stitch_status` is derived from these edge conditions, not asked of the model.

**Soft word cap (deterministic post-process).** After labelling, if a reconstructed scene exceeds
`SOFT_MAX_WORDS`, insert a soft cut at the nearest paragraph break. The LLM finds *dramatic* boundaries;
the cap is a mechanical safety valve so retrieved scenes never run aggressively long. Accept that a
capped scene may split one dramatic unit — the moments still capture its beats.

`segment_book(book) -> records` is the single door: it runs the pre-gate, segments every chunk (parallel,
checkpointed), stitches, applies the cap, drops noise, and returns flat records from
`schema.blank_record()` (enrichment null). Keep it DB-agnostic. The presegmentation gate
(US-public-domain via `dc.rights`, non-prose subject) folds inside this door so `tests` calls one thing.

Prompt (`PROCESS_PROMPT`, owner's surface): rewrite from "flavor-pure scenes" to "dramatic-unit
boundaries, consistent with scene-craft literature."

### 5.3 Stage 3a — enrichment (`enrich.py`)

`enrich_file(path)` (single door), batched + checkpointed as today. Per scene the LLM returns, **in
comprehend-before-judge order** (understanding first makes the judgments reliable — the same principle as
today's "moment writes the sentence *then* extracts SVOS"):

1. `summary` (richer, multi-clause)
2. `moments[]` — each `{sentence, subject, verb, object, setting, tone, intensity}` (words per D1)
3. `descriptors` (3–5, open vocabulary)
4. `pov`, `tense`
5. `prose_register` (a word)

`Moment` / `SceneEnrichment` pydantic models mirror this; the import-time drift guard stays
(`SceneEnrichment` field set == `schema.LLM_FIELDS`). Reuse `_run_tool` + `inject_retry_notes` from
`utils.llm`. **D4 — RESOLVED: moment cap = 6** (dramatic units are longer; the `Moment` validator's cap
rises from 3 to 6). Prompt (`EMBED_PROMPT`, owner's surface): rewrite for the word-picking + per-beat affect.

### 5.4 Stage 3b — derivation (`derive.py`)

`derive_file(path)` (single door). Everything mechanical, no LLM:

- `svos` ← the `moments[].sentence` list (the multivector rows).
- `vdi_curve` ← map each moment's `tone`/`intensity` words → coords via `tags.py` tables.
- `dialogue_ratio` ← quote-character ratio over `text_html`.
- `arc` ← classify the `vdi_curve` intensity axis (up=rising, down=falling, flat=steady, sign-change=turn).
- `prose_register` (float) ← `tags.py` prose word→coord.
- neighbour tones (if kept, D2) ← denormalize from adjacent scenes.

Because these are pure word→number / text→number, re-tuning the `tags.py` tables re-runs **derive only**
(a payload refresh), never enrichment.

### 5.5 Stage 3c — indexing (`index.py`)

`index_scenes(file_ids)` (single door). Uses `vectorstore.py` for the Qdrant contract and
`utils.relational` + `utils.subjects` for SQLite. Order preserved from today: **SQLite mirror first**
(every record, enriched or not), then vectors (the 7 named vectors — `summary`/`svos`/`descriptors` +
`subject`/`verb`/`object`/`setting`, per 0b), one `PointStruct` per
scene with the full payload (**including** `pov`, `tense`, `prose_register`, `dialogue_ratio`,
`vdi_curve`, and `subject_paths`). `point_id = uuid5(scene_id)` so re-runs overwrite. Rebuild is
explicit (no import-time side effects — already true).

### 5.6 Stage 4 — search (`search.py`), the read path

Four stages, only the last is new. Semantic stage and the `channel_vectors` HyDE seam are already built.

```
search(request):
  1  HARD PRE-FILTER  flt = and(book_filter, facet_filter("pov"), facet_filter("tense"))
  2  SEMANTIC RANK    pool = rank(summary, moments, frame, descriptors, flt, limit=PREFETCH)   # PREFETCH >> limit
                       #  what-happens = summary + svos + subject/verb/object/setting (all per 0b), z-norm+blend;
                       #  flavor = descriptors ; the two methods merged by RRF (channel_vectors= feeds HyDE)
                       #  (channel_vectors= lets the HyDE adapter feed pre-embedded query vectors)
  3  SOFT RE-RANK     if any slider set:
                         U = { m: resample(request.tones, m) for m in MOMENT_RANGE }   # precompute per length
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

- The user enters **1–5 tone words** → a k-point VDI curve; it is resampled to every scene length
  (memoized by `m`, only ~5 lengths) and compared to each candidate's `vdi_curve` at its own length.
- Soft facets **re-rank the semantic pool via the payload** — no extra vector work — so `PREFETCH` must
  be generous (≈100–200) to give the tilt room. Orthogonal to HyDE (which only changes stage-2 vectors).
- **Pure-browse mode:** if there is no semantic text, stage 2 becomes a Qdrant `scroll` over the hard
  filter and stage 3 ranks that set (same code, different pool source).

**Knobs (all in code, none require re-enrichment). D5 — RESOLVED (defaults chosen; retune on the gold):**
- soft-axis mix `w_tone=0.5, w_prose=0.3, w_dia=0.2` — tone is the richest, most-intentional soft signal;
  prose/dialogue are secondary style filters.
- affect-axis mix inside the tone distance `wI=0.5, wV=0.3, wD=0.2` — "rising" is mostly an intensity
  claim; valence (positive vs negative feel) next; dominance is the subtlest.
- global soft strength `λ=0.5`. Rationale: the semantic score is z-normalized (roughly spans ±2), the soft
  penalty is bounded [0,1]; at `λ=0.5` a worst-case soft miss (1.0) costs ~0.5 z — enough to reorder
  near-ties, not enough to override a clear semantic winner. Soft facets **tilt**, never dominate.
- `λ` stays **fixed in code** for now; exposing it as a UI "how strict are the sliders" control is a later
  UX decision, not a retrieval one.

**Refinements (later):** shape term (cosine of mean-subtracted curves) for "rising regardless of
baseline"; semantic-aligned tone comparison (compare tone at the beat that *semantically* matched, not
by position). Ship absolute + positional first.

### 5.7 Read front door / normalizer (`query.py`)

`run(client, request)` (single door). Turns a writer's request into `search()` inputs:

- **Semantic:** the free text → `summary`; pick it apart into beats → `moments`; strip the *feeling
  words* → route to `descriptors`. (This is query-side extraction — the one place it is warranted; it is
  semantic routing, not the SVOS-facet extraction that was declined.)
- **Hard:** `pov`, `tense`, `book_id` pass straight through.
- **Soft:** `prose` and `dialogue` slider values; `tones` = the 1–5 tone words → VDI list.

Single-beat stays the floor; the pick-apart is what upgrades it. The HyDE adapter, when built, plugs in
here by emitting `channel_vectors` instead of (or beside) the text.

### 5.8 HyDE — learned adapter (`train/`, later)

Unchanged from the agreed plan (see memory `project-query-normalizer`): generation-free `g(beat) →
channel query vectors`, self-labelled from the corpus, same-book hard negatives; v0 = numpy ridge
(no deps), v1 = torch InfoNCE (gated on v0 lift). It consumes the `channel_vectors` seam already in
`search`. **Do HyDE after the schema + rebuild are frozen**, so the adapter learns the final manifold.

---

## 6. Migration plan (ordered; one step at a time on `restructure`)

**Progress checklist** (the current phase = the first one not ✅; flip to ✅ when its Done-criteria pass):

- ✅ 0a  PLAN.md written
- ✅ 0b  vector-set ablation — **KEEP the 4 facet vectors (7-vec set)**; numbers in the Phase 0 block
- ✅ 0c  D1–D5 resolved (§8)
- ✅ 1  schema + tags (`utils/`) — schema v4 (7 vec, pov/tense, soft facets, float/REAL), tags word→coord tables, weight retired
- ☐ **2  `utils/vectorstore.py`** ← **NEXT**
- ☐ 3  `data.py`
- ☐ 4  `segment.py` (delete `process.py`)
- ☐ 5  `enrich.py`
- ☐ 6  `derive.py`
- ☐ 7  `index.py` (delete `embed.py`)
- ☐ 8  `search.py` rewrite
- ☐ 9  `query.py` + harness + `webtest/`
- ☐ 10 full rebuild → HyDE

Each phase: **create** new file(s), **port** the survivors from the old file (rewritten to the new design
+ comment framework + downward ordering), **delete** the old file, then run the phase's **checks** and
meet its **done-criteria**. **Diff every phase against Appendix A** — that is the keep/change/move/drop
checklist for the file(s) it touches; a behavior tagged `[KEEP]` there must still work after the phase.
**Every phase's done-criteria also include: update `CLAUDE.md`'s affected architecture lines (pipeline
diagram, invariants, ownership, run-reference) + the `docs/` for what the phase landed**, so the
always-loaded map never lies. The binding house style now lives in `CLAUDE.md` → "Code principles"
(promoted from §1); §1 here is the same rules, kept for local reference. Keep the tree importable at phase boundaries where noted; the schema wave
(Phases 1–2 + 5–7) is coupled and may be transiently red until Phase 7 closes — that is expected on a
restructure branch.

> Reality check first: a **corpus rebuild is pending anyway**. This restructure changes the record
> shape, so it *forces* a rebuild. Sequence: land Phases 1–9, then one clean full rebuild on the final
> schema, then Phase 10 (HyDE).

### Phase 0 — scaffolding
- 0a. This `PLAN.md` committed.
- 0b. **Ablation gate** (deps-free, decides §3.1) — **DONE 2026-09-07.** Ran `evals.run_search(use_frame=False)`
  (summary+svos, the 3-vec candidate) vs `True` (+ the 4 `subject/verb/object/setting` facet vectors) over
  the 100-query gold on the current stores; `combine=sum, normalize=zscore`. **Outcome: KEEP the facets →
  the committed set is 7 named vectors, NOT 3** (the plan's original "collapse to 3" default is overturned
  by the data). Numbers (OFF = 3-vec, ON = 7-vec):
  | metric | OFF | ON | Δ |
  |---|---|---|---|
  | scene@1 (aggregate) | .660 | .850 | +.190 |
  | book@1 (aggregate, bias-free) | .920 | **1.000** | +.080 |
  | scene_mrr | .745 | .901 | +.156 |
  | top1 composite | .790 | .925 | +.135 |

  Per-sharpness **scene@1** (1 unique .. 5 generic): OFF .65/.80/.55/.70/.60 → ON .80/.95/.85/.80/.85
  (**every bucket up**, +.10 to +.30). Per-sharpness **book@1** (bias-free): ON = 1.000 in all 5 buckets
  (OFF .95/.95/.95/.95/.80). Head-to-head on top1: frame ON wins 26, OFF wins 3, 71 ties.

  Caveat handled: gold `target_scene_id` was auto-labelled by a frame-ON search, so scene@1 mildly favours
  ON by construction — but **book@1 uses human ground-truth labels and still improves to a perfect 1.000**,
  so the "keep" decision does not depend on the scene-label provenance. Driver:
  `scratchpad/ablation_0b.py` (uses `evals.run_search` / `score_run` / `by_sharpness`).
- 0c. **DONE** — D1–D5 resolved (§8); answers written into §3/§5. Only the 0b ablation remains open.

### Phase 1 — schema + tags (`utils/`)
- Rewrite `scene_schema.json` to §3 (moments+affect words, `prose_register`, `dialogue_ratio`,
  `vdi_curve`, `pov`, `tense`; drop the demoted fields; vector set per 0b).
- `utils/schema.py`: add `float`/`REAL` type + codec; regenerate derived lists; apply D3 (weight).
- `utils/tags.py`: add `POV`, `Tense` enums; add word→coord tables (tone→VD, intensity→i, prose→register).
- **Checks:** `python -m utils.schema --check` green (against the new consumers once they exist — until
  then, `--check` will flag the not-yet-updated stores; that is the schema wave).
- **Done:** schema file + tables are the agreed shape; `tags.py` tables round-trip word↔coord.

### Phase 2 — `utils/vectorstore.py`
- Extract the Qdrant contract out of `search.py` into `vectorstore.py` (§4.1). Update `search.py` to
  import from it (temporary — `search.py` is rewritten in Phase 8 anyway).
- **Checks:** `import search`, `import utils.vectorstore` clean.
- **Done:** the contract has exactly one home; nothing imports it from `search`.

### Phase 3 — `data.py`
- Light edits: confirm §7 invariants; add the single pre-gate door `segment` needs (§5.1).
- **Checks:** `build_library()` + `ensure_book()` round-trip a book unchanged.

### Phase 4 — `segment.py` (delete `process.py`)
- Implement per-paragraph boundary classification + reconstruction + stitch + soft cap (§5.2).
- Port `scenes_to_records` (flattening/stitch) into `segment.py`, rewritten. Delete `process.py`.
- **Checks:** segment one book; every input paragraph labelled once; scenes are dramatic units; capped
  lengths ≤ `SOFT_MAX_WORDS`; records start from `blank_record()`.

### Phase 5 — `enrich.py` (from `embed.py` enrichment half)
- New `Moment`/`SceneEnrichment` models per §3.4/§5.3; comprehend-before-judge order; drift guard.
- **Checks:** enrich a handful of scenes; every LLM field present; words are in-vocabulary.

### Phase 6 — `derive.py`
- `svos`, `vdi_curve`, `dialogue_ratio`, `arc`, neighbour tones — all mechanical (§5.4).
- **Checks:** words→coords match the tables; `dialogue_ratio` sane vs eyeballed quotes; `arc` matches the
  curve; re-running derive after a `tags.py` table edit changes payloads *without* re-enrichment.

### Phase 7 — `index.py` (delete `embed.py`)
- Build Qdrant (7 vectors — `summary`/`svos`/`descriptors` + `subject`/`verb`/`object`/`setting`, per 0b) +
  SQLite + subject trie + full payload with the soft fields (§5.5), via `vectorstore.py`. Delete `embed.py`.
- **Checks:** `python -m utils.schema --check` **green** (schema wave closes here); one book indexes;
  payload carries `pov/tense/prose_register/dialogue_ratio/vdi_curve`.

### Phase 8 — `search.py` (rewrite read path)
- Hard filters: add `pov`/`tense`, retire `tone`/`intensity`/`arc`. Keep semantic + `channel_vectors`.
- Add stage-3 soft re-rank + `resample`/`curve_dist` (§5.6). Bigger `PREFETCH`.
- **Checks:** hard filters exclude correctly; a tone-curve query reorders the pool sensibly; sliders left
  unset = no effect; vector path still == text path (the seam invariant).

### Phase 9 — `query.py` + harness + `webtest/`
- `query.run` normalizer (§5.7). Update `tests.py` (one door per stage), `evals.py` (per-sharpness +
  soft-facet A/B), `webtest/` (slider UI instead of tone/intensity/arc dropdowns).
- **Checks:** end-to-end: request → `run` → ranked scenes; gold A/B executes.

### Phase 10 — full rebuild, then HyDE
- One clean rebuild on the frozen schema (re-segment → re-enrich → re-derive → re-index).
- Then `train/` per §5.8, feeding the `channel_vectors` seam.

---

## 7. Invariants to preserve (do not break during the move)

- **data.py:** global contiguous `Paragraph.index`; lossless extraction + recall round-trip; `_pack`
  caps; sharded lazy recall.
- **segment.py:** every input paragraph accounted for exactly once (now trivially, via one label each).
- **schema.py:** `scene_schema.json` is the **only** place the field set is defined; every store derives
  from it; never hand-edit the derived lists. `SCHEMA_VERSION` lives only here (already fixed).
- **vectorstore/search:** `EMBED_MODEL` must match the index; `point_id = uuid5(scene_id)`; bge is
  asymmetric (summary/svos **queries** get `QUERY_PREFIX`; indexed passages + descriptor queries stay
  raw); multivector fields are queried with a matrix.
- **The per-field `weight` is retired** — search tunes with `method_weights` + the soft-rank knobs, not
  per-vector weights. (Removes the live/legacy contradiction the audit found.)
- **Rebuilds are explicit** — importing any pipeline module has no build side effects.

---

## 8. Decisions (D1–D5 RESOLVED; ablation still open)

- **D1 — RESOLVED:** tone word → (valence, dominance) on a 2-D grid + a **separate** intensity word → i,
  per moment (§3.4).
- **D2 — RESOLVED:** drop `prev_tone`/`next_tone`; two-scene spanning is a future, **summary-first**
  read-path feature over `next_scene_id` + the `summary` vectors (§3.6).
- **D3 — RESOLVED:** retire the per-field `weight` and the entire `evals.py` weight-tuning stack (§3.7).
- **D4 — RESOLVED:** moment cap = 6 (§5.3).
- **D5 — RESOLVED:** soft-rank defaults `w_tone/w_prose/w_dia = .5/.3/.2`, `wI/wV/wD = .5/.3/.2`, `λ=.5`
  fixed in code (§5.6).
- **Also decided:** keep the redundant `derive` call inside `index` as a safety net (idempotent — cheap
  insurance that a scene is never indexed with an un-derived frame/curve).
- **Ablation (Phase 0b) — RESOLVED 2026-09-07:** committed vector set = **7 named vectors** (`summary`/
  `svos`/`descriptors` + the four `subject/verb/object/setting` facets). The facets earn their place —
  book@1 → 1.000 (bias-free), scene@1 +.19, every sharpness bucket up (numbers in the Phase 0 block). The
  plan's original "collapse to 3" default is overturned by the data.

## 9. Eval plan (gates the risky changes)

- **Segmentation:** spot-check dramatic-unit boundaries vs the old flavor-pure cuts on 1–2 books.
- **Vector set (0b):** per-sharpness scene@1, `use_frame` on/off.
- **Soft facets:** hold semantic fixed, move one slider, confirm the intended reorder; confirm unset
  sliders are inert.
- **Tone curve:** synthetic "rising" vs "falling" queries retrieve the right arc.
- **HyDE:** per-sharpness lift, no regression on specific queries (existing `evals.by_sharpness`).
- Reuse the 100-query gold; harden with more/diverse queries only once a change shows a small-but-real
  signal the current gold can't resolve.

---

## Appendix A — Current behavior inventory (what each file does today → its fate)

The exact behaviors the restructure must **keep**, **change**, **move**, or **drop**. Each migration
phase (§6) diffs its new file against the matching block here — nothing is lost by accident, nothing new
is smuggled in unnoticed. Legend: **[KEEP]** survives ~as-is · **[CHANGE]** behavior changes ·
**[MOVE→x]** same behavior, new home · **[DROP]** removed · **[NEW]** did not exist.

### `data.py` — Stage 1 parse → stays `data.py` [KEEP, light edits]
- `MetadataParser`: load `pg_catalog.csv` → per-book metadata; `_parse_name` "Last, First"→"First Last";
  `feed(code)`→metadata dict; `to_dict`/`from_dict` (Subjects set↔list). **[KEEP]**
- `Book`/`Chunk`/`Paragraph` dataclasses; lossless `to_dict`/`from_dict` (recall round-trip);
  `payload()`/`scene_payload()` lossy views. **[KEEP]**
- `Chunk.scene_payload()` = segmenter input (context + `indexed_paragraphs`). **[KEEP]** (the new
  per-paragraph segmenter reads the same input; only its *output* format changes — §5.2).
- `SceneParser`: `parse_file` (zip→body, encoding sniff), `parse_html` (strip boilerplate/comments),
  `get_segments` (chapters > headings > whole; loose `<p>` swept into "Front Matter" losslessly),
  `_pack` (`TARGET_CHARS`/`MAX_PARAGRAPHS`), `parse_book` (**global** paragraph index,
  `MIN_SEGMENT_CHARS` drop, `OVERLAP` lookback), `parse`. **[KEEP]** (invariants §7).
- `parse_rights` (dc.rights meta), `book_file`, `ensure_book` (lazy per-book recall shard),
  `build_library` (metadata cache + **empty lazy** books). **[KEEP]**
- **Edit:** give `segment` one door for its pre-gate instead of reaching for both `MetadataParser` and
  `parse_rights` (principle #4).

### `process.py` — Stage 2 → **new `segment.py`; DELETE `process.py`**
- `SceneData`/`MultiSceneData` + `output_scenes` TOOL (span emission: start/end/type/content_form/open
  flags/title). **[CHANGE]** → per-paragraph label model (`SCENE_START`/`CONTINUE`/`NOISE`).
- `_expected_indices` + `_validate_coverage` (every index covered exactly once). **[DROP]** — coverage is
  automatic with one label per paragraph; keep only a trivial "all labelled" check.
- `_retry_note` (segmentation wording). **[MOVE→segment.py]** (wording updated; splice stays
  `utils.llm.inject_retry_notes`).
- `SceneBreaker.break_chunk` (forced call + fresh-convo retry, temp climb-then-freeze). **[CHANGE]** same
  loop, new tool + validation.
- `presegmentation_gate` (US-public-domain via dc.rights + non-prose subject) + `_log_exclusion` +
  `EXCLUDED_BOOKS_FILE`. **[KEEP→segment.py]**, folded inside the `segment_book` door.
- `segment_book` (per-chunk parallel, per-chunk `Checkpoint`, flatten in reading order). **[KEEP
  structure→segment.py]**; output reconstructed from labels.
- `scenes_to_records` (stitch open_start↔open_end → `stitch_status`; rebuild `text_html`; `word_count`;
  start from `schema.blank_record()`; DB-agnostic). **[CHANGE→segment.py]** stitch derived from labels;
  **[NEW]** soft word-cap post-process (§5.2).

### `embed.py` — Stage 3 → **split into `enrich.py` + `derive.py` + `index.py`; DELETE `embed.py`**
Enrichment half → **`enrich.py`**:
- `Moment` (sentence-first, then S/V/O/S). **[CHANGE]** add `tone` + `intensity` words per beat (D1).
- `SceneEnrichment` (dominant_tone/intensity/arc/descriptors/summary/moments) + import-time drift guard.
  **[CHANGE]** drop dominant_tone/intensity/arc as LLM fields; add `pov`, `tense`, `prose_register` word;
  richer `summary`; comprehend-before-judge order (§5.3).
- `BatchEnrichment`/`BATCH_TOOL`, `_retry_note`, `_run_tool` (forced call + retry/temp policy), `_plain`
  (strip HTML for LLM), `_batches` (char/count budget), `_enrich_batch` (one call/batch, one item/scene).
  **[KEEP→enrich.py]** (adjust to new fields).
- `_apply` (write enrichment onto record). **[CHANGE]** new field set.
- `enrich_file` (resume/checkpoint, parallel, **denormalize neighbor tones**, rewrite in place).
  **[KEEP structure]**; the neighbour-tone denorm block is **[DROP]** (D2 — `prev_tone`/`next_tone` gone).
Derivation half → **`derive.py`**:
- `_derive_frame`/`derive_frame`/`derive_frame_file`/`derive_frame_scenes` (moments → S/V/O/S facet
  lists). **[CHANGE]** becomes the general derive pass: `svos` (moment sentences), `vdi_curve`
  (tone/intensity words→coords), `dialogue_ratio` (quote ratio), `arc` (curve shape); S/V/O/S facet lists
  only if the ablation keeps those vectors (§3.1).
Indexing half → **`index.py`** (via `utils/vectorstore.py`):
- `_vec_params`/`_ensure_collection` (named-vector config; drop+rebuild if stale). **[MOVE→index.py]**.
- `_ensure_subject_index` (`subject_paths` keyword index). **[KEEP→index.py]**.
- `_multivector_field` (embed svos matrix, summary fallback). **[KEEP→index.py]**.
- `index_records` (SQLite mirror FIRST, then summary/descriptors/svos vectors; payload = full record;
  stamp `subject_paths`; stable `point_id`). **[KEEP→index.py]**; payload gains the soft fields; vector
  set per ablation. **Keep** the idempotent `derive` call it makes before embedding — cheap safety net so
  a scene is never indexed with an un-derived frame/`vdi_curve`, even if the `derive.py` pass was skipped.
- `index_scenes` (rebuild driver; one client/conn). **[KEEP→index.py]**.
- Contract imports from `search` (`COLLECTION`, `VECTOR_NAMES`, `MULTIVECTOR_NAMES`,
  `SUBJECT_PATHS_FIELD`, `embed`, `point_id`, `_as_terms`). **[MOVE→utils/vectorstore.py]** (this is the
  coupling the restructure removes).

### `search.py` — Stage 4 read → **rewrite in place**
- Qdrant contract: `COLLECTION`, `EMBED_MODEL`, `QUERY_PREFIX`, `NAMESPACE`, `point_id`, `_embedder`,
  `embed`, `open_client`, `_search_params`, `SUBJECT_PATHS_FIELD`, `facet_filter`, `book_filter`,
  `subject_filter`. **[MOVE→utils/vectorstore.py]**.
- Flavor: `_unit`/`_check_weights`/`weighted_vector`/`search_weighted_descriptors` (+ anti-descriptors).
  **[KEEP]**.
- Per-field weights: `DEFAULT_FIELD_WEIGHTS`/`active_field_weights`/`_resolve_field_weights` + the
  `TUNED_WEIGHTS_PATH` override. **[DROP]** per D3 (retire per-field weights).
- `_as_terms` **[MOVE→vectorstore]**, `_normalize_pool` **[KEEP]**.
- `_moment_sentences`/`_frame_query_terms`/`_channel_queries` (+ the `channel_vectors` HyDE seam already
  added). **[KEEP]** (frame facets per ablation).
- `score_channels`/`blend_channels`/`search_scenes` (z-norm per channel, weighted blend). **[CHANGE]**
  blend simplifies if weights retired.
- `search_frame` **[KEEP or DROP with the facet vectors]**; `_rrf` **[KEEP]**.
- `tone_filter`/`intensity_filter`/`arc_filter`. **[DROP]** as hard filters.
- `search` (ANDed hard filters + scenes/flavor RRF + `channel_vectors`). **[CHANGE]** swap in
  `pov`/`tense` filters, retire tone/intensity/arc; **[NEW]** stage-3 soft re-rank + `resample` +
  `curve_dist` (§5.6).

### `query.py` — read front door → **extend**
- `to_query_object` (single-beat: whole text → summary + one svos moment). **[CHANGE]** pick text apart
  into beats; strip feeling words → `descriptors`; carry sliders + the tone-word list.
- `normalize`/`run` (one door). **[KEEP entry, extend]**; **[NEW]** HyDE adapter hook via `channel_vectors`.

### `evals.py` — A/B + tuning → **update**
- `load_gold`/`_target`/`_target_scene`. **[KEEP]** (gold entries gain pov/tense/sliders/tone-curve).
- `score_run` (rank-1 book+scene, MRR/Hit@k), `by_sharpness` (the J1 instrument), `compare_runs`,
  `format_comparison`. **[KEEP]**.
- `run_search` (channel-isolation flags), `_gold_frame`/`_gold_moments`. **[CHANGE]** flags for the new
  lanes + a soft-facet A/B path.
- `autolabel_scenes`. **[KEEP]**.
- `collect_vector_channels`/`blend_run`/`coordinate_ascent`/`save_tuned_weights`/`reset_tuned_weights`.
  **[DROP]** with per-field weights (D3); tuning moves to the soft-rank knobs.
- `main` CLI. **[UPDATE]**.

### `tests.py` — build + smoke harness → **update (one door per stage)**
- `FILE_IDS` (10 active). **[KEEP]**. `OTHER_SKIP_RATIO`. **[KEEP]**.
- `TEST_QUERIES`/`COMBINED_QUERIES`/`MOMENTS_QUERIES`/`DESCRIPTOR_QUERIES`. **[CHANGE]** new query shapes.
- `subject_sql_test`/`backfill_subject_paths`/`payload_dump_test`. **[KEEP]**.
- `segment_test` (gate + segment + noise drop + `OTHER_SKIP_RATIO` book gate + records). **[CHANGE]** new
  segmenter; calls `segment.segment_book` (one door).
- `_load_status`/`_mark_status` (status.json skip). **[KEEP]**.
- `embed_test`. **[CHANGE]** → enrich + derive + index (the split).
- `_show`/`search_test`/`manual_search`. **[CHANGE]** new search inputs.
- `stay_awake`/`_pmset_disablesleep` (macOS lid survival). **[KEEP]**.
- `step_one_retrieval` (wget). **[KEEP]**. `step_two_processing`/`step_three_embedding`. **[CHANGE]** to
  the new stage doors. `main`. **[UPDATE]**.

### `main.py` — thin entry (532 B) → **verify/update** to the new stage doors. **[KEEP/UPDATE]**.

### `webtest/server.py` — local read-path UI (port 8765) → **update**
- Qdrant single-process lock eviction (`_lock_holders`/`_evict`/`_open_qdrant`). **[KEEP]**.
- `_load_scenes`/`_build_subject_tree`. **[KEEP]**.
- `_strip`/`_preview`/`_card`/`_pos` (result cards). **[KEEP]** (card may surface new fields).
- `_run_query` (search dispatch + tuning). **[CHANGE]** new search inputs + soft sliders.
- `Handler` routes / `_ctype` / `main`. **[KEEP]**; UI swaps tone/intensity/arc dropdowns for **sliders**.
- imports `utils.tags` for dropdown vocab. **[CHANGE]** → word→coord tables driving sliders.

### `utils/` foundation
- `schema.py` (single source; loader + derived lists + reconcile + `--check`). **[CHANGE]** new field set,
  `float`/`REAL` type + codec, retire `weight` (D3). **[KEEP]** the reconcile machinery + `SCHEMA_VERSION`
  single-source.
- `tags.py` (`Tone`/`Intensity`/`Arc` enums). **[CHANGE]** add `POV`, `Tense`; add word→coord tables
  (tone→VD, intensity→i, prose→register); `Tone` becomes a word→coord lookup, not a hard-filter enum.
- `relational.py` (SQLite mirror: `open_db`/`_migrate`/`sql_upsert`/`get`/`neighbors`/`find`/`count`/
  `tally`, all registry-driven + whitelisted). **[KEEP]** mechanics; columns follow the new schema.
- `subjects.py` (subject-path trie: derive/upsert/browse). **[KEEP]** unchanged.
- `llm.py` (`CLIENT`/`MODEL`/`MODEL_PARAMS`/`classify_llm_error`/`llm_ready_up`/`inject_retry_notes` +
  `PROCESS_PROMPT`/`EMBED_PROMPT`). **[KEEP]** plumbing; the two prompts are rewritten (owner's surface)
  for boundary-classification + word-picking enrichment.
- `checkpoint.py` (resumable per-item cache). **[KEEP]**.
- `log.py` (step/info/done/skip/warn/fail/debug/set_level). **[KEEP]**.
- `storage.py` (`SrcPaths`, 16 paths). **[KEEP]**; add any new dirs (adapter weights).
- `read_write.py` (atomic JSON/text IO). **[KEEP]**.
- `vectorstore.py`. **[NEW]** — the Qdrant contract extracted from `search.py` (§4.1), imported by both
  `index.py` (write) and `search.py` (read).
```
