# PLAN.md — Project Alexandria restructure — Phase 10 (done) + Phase 11 (eval realignment)

**Branch:** `restructure`. Phases **0–8.5 are DONE** — their reference is the built code + the `CLAUDE.md`
invariants + memory `project-restructure-plan`; this file no longer re-describes them. **Phase 10 is now
essentially done** (§2: 10.1 rebuild ✅, 10.2 verify ✅, 10.3 β=0.2 ✅, 10.4 HyDE v0 built + measured → **no
lift on the needle gold → parked**). The live driver is now **Phase 11 (§6): eval realignment** — the
product goal is a **top-5 of content-similar scenes**, NOT "retrieve the one iconic scene." The first
attempt (a pooled, LLM-selected top-5 gold) was **abandoned as circular** (§6). Since built: a **non-circular
writer-realistic gold** (`webtest/gold/writer_queries.json`, scored scene-level via `query.run`), the gate
proof that **facets carry the moments-channel win** (summary .42 → +facets .91), the **blob-svos removal**, and
the intensity **arc/vd** query input. **NEXT: Bridge 2** — a learned summary→facet map to close the writer-real
.31 → oracle .91 gap.

> ## ▶ START HERE (new session)
> 1. **House style = `CLAUDE.md` → "Code principles"** (readability; comment framework; arrows-down;
>    one-import-one-method; single-responsibility). Binding on every edit.
> 2. **Phase 10 (§2) is done** — skim it for context. **The work now is Phase 11 (§6):** the pooled
>    top-5-gold approach was abandoned as circular; design a non-circular, scene-level similarity eval.
> 3. Author to the house style, run the step's **Checks**, meet **Done**, update the affected `CLAUDE.md`
>    lines (diagram / invariants / ownership / run-ref) so the always-loaded map never lies, flip ✅, commit.
> 4. Running modules on this machine (PYTHONPATH + venv): memory `reference-restructure-run-verify`.

**Status:** design frozen; D1–D5 + the 0b ablation resolved (§5). `import query/evals/tests/search/
webtest.server/train` GREEN. New-schema stores built (1647 scenes, `svos` 386-d). On the needle gold:
`combine="sum"` book@1 .97 / scene@1 .91 (no regression, `max` still loses); soft sliders bite + inert when
unset; tone curve retrieves the right arc. **HyDE v0 = no lift on that gold** — but the gold is the WRONG
yardstick (§6), so v0 is parked (dormant `train/`, not wired into `query.run`), v1 torch NOT built.

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

### ✅ 10.1 Full rebuild  *(the gate — DONE: 1647 scenes re-segmented + enriched + indexed, new v4 schema, `svos` 386-d)*
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

### ✅ 10.2 Verify on the new index  *(DONE via evals: no book regression, soft sliders bite + inert-when-unset, tone curve directional; webtest BROWSER smoke still optional)*
- **webtest live UI** (the Phase-9 deferral): prose/dialogue **sliders** + a 3-beat **tone-curve** + pov/tense
  dropdowns route through `query.py`. `python -m webtest.server` → http://localhost:8765/.
- **`evals --mode soft --axis prose|dialogue|tones`**: hold semantic fixed, move one slider, confirm the
  intended reorder; confirm unset sliders are inert.
- **Tone curve:** synthetic "rising" vs "falling" queries retrieve the right arc.
- **`autolabel_scenes`:** stamp each gold query's target `scene_id` + `pov`/`tense`/`prose_register`/
  `dialogue_ratio`/`vdi_curve`.

### ✅ 10.3 β — ship **β=0.2** (order mechanism confirmed; full re-index sweep low-value)
Mechanism verified on a real 5-beat scene (numpy MAX-SIM, no store touched): in-order vs reversed margin
scales ~linearly with β — 0.0→0.000 (order-free), 0.1→+0.26, 0.2→+0.52, 0.3→+0.69. The index is already
built at **β=0.2** and its gold is excellent (book@1 .97 / scene@1 .91) → no regression concern. A full
per-β **re-index** sweep is low-value: the needle gold is single-beat summaries, so order-aware `svos`
barely engages — the gold reads ~flat across β. Ship 0.2 (no re-index). **Knob:** `SVOS_POS_BETA` in
`utils/vectorstore.py` (β baked into stored `svos` → a β change needs an `svos` re-index).

### ⚠ 10.4 HyDE — learned adapter (`train/`) — v0 BUILT + measured → **NO needle lift → PARKED**
Generation-free query→summary-vector map, self-labelled from the corpus, feeding the `channel_vectors` seam.
**Built (v0):** `train/build_dataset.py` (self-label: `summary`=COMPLEX, `moments[].sentence`=MEDIUM,
`subject+verb+object`=SPARSE; all target that scene's stored `summary` vector; book_id rides each row for
v1 negatives) → `train/fit_adapter.py` (closed-form ridge, shrinks toward the IDENTITY prior so high λ = the
baseline) → `train/query_adapter.py` (`adapt()` runtime door + a gold A/B `main()`). `storage.ADAPTER_DIR`
added; `evals.run_search` gained an `adapter=` hook. `train_adapter.py` (v1 torch InfoNCE) is **NOT built**
— it is gated on a measured v0 lift.
- **Result:** on the needle gold, NO λ beats baseline (best case *ties* scene@1 .91 while nicking book@1
  .96 and sharp-5 .85). The behavior is textbook ridge (high λ → baseline, low λ → regress), so it is the
  METHOD having no headroom, not a bug: ridge lifts *average* query→target cosine (+0.08) but that is
  CLOSENESS, not ranking margin — it also pulls the query toward same-book lookalikes.
- **Parked, not deleted:** `train/` stays dormant (nothing wired into `query.run`; reachable only via
  `train.adapt()`). v1 is not built.
- **Why this is not the last word (→ Phase 11):** the needle gold is polished, near-document queries —
  exactly where HyDE has no gap to close. HyDE's payoff is *terse, loose* queries, which that gold lacks,
  and its "pull toward nearby lookalikes" behaviour is a DEFECT for needle-finding but plausibly a VIRTUE
  for "return 5 similar scenes." So v0 is untested against the real goal; Phase 11 builds that test.

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

---

## 6. Phase 11 — eval realignment (scene-level needle → non-circular similarity)  *(ACTIVE)*

**Why.** A writer wants scenes *like* the one they describe — matched on action / plot / beats / feel, to
reuse — NOT one specific book scene. So retrieval quality means **the top-5 are all strong CONTENT matches,
best-first**. `scene@1` / `book@1` vs one labelled iconic scene grades *needle-finding*, blind to top-5
similarity — which is why HyDE 10.4 "failing" the needle gold says nothing about the real goal.

**ABANDONED — the pooled, LLM-selected top-5 gold (removed 2026-09-15; do NOT rebuild).** The first attempt
froze a relevance gold: per rough query, pool the top-20 candidates over a few configs → LLM selects the 5
best → freeze → score P@5/nDCG@5. It is **circular**:
- The gold can only contain scenes the retriever ALREADY surfaced (**gold ⊂ the retriever's own top-20
  pool**), so it grades rank-agreement *inside the retriever's funnel*: blind to a recall miss, and it
  **punishes** a retrieval change that finds a genuinely better scene the original pool lacked (counts as
  non-relevant). Pooling near-clone configs of ONE retriever makes the bias severe (unlike TREC's diverse
  systems). It also graded `book_id` match, which the product goal doesn't care about.
- Diagnostic that settled it (keep for the record): on the 30 rough queries one `sum` config already
  recovered the gold at **recall@20 ≈ .91 / recall@50 ≈ .97** but only **recall@5 ≈ .38**. Retrieval finds
  the right neighborhood; the real gap is **top-5 ORDERING on loose queries**. The P@5 (~.33) was measuring
  retriever-vs-judge ordering agreement, not similarity — and sliders / `combine=max` both slightly hurt it.
- Removed with it: `similarity_gold.py`, `webtest/gold/{pool_candidates,pool_queries,rough_queries}.json`,
  `SELECT_PROMPT` (+ its `utils.__init__` export), and everything behind `evals --mode similarity`
  (`load_rough` / `load_similarity_gold` / `_rel` / `_ndcg` / `score_similarity` / `format_similarity` /
  `run_rough` / `_similarity_cli` + the `--rough`/`--sim-gold`/`similarity` CLI args). `evals` modes today:
  `norm|lift|flavor|combine|soft`.

**KEPT — the needle gold as a capability floor, re-headlined to SCENE level.** `webtest/gold/test_queries.json`
stays the exact-retrieval check. Move the emphasis off book: `book@1` is trivially easy (.97 — books have
distinct vocab/style) and says nothing about scene similarity, so headline **scene@1 / scene-MRR / scene
Hit@5** and demote book-match to a diagnostic. (`evals.score_run` still reports a book+scene composite as its
headline `top1` — the scene-level re-headline is a small, **unbuilt** code change.)

**BUILT (this session, 2026-09-19) — the writer-realistic gold, the gate proof, and the blob-svos fix.**
OPEN #1's *held-out paraphrase recall* is now built: `webtest/gold/writer_queries.json` (from
`webtest/gold/build_writer_gold.py`) = 55 source scenes × 3 loose variants — **similar** (detailed) /
**ambiguous** (general language, same plot) / **sparse** (general language, 1-2 beats) — carrying ONLY writer
inputs (summary, descriptors, pov, tense, and an intensity **arc** on similar/ambiguous). NON-CIRCULAR: target
= the source `scene_id`, fixed before retrieval. Consumed via **`query.run`** (the real front door), scored
scene-level by `evals.score_run`.
- **Gate proof — the moments channel is worth synthesizing.** 3-way ablation on the OLD facet-populated gold
  (scene@1, descriptors off): summary-only **.42** → +svos **.50** → +svos+facets **.91**. The FACETS carry the
  win (+.41); svos adds almost nothing (+.08). So Bridge 2 targets subject/verb/object/setting, NOT svos.
- **blob-svos REMOVED.** The front door used to fire the whole summary as one svos "blob" beat; it scored
  *below* summary-only (writer-gold scene@1 .31 vs .38; old-gold .36 vs .42) — one blob matches no single
  indexed per-beat clause. Deleted from `query.py`; the query now fires summary-only semantic (no `moments`).
- **Writer-realistic baseline** (post-blob-removal, arc off, scene@1): sparse **.545** / similar **.382** /
  ambiguous **.309**. NOT monotonic by length — **vagueness, not brevity, is the hard axis**; `ambiguous`
  (general words + full plot → diffuse embedding) is the loose-query stress tier. This **.31–.55** sits far
  below the facet ORACLE **.91** = the Bridge 2 prize. (Arc's scene@1 effect here is ±1 query = noise — it is
  a soft-rerank tilt, not a top-1 lever.)
- **Intensity-only arc + VD query input** (`query.py` + `search.py`): the writer picks 1 of 6 `INTENSITY_ARCS`
  (the i-axis) + an OPTIONAL ordered VD tone-word list (len 1/2/3); `arc_curve(arc, vd)` builds a fixed 3-point
  curve leaving unspecified axes `None`, and `curve_dist`/`resample` skip `None` axes. `query.run(..., arc=, vd=)`.

**NEXT — Bridge 2 (approved; Bridges 1 sentence-split + 3 PRF were rejected: too primitive / drift).** Learn an
offline linear map (ridge, cross-channel, from the 1647 aligned scenes) summary → subject/verb/object/setting
query vectors; feed via the `channel_vectors` seam; A/B against the writer gold's `ambiguous` tier (goal: close
the .31→.91 gap). Skip svos synthesis.

**OPEN — resolve before any fine-tune:**
1. **A non-circular scene-similarity eval** — the target must be fixed *before* retrieval (so it can reward
   finding scenes the current retriever misses). ✅ *Held-out paraphrase recall* is BUILT (the writer gold
   above). Still-open alternatives to layer on:
   - *Synthetic "twin" recall* — LLM rewrites a scene (new setting/names, same action + beats + feel); query
     with the twin, target = the original. Measures action/feel similarity invariant to surface words.
   - *Eval-time LLM graded relevance* — judge each (query, retrieved-scene) pair **fresh** at eval time (0–3),
     average over the top-5. The honest version of the abandoned idea: never freeze a pool, so a
     newly-surfaced good scene is rewarded. Costs LLM calls per run → keep the query set small.
   - *Cheap proxies* for CI regression: subject/verb/object overlap + tone-arc distance, query-target vs top-5.
2. **Corpus size.** 10 books / 1647 scenes is small — for many queries a true "5 similar scenes" may not even
   exist, capping any similarity metric. Growing the corpus (a) makes similar-5 real (density) and (b) lets us
   **hold out whole books** for a generalization eval → do it *before* serious fine-tuning.
3. **Which fine-tune.** HyDE v0 (parked, §2 10.4) hurt needle by pulling toward same-book lookalikes; that may
   HELP top-5 similarity — but only a non-circular similarity eval can tell. Re-open it (and `combine` / slider
   variants, an embedding fine-tune, or a re-ranker over the top-20) only once #1 exists.

**Build order:** (a) ✅ scene-level scoring on a non-circular gold — via `evals.score_run` on the writer gold →
(b) ✅ one non-circular similarity eval — the writer gold (held-out paraphrase recall) → (c) **NEXT: build
Bridge 2** (summary→facet map) + A/B it against the writer gold's `ambiguous` tier → (d) optionally expand the
corpus + hold out books, then A/B other fine-tune candidates. Update `CLAUDE.md` + flip ✅ as each lands.
