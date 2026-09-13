# PLAN.md — Project Alexandria restructure — Phase 10 (done) + Phase 11 (eval realignment)

**Branch:** `restructure`. Phases **0–8.5 are DONE** — their reference is the built code + the `CLAUDE.md`
invariants + memory `project-restructure-plan`; this file no longer re-describes them. **Phase 10 is now
essentially done** (§2: 10.1 rebuild ✅, 10.2 verify ✅, 10.3 β=0.2 ✅, 10.4 HyDE v0 built + measured → **no
lift on the needle gold → parked**). The live driver is now **Phase 11 (§6): eval realignment** — the
product goal is a **top-5 of content-similar scenes**, NOT "retrieve the one iconic scene," so we build the
sets that measure THAT before trusting any retrieval change (HyDE included).

> ## ▶ START HERE (new session)
> 1. **House style = `CLAUDE.md` → "Code principles"** (readability; comment framework; arrows-down;
>    one-import-one-method; single-responsibility). Binding on every edit.
> 2. **Phase 10 (§2) is done** — skim it for context. **The work now is Phase 11 (§6):** build the rough
>    query set + the LLM-selected top-5 similarity gold, then A/B needle-vs-similarity for the best top-5.
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

## 6. Phase 11 — eval realignment (needle → similarity → scored top-5)  *(ACTIVE)*

**Why.** A writer wants scenes *like* the one they describe — matched on action / plot / beats / feel, to
reuse — NOT one specific book scene. So retrieval quality means **the top-5 are all strong CONTENT matches,
best-first**. Every metric so far (scene@1 / book@1 vs one labelled iconic scene) grades *needle-finding*,
which is blind to top-5 similarity — that is why HyDE 10.4 "failing" the needle gold says nothing about the
real goal. Build sets that measure the goal, then re-judge every retrieval change against them.

**Three eval directions** — score the SAME methods under each; the payoff is learning whether the needle
lens and the similarity lens AGREE or DIVERGE:
1. **Needle** — existing `webtest/gold/test_queries.json` (one `target_scene_id`) → scene@1 / book@1 / MRR.
   Keep as an exact-retrieval capability check.
2. **Rough / similarity (qualitative)** — realistic *loose* writer queries; read the top-5 side-by-side
   (baseline vs a method) + cheap proxies (subject/verb/object + tone-arc overlap). No fixed answer.
3. **Scored top-5 (benchmark)** — the SAME rough queries + an LLM-selected top-5 relevance set → **P@5 /
   nDCG@5**. Reproducible; the durable benchmark for the product goal.

**Rough query set** (`webtest/gold/rough_queries.json`; hand-authored, easy to extend):
- **10** GENERIC, summary-only, ONE sentence · **10** SPECIFIC (no scene in particular), ONE sentence ·
  **10** SPECIFIC, TWO–THREE sentences  → **30 base configs (semantic only)**.
- **THEN repeat the same 30 WITH sliders tuned on intuition** (e.g. a fight → higher tension + a
  rage/vengeance tone; a deathbed → falling arc, low dialogue) → **30 slidered configs**. 60 total = an A/B
  of "semantic only vs + intuition sliders."
- **Slider numbers come DIRECTLY from the tag tables:** store the tone/intensity/prose WORDS
  (`tags.Tone` / `tags.Intensity` / `tags.ProseRegister`) and convert through `query.tone_curve` /
  `query.prose_level` (the app's word→coord path: `tags.moment_vdi` / `tags.prose_coord`). NEVER hand-type
  coordinates — the tag tables stay the one source of truth.

**Scored top-5 gold builder** — a reproducible py **CLASS** + a NEW **system prompt** (regenerate whenever we
add queries):
- **Step 1 — pool:** for each query, run search over a few configs, dedupe, and store the **top-20 candidate
  scenes** per query to a JSON (`scene_id` + `summary` + facets — enough for the model to judge).
- **Step 2 — select:** chunk-feed the query + its 20 candidates to the LLM under a select/judge system
  prompt; it returns the **top-5** most content-similar, ranked. Freeze `{query → [5 scene_ids]}` as the gold.
- The class owns pool → chunked-LLM-select → write-back (one call makes more gold). The **system prompt is
  the USER's tuning surface** (like the pipeline prompts). Uses `utils.llm` (CLIENT / MODEL / retry policy).
- Score any method with **P@5 / nDCG@5** vs the frozen gold.
- Caveat (pooling): a great scene no config surfaces never enters the pool → counts non-relevant; pool
  diverse configs deep to mitigate.

**Then:** A/B baseline vs HyDE-adapter (and `combine` / slider variants) under BOTH the needle gold and the
top-5 gold. If similarity ≠ needle, HyDE / retuning re-opens against the top-5 objective — the adapter's
"pull toward nearby lookalikes" that hurt needle-finding may HELP a top-5 of similar scenes.

**Build order:** (a) rough query set → (b) pool-top-20 JSON → (c) the select class + system prompt → the
top-5 gold → (d) P@5/nDCG@5 scorer in `evals` → (e) the cross-lens A/B. Then update `CLAUDE.md` + flip ✅.
