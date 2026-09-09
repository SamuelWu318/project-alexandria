# PLAN.md — Project Alexandria restructure

**Branch:** `restructure` · **Execution manual for the REMAINING work** (Phases 8.5 + 10). Phases 0–9 are
DONE — the built code + `CLAUDE.md` invariants + the memory files are the reference for those; this file now
drives only order-aware `svos` (8.5) and the full rebuild → HyDE (10).

> ## ▶ START HERE (new session)
> 1. Binding house style = `CLAUDE.md` → "Code principles" (readability; comment framework; arrows-down;
>    one-import-one-method; single-responsibility). It governs every edit.
> 2. **Current phase = first §4 checklist entry not ✅ → now Phase 8.5 (order-aware `svos`), then Phase 10.**
> 3. Read that phase's spec — **8.5 = §3.2 (full mechanism)**, **10 = §3.1 + the module map in §1**. Then:
>    author to the house style, run the phase's **Checks**, meet **Done-criteria**, update the affected
>    `CLAUDE.md` lines, flip ✅ in §4, commit. **One phase per session** unless told otherwise.
> 4. Running modules on this machine + the old-store caveat: memory `reference-restructure-run-verify`.

**STATUS:** design frozen · D1–D5 + the 0b ablation resolved (§6). **Phases 0–9 DONE** (§2): schema v4 + tags
+ `vectorstore.py`; `data.gate_facts`; `segment.py` sparse labelling (`process.py` deleted); `enrich.py` /
`derive.py` / `index.py` (Stage-3 split, `embed.py` deleted); `search.py` 4-stage read path (hard `pov`/`tense`
→ semantic → soft re-rank, weight-free `combine="sum"`); `query.py` single-beat + soft word→coord + the
split-door harness + the webtest slider/pov-tense UI. `import query/evals/tests/search` GREEN
(`import webtest.server` still fails on the OLD on-disk `scenes.db` — `no such column: pov` — until the
Phase 10 rebuild). **▶ NEXT: Phase 8.5 (order-aware `svos`, §3.2); then Phase 10 (rebuild → HyDE, §3.1).**

---

## 1. Design + data model (frozen — the built reference)

**House style (binding, `CLAUDE.md` → "Code principles"):** readability first; the comment framework;
arrows-down (helpers above callers, entries last); **one import = one method** (push composition to the
owner; foundation/contract modules exempt); single responsibility per file.

**North star (grounds the lanes):** a writer wants a scene by FEEL and by WHAT HAPPENS. Example — "an expert
hunter analytically waits in hiding, before shooting his bow and barely missing" · 3rd/past · low dialogue ·
"low intensity rising to the bow shot" · "a logical feel." A request splits into **three lanes**: SEMANTIC
(free text → summary + svos beats + descriptors, vectors), HARD facets (book/pov/tense — exclude, never
softened), SOFT facets (prose register, dialogue level, tone CURVE — tilt, never exclude). The unit returned
is a **scene** (a dramatic unit with an arc); beat matching happens INSIDE it via the moment multivector.

**Schema (v4, frozen — `utils/schema/scene_schema.json` is the single source).** ONE Qdrant collection
`scenes`, **7 named vectors**: `summary` + `descriptors` (single, LLM) and `svos` + `subject`/`verb`/`object`/
`setting` (multivector, MAX-SIM, derived from `moments[]`). `svos` = the moment sentences (**order-aware after
8.5, §3.2**); the four facets stay order-free (dedup'd sets). Hard facets `book_id`/`pov`/`tense` (payload
filter). Soft facets `prose_register` (float), `dialogue_ratio` (float), `vdi_curve` (list of `[v,d,i]`) — the
affective arc, matched by curve distance. **Moments carry the arc:** each `moment = {sentence, subject, verb,
object, setting, tone, intensity}`, `tone`/`intensity` are LLM-picked WORDS → `(v,d,i)` via `tags.py` →
`vdi_curve`. **Store the word, derive the number** (retuning a `tags.py` coord = a `derive` payload refresh,
never a re-enrich). SQLite mirror (`relational.py`) beside the vectors, joined on `scene_id`, holds EVERY
record for exact-match / COUNT / navigation.

**Read path** = one `search.search()` (4 stages: hard filter → semantic pool → soft re-rank → slice), fed by
the `query.run` front door. Semantic blend is WEIGHT-FREE, default `combine="sum"` (the 0b gold demoted
`max`, §6). `query.py` is SINGLE-BEAT (whole summary = one scene target; NO qsplit / facet extraction — both
built, tried, DELETED, do not resurrect) and owns the soft **word→coord** (tone/intensity word-curve → numeric
`tones`; prose word → `prose`; `dialogue` + pov/tense passthrough) — `search` stays pure-numeric (no `tags`
import). Multi-beat is a separate, caller-built future feature.

**Module map + one door each** (arrows = "imports one door from"):
```
data     Stage 1 parse + recall + gate_facts        data ──> (utils)
segment  Stage 2 boundary-classify -> records       segment ──> data.gate_facts , utils.llm/schema
enrich   Stage 3a LLM enrich (enrich_file)          enrich  ──> utils.llm/schema
derive   Stage 3b word->number (derive_records/_file) derive ──> utils.tags , utils.schema
index    Stage 3c build stores (index_scenes/_records) index ──> vectorstore , relational/subjects
search   Stage 4 read (search)                      search  ──> vectorstore , utils.schema
query    read front door (run) + HyDE hook          query   ──> search.search , utils.tags
train/   HyDE adapter (Phase 10)                     train   ──> search channel_vectors seam
tests / evals / webtest ──> the stage doors + search.search + query.run (one door each)
```
No feature file imports another feature file's internals. `utils/`: `storage` (paths = `SrcPaths`, everything
under `logs/test/`), `read_write`, `checkpoint`, `log`, `llm` (client/model/prompts — owner's surface),
`schema` (single source of truth), `tags` (enums + word→coord), `vectorstore` (Qdrant contract — read+write
share it), `relational`, `subjects`.

---

## 2. What's built (Phases 0–9) — the ✅ log

Detail for any done phase = the code + `CLAUDE.md` invariants + memory `project-restructure-plan`.

- **0** scaffolding + 0b ablation (KEEP the 4 facet vectors → 7-vec set, §6) + D1–D5 resolved (§6).
- **1** schema v4 + `tags` (7 vec; pov/tense hard; prose_register/dialogue_ratio/vdi_curve soft; `weight` retired, D3).
- **2** `utils/vectorstore.py` — Qdrant contract extracted (killed the embed→search coupling).
- **3** `data.gate_facts` one-door pre-gate (public-domain + subject).
- **4** `segment.py` sparse boundary labelling (book-parallel, sequential chunks + continue-flag); `process.py` deleted.
- **5** `enrich.py` comprehend-before-judge (LLM fields only; moment cap 6; per-beat tone/intensity words).
- **6** `derive.py` mechanical word→number (svos/facets/vdi_curve/prose_register/dialogue_ratio/arc; idempotent).
- **7** `index.py` builds 7-vec Qdrant + SQLite + subject trie; `embed.py` deleted; schema wave CLOSED.
- **8** `search.py` 4-stage read path; per-field weight stack deleted (D3); soft re-rank; **`combine="sum"` default** (`max` regressed the 0b gold, book@1 .86 vs .99).
- **9** `query.py` soft word→coord + split-door harness (`enrich_file`→`derive_file`→`index_records`) + webtest sliders/tone-curve/pov-tense. Verified synthetic + 0b gold unchanged; **live webtest UI deferred to the Phase 10 rebuild** (old-schema on-disk stores).

---

## 3. Remaining specs

### 3.1 HyDE — learned adapter (`train/`, Phase 10)

`train/`: `build_dataset.py` · `fit_adapter.py` (v0 numpy ridge) · `query_adapter.py` · `train_adapter.py`
(v1 torch, gated). Generation-free `g(beat) → channel query vectors`, self-labelled from the corpus (each
scene's `summary` = complex variant, `moments[].sentence` = medium, subject+verb+object = sparse, all
targeting that scene's stored `summary` vector; same-book scenes = hard negatives); v0 = numpy ridge (no
deps, closed-form), v1 = torch InfoNCE (gated on v0 lift). Feeds `search` via the `channel_vectors` seam
(already built — `search`/`search_scenes`/`score_channels`/`_channel_queries` take `channel_vectors={name:
vec}`, a supplied vector overrides the text-derived one, default None = unchanged). **Do HyDE AFTER the schema
+ rebuild are frozen**, so the adapter learns the final manifold. Adds an adapter-weights dir under
`storage.py`. Full plan: memory `project-query-normalizer`.

### 3.2 Order-aware `svos` — positional beat encoding (Phase 8.5, decided 2026-09-09)

**Goal:** make event ORDER matter in the `svos` multivector itself (recall time), not only in the affect
`vdi_curve`. MAX-SIM is permutation-invariant, so order cannot come from the matching — it must be baked
into each beat vector. Chosen mechanism = **concatenate a normalized-position component** onto every `svos`
beat vector, so a beat's cosine becomes a tunable blend of semantic + positional agreement. `subject`/`verb`/
`object`/`setting` are UNCHANGED (order-free sets); this applies to `svos` ONLY.

**Mechanism (index + query must share it):** for a beat at position `k` in a `K`-beat sequence, normalized
`u = k/(K-1)` (`u=0` when `K=1`; symmetric with the `resample` convention in `search.py`). Build the
stored/queried beat vector as
```
v = [ sqrt(1-β) · unit(sem_embed) ;  sqrt(β) · p(u) ]        # β in [0,1);  ||v|| = 1
p(u) = [cos(π/2 · u), sin(π/2 · u)]                          # unit; p(u1)·p(u2) = cos(π/2·|u1-u2|) in [0,1]
```
Then `cos(v_q, v_s) = (1-β)·sem_cos + β·pos_cos`, `pos_cos = 1` at equal normalized position, falling to `0`
at opposite ends (non-negative — position never flips a sign, only withholds reward). MAX-SIM over the
augmented beats therefore prefers matches that are BOTH semantically close AND at the same relative position,
i.e. in-order alignment — **softly**: a strong out-of-order semantic match can still win if its `sem_cos`
clears the `β` gap. So **`β` is the order-strength ⟷ recall-robustness knob** (`β=0` reproduces today's
order-free behavior). Query beats build `p(u)` over the QUERY's own length; scene beats over the SCENE's — so
a 3-beat query's middle beat aligns to a 6-beat scene's middle beat (normalized position, not raw index).

**Variable-length queries — do NOT resample/reshape the `svos` matrix.** `u = k/(K-1)` already makes any
two beat counts comparable (both live on `[0,1]`), and MAX-SIM's many-to-one collapse then handles a
near-but-unequal query *gracefully by construction*: a 4-beat `forward, right, right, down` against a 3-beat
`forward, right, down` lets both `right`s grab the one scene `right`, and the ends pin at `u=0`/`u=1`. Length
is absorbed by the encoding, never by forcing the query to the scene's beat count. **Never interpolate /
"connect the dots"** to a common length: only `p(u)` is smooth — the semantic half is not, so an averaged
`unit(sem_embed)` between two beats is a fabricated beat in a non-linear space (you'd retrieve against events
that never existed). Resampling to a fixed length is the tone-curve lane's move ONLY (it compares two curves
point-by-point at a fixed alignment); the `svos` lane deliberately does the opposite — MAX-SIM + relative
`u`, no length coercion. Guard `K=1` (`u=0`, avoid `0/0`).

**What changes (all small, but it is a vector-format change ⇒ a re-index):**
- `utils/vectorstore.py` (contract): `svos` vector size `384 → 384+2`; add the positional-scheme constants
  (`SVOS_POS_BETA`, the `p(u)` frequency, dims) + a `svos_beat_vectors(sentences) → matrix` helper that both
  sides call, stamped like `EMBED_MODEL` (index/query MUST agree or scores are junk). The other named vectors
  keep size 384.
- `index.py`: build the `svos` matrix through the new helper (positional dims appended per moment, in
  `moments[]` order). `_ensure_collection` already rebuilds on a vector-size/dims drift — so a re-index
  re-lays `svos`. Facet vectors untouched.
- `search.py` `_channel_queries`: build the query `svos` matrix through the SAME helper (append `p(u)` over
  the query beats) so query and index live in the same space. No other search logic changes; MAX-SIM stays.
- `query.py`: already supplies ordered `moments`; nothing extra (order is intrinsic to the moment list).
- **Tune `β` on the gold** (and the `p(u)` frequency): sweep `β ∈ {0, .1, .2, .3}`, confirm an in-order query
  out-ranks the same beats shuffled, and that book@1/scene@1 do NOT regress vs `β=0`. Ship the smallest `β`
  that gives a real order signal without a recall drop. `β=0` is always the safe fallback.

**Sequencing:** a `svos` **format** change, so it must land WITH a re-index. Do the CODE (vectorstore + index
+ search) as its own step, verify on a SMALL new-schema index (hand-built scenes, per Phase 8/9), then the ONE
full **Phase 10 rebuild** bakes the positional `svos` corpus-wide. Facet order-independence + the
`channel_vectors` seam are both preserved.

---

## 4. Remaining work (checklist + detail)

**Progress:** Phases **0–9 ✅** (see §2). Remaining, in order:

- ☐ **8.5 order-aware `svos`** ← **NEXT**. Positional beat encoding (§3.2). `svos`-format change; code in
  `vectorstore` + `index` + `search`, verify on a small new-schema index, `β` tuned on the gold; the corpus
  re-lay rides the Phase 10 rebuild. May fold into Phase 10 (either way the corpus is embedded ONCE, with the
  positional `svos` in place).
  - **Checks:** an in-order query out-ranks the same beats shuffled (impossible today); `β=0` reproduces the
    current ranking bit-for-bit; book@1/scene@1 do not regress at the chosen `β`; a query whose beat count
    differs from the target scene's still matches (variable length absorbed by `u`, NOT by resampling — §3.2).
  - **Done-criteria:** checks green; update `CLAUDE.md` (the search invariant: `svos` is order-aware via the
    positional encoding + the `β` knob); flip this item ✅.
- ☐ **10 full rebuild → HyDE**.
  - One clean rebuild on the frozen schema (re-segment → re-enrich → re-derive → re-index), with the 8.5
    positional `svos` in place. Clears the old-schema on-disk stores → unblocks `import webtest.server`, the
    live webtest UI (sliders/pov-tense/tone-curve), and the evals soft-facet A/B (`--mode soft`).
  - Then `train/` per §3.1, feeding the `channel_vectors` seam.

Each phase: author to the house style, run **Checks**, meet **Done-criteria**, update `CLAUDE.md`'s affected
lines (pipeline diagram, invariants, ownership, run-reference) so the always-loaded map never lies, flip ✅
here, commit.

---

## 5. Invariants to preserve

- **data.py:** global contiguous `Paragraph.index`; lossless extraction + recall round-trip; `_pack` caps;
  sharded lazy recall.
- **segment.py:** every input paragraph accounted for (unlabelled = continuation of the open scene).
- **schema.py:** `scene_schema.json` is the **only** place the field set is defined; every store derives from
  it; never hand-edit the derived lists. `SCHEMA_VERSION` lives only here.
- **vectorstore/search:** `EMBED_MODEL` must match the index; `point_id = uuid5(scene_id)`; bge is asymmetric
  (summary/svos **queries** get `QUERY_PREFIX`; indexed passages + descriptor queries stay raw); multivector
  fields are queried with a matrix. (8.5: `svos` index + query MUST share `svos_beat_vectors`.)
- **query/search split:** the soft **word→coord** lives in `query.py` (`tags`); `search` stays pure-numeric.
- **Per-field `weight` is retired** — search tunes with `method_weights` + the soft-rank knobs.
- **Rebuilds are explicit** — importing any pipeline module has no build side effects.

## 6. Decisions (D1–D5 + ablation, resolved)

- **D1:** tone word → (valence, dominance) + a **separate** intensity word → i, per moment.
- **D2:** drop `prev_tone`/`next_tone`; two-scene spanning is a future summary-first read-path feature over
  `next_scene_id` + the `summary` vectors.
- **D3:** retire the per-field `weight` and the entire `evals.py` weight-tuning stack.
- **D4:** moment cap = 6.
- **D5:** soft-rank defaults `w_tone/w_prose/w_dia = .5/.3/.2`, `wI/wV/wD = .5/.3/.2`, `λ=.5`, fixed in code.
- **Also:** keep the redundant idempotent `derive` call inside `index` (cheap insurance).
- **Blend (Phase 8, gold-driven):** default `combine="sum"` (weight-free) — `max` REGRESSED the 0b gold
  (book@1 .86 / scene@1 .53 vs sum .99 / .77, h2h sum 32 / max 4 / tie 64); `max` kept as the A/B alt only.
- **Ablation (0b, resolved 2026-09-07): committed set = 7 named vectors.** Adding the 4 facet vectors lifted
  **book@1 to 1.000** (from .920, bias-free human labels), **scene@1 +.19** (.66→.85), scene_mrr +.156, and
  improved *every* one of the 5 sharpness buckets. Head-to-head on top1: ON wins 26, OFF 3, 71 ties. book@1
  uses human ground truth, so "keep" doesn't depend on the scene-label provenance.

## 7. Eval plan

- **Soft facets:** hold semantic fixed, move one slider (`evals --mode soft`), confirm the intended reorder;
  confirm unset sliders are inert.
- **Tone curve:** synthetic "rising" vs "falling" queries retrieve the right arc.
- **Order-aware `svos` (8.5):** in-order query out-ranks the shuffled beats; `β=0` reproduces today; no
  book@1/scene@1 regression at the chosen `β`.
- **HyDE (10):** per-sharpness lift (`evals.by_sharpness`), no regression on specific queries.
- Reuse the 100-query gold (`webtest/gold/test_queries.json`); harden with more queries only once a change
  shows a small-but-real signal the current gold can't resolve. (`autolabel_scenes` stamps each query's target
  scene id + its pov/tense/prose_register/dialogue_ratio/vdi_curve once a new-schema index exists.)

---

**▶ New session? Go to the START HERE box at the top.** Everything before Phase 8.5 is BUILT — read the code +
`CLAUDE.md` + the memory files for it, don't re-derive it. **Current phase = 8.5 (order-aware `svos`, §3.2),
then 10 (rebuild → HyDE, §3.1).**
