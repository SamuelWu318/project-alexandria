import json, re, time, math, threading
from collections import Counter
from pathlib import Path, PurePath
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, ValidationError, Field, field_validator
from openai import pydantic_function_tool

from utils import (CLIENT, MODEL, MODEL_PARAMS, WORKERS, EMBED_PROMPT, Checkpoint, SrcPaths,
                   POV, Tense, ProseRegister, Tone, Intensity,
                   classify_llm_error, inject_retry_notes, read_json, write_json)
from utils import log, schema   # scene-record registry: the drift guard below checks the model against it

# ---- Stage 3a: enrichment (LLM authors the semantic + facet fields) ----
# in:  scenes/pg{code}-s.json (flat records from segment.py, enrichment fields null)
# out: same file enriched IN PLACE. Per scene the LLM returns, in comprehend-before-judge order: a richer
# multi-clause `summary`; up to 6 `moments`, each a SENTENCE-FIRST SVOS clause PLUS a per-beat `tone` +
# `intensity` WORD; 3-5 `descriptors`; and the hard/soft facets `pov`, `tense`, `prose_word`. Scenes are
# enriched in BATCHES (one call covers a few scenes, one item per scene) and each scene is checkpointed
# before its record is mutated, so a run resumes. This stage writes ONLY the LLM fields — `svos`, the
# per-facet lists, `vdi_curve`, `dialogue_ratio`, `arc` and `prose_register` are DERIVED from these later
# (derive.py, Stage 3b); indexing is Stage 3c. OWNERSHIP: the prompt (EMBED_PROMPT in utils/llm.py) and
# the model/retry tuning (MODEL_PARAMS, BATCH_CHAR_LIMIT / BATCH_SCENE_LIMIT) are the USER'S surface.

# ---- tuning constants (model/prompt surface — the user's to tune) ----

BATCH_CHAR_LIMIT = 12000          # ~12-15k chars of scene prose packed per prompt
BATCH_SCENE_LIMIT = 4             # per-batch scene cap; a batch flushes at whichever trips first
                                  # (chars or count), mirroring data._pack's MAX_PARAGRAPHS
MAX_MOMENTS = 6                   # dramatic-unit scenes are longer than the old flavor cuts (PLAN D4)


# ---- batch enrichment schema (tag vocab enums live in utils/tags.py) ----

class Moment(BaseModel):
    # ONE pivotal beat. ORDER IS LOAD-BEARING: `sentence` is declared FIRST so the model writes the
    # bound SVOS clause, THEN fills subject/verb/object/setting by reading its OWN sentence back
    # (extraction, not invention); tone/intensity come LAST (judge the beat once it is understood).
    # The sentence becomes an `svos` multivector row (derived later); tone/intensity WORDS become one
    # (valence, dominance, intensity) sample of the scene's vdi_curve (derived later via utils.tags).
    sentence: str
    subject: str = ""
    verb: str = ""
    object: str = ""
    setting: str = ""
    tone: Tone           # per-beat feeling WORD (utils.tags.Tone); the enum type rejects out-of-vocab
    intensity: Intensity # per-beat strength WORD (utils.tags.Intensity)

    # Normalize the moment sentence to one uniform surface form (capital start, single trailing period); reject empty.
    @field_validator("sentence")
    @classmethod
    def _clean_sentence(cls, v: str) -> str:
        v = re.sub(r"\s+", " ", v or "").strip()
        if not v:
            raise ValueError("moment sentence must be non-empty")
        v = v[0].upper() + v[1:]     # uniform surface form: capital start ...
        v = v.rstrip(" .") + "."     # ... and exactly one trailing period
        return v

    # Coerce a part before validation: None -> "", a list -> its first term.
    @field_validator("subject", "verb", "object", "setting", mode="before")
    @classmethod
    def _coerce_part(cls, v):
        if v is None:
            return ""
        if isinstance(v, list):
            return v[0] if v else ""
        return v

    # Trim + collapse whitespace; a missing part stays "".
    @field_validator("subject", "verb", "object", "setting")
    @classmethod
    def _clean_part(cls, v: str) -> str:
        return re.sub(r"\s+", " ", v or "").strip()

    # Lowercase/trim the affect WORD before the enum check, so "High" / " Dread " still resolve in-vocab.
    @field_validator("tone", "intensity", mode="before")
    @classmethod
    def _norm_word(cls, v):
        return v.strip().lower() if isinstance(v, str) else v


class SceneEnrichment(BaseModel):
    # one scene's full enrichment. `index` ties it back to its slot in the batch. Field order is
    # COMPREHEND-BEFORE-JUDGE (§5.3): understand the scene (summary -> moments) before judging its
    # facets (descriptors -> pov -> tense -> prose_word). The field SET (minus `index`) must equal
    # schema.LLM_FIELDS — the import-time drift guard at the bottom of this file enforces it.
    index: int
    summary: str
    # moments: the up-to-6 pivotal beats, written SENTENCE-FIRST (see Moment), each carrying a per-beat
    # tone + intensity word. `summary` stays the holistic vector; the sentences seed the `svos` beats.
    moments: list[Moment]
    descriptors: list[str] = Field(min_length=3, max_length=5)
    pov: POV               # narrative point of view — a HARD facet (utils.tags.POV)
    tense: Tense           # narrative tense — a HARD facet (utils.tags.Tense)
    prose_word: ProseRegister  # register WORD (utils.tags.ProseRegister) -> the soft `prose_register` float later

    # Normalize the summary to one uniform surface form (capital start, single trailing period); reject empty.
    @field_validator("summary")
    @classmethod
    def _clean(cls, v: str) -> str:
        v = re.sub(r"\s+", " ", v or "").strip()
        if not v:
            raise ValueError("summary must be non-empty")
        v = v[0].upper() + v[1:]     # uniform surface form: capital start ...
        v = v.rstrip(" .") + "."     # ... and exactly one trailing period
        return v

    # Normalize descriptors to 3-5 lowercase non-empty adjectives (raises otherwise).
    @field_validator("descriptors")
    @classmethod
    def _norm_desc(cls, v: list[str]) -> list[str]:
        cleaned = [d.strip().lower() for d in v if d and d.strip()]
        if not (3 <= len(cleaned) <= 5):
            raise ValueError("descriptors must have 3-5 non-empty items")
        return cleaned

    # Require at least one beat; keep the first MAX_MOMENTS (the prompt targets up to 6).
    @field_validator("moments")
    @classmethod
    def _cap_moments(cls, v: list[Moment]) -> list[Moment]:
        if not v:
            raise ValueError("need at least one moment")
        return v[:MAX_MOMENTS]

    # Lowercase/trim the facet WORD before the enum check ("Third person" still fails, but "Third" / "PAST" resolve).
    @field_validator("pov", "tense", "prose_word", mode="before")
    @classmethod
    def _norm_facet(cls, v):
        return v.strip().lower() if isinstance(v, str) else v


class BatchEnrichment(BaseModel):
    items: list[SceneEnrichment]


BATCH_TOOL = pydantic_function_tool(
    BatchEnrichment,
    name="output_enrichment",
    description="Return the summary, moments, descriptors, and pov/tense/prose facets for EVERY scene in the batch.",
)
# NON-STRICT: providers that don't enforce JSON schema get dropped by require_parameters:True when
# strict is set. Pydantic + the coverage-retry loop validate the tool args instead.
BATCH_TOOL["function"]["strict"] = False


# ---- LLM helper: one forced tool call + the shared retry loop (retry/temperature policy is the user's) ----

# System-prompt addendum for a RETRY (the scenes missed on earlier attempts, replayed); "" on the first attempt.
# The generic slot-[1] splice lives in utils.llm.inject_retry_notes; this is just enrichment's wording.
def _retry_note(notes: list[str]) -> str:
    if not notes:
        return ""
    lines = "\n".join(f"- attempt {i + 1}: {n}" for i, n in enumerate(notes))
    return ("\n\n# RETRY — ENRICH THE SCENES YOU MISSED\n"
            "Earlier attempts on THIS SAME batch did not return one item per scene. "
            "Return EXACTLY one item per input index now — cover every index once, no "
            "gaps, no duplicates, no indices that were not in the input. Problems from "
            f"previous attempts:\n{lines}")


# One forced tool call validated into `model_cls` (generic over system_prompt / tool / model_cls). The tool
# name already matches MODEL_PARAMS' tool_choice (output_enrichment), so MODEL_PARAMS is splatted as-is.
# Retries never abort: fresh convo + replayed misses; only a fatal API error raises.
def _run_tool(user_content: str, validate=None, *,
              system_prompt: list = EMBED_PROMPT, tool: dict = BATCH_TOOL,
              model_cls=BatchEnrichment, note_fn=_retry_note):
    TEMP_FREEZE_ATTEMPTS = 10   # attempts before the temperature stops climbing (hard cap)
    tool_name = tool["function"]["name"]
    notes = []                  # misses from earlier attempts, replayed in the system note
    transient_tries = validation_tries = attempt = 0

    while True:
        temp = 0 if attempt == 0 else min(0.75, math.log(attempt ** 0.20) + 0.15)
        # FRESH conversation every attempt: earlier misses are replayed as a note appended to
        # the system prompt (no chat history carried).
        messages = [
            {"role": "system", "content": inject_retry_notes(system_prompt, notes, note_fn)},
            {"role": "user", "content": user_content},
        ]
        try:
            response = CLIENT.chat.completions.create(
                model=MODEL, temperature=temp, tools=[tool],
                messages=messages,
                **MODEL_PARAMS,   # tool_choice (== output_enrichment) + reasoning + routing, centralized in utils/llm.py
            )
        except Exception as e:
            if classify_llm_error(e) == "fatal":            # non-retryable 4xx -> abort
                raise RuntimeError(f"{tool_name} fatal (no retry): {e}") from e
            transient_tries += 1
            sleep = min(2 ** transient_tries, 30)
            # never give up: after the cap the temperature stops climbing and we keep retrying
            attempt = min(attempt + 1, TEMP_FREEZE_ATTEMPTS)
            log.warn(f"transient retry {transient_tries} (sleep {sleep}s, temp held ~{temp}): {e}")
            time.sleep(sleep)
            continue

        choices = response.choices
        msg = choices[0].message if choices else None
        if not msg or not msg.tool_calls:
            reason = f"did not call {tool_name}"
        else:
            args = msg.tool_calls[0].function.arguments
            try:
                data = model_cls.model_validate_json(args)   # schema-validate the tool args
            except (ValidationError, json.JSONDecodeError, ValueError) as e:
                reason = f"arguments failed schema validation: {e}"
            else:
                ok, why = validate(data) if validate else (True, "")   # optional semantic check
                if ok:
                    return data
                reason = why

        # remember the miss; the NEXT attempt is a fresh conversation whose system note
        # reminds the model what to fix this time.
        validation_tries += 1
        attempt = min(attempt + 1, TEMP_FREEZE_ATTEMPTS)
        notes.append(reason)
        log.warn(f"validation retry {validation_tries} (fresh convo, temp: {temp}): {reason[:120]}")


# ---- enrichment ----

# ** LOCKED **
# Scene prose with markup stripped and whitespace collapsed (LLM input, not stored).
def _plain(text_html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text_html or "")).strip()


# Group scenes into batches, flushing at whichever trips first: `limit` (chars) or `scene_limit` (count) — mirrors data._pack; a lone over-budget scene is kept whole.
def _batches(records: list[dict], limit: int = BATCH_CHAR_LIMIT,
             scene_limit: int = BATCH_SCENE_LIMIT):
    batch, size = [], 0
    for r in records:
        n = len(_plain(r.get("text_html", "")))
        over_chars = size + n > limit
        over_count = len(batch) >= scene_limit
        if batch and (over_chars or over_count):
            yield batch
            batch, size = [], 0
        batch.append(r)
        size += n
    if batch:
        yield batch


# Turn one validated scene item into the plain-dict enrichment payload (checkpoint-safe; enum WORDS as strings).
def _item_to_dict(it: SceneEnrichment) -> dict:
    return {
        "summary": it.summary,
        "moments": [m.model_dump(mode="json") for m in it.moments],   # tone/intensity enums -> word strings
        "descriptors": it.descriptors,
        "pov": it.pov.value,
        "tense": it.tense.value,
        "prose_word": it.prose_word.value,
    }


# Enrich one batch in a single coverage-validated call -> per-scene enrichment dict in batch order.
def _enrich_batch(batch: list[dict]) -> list[dict]:
    payload = json.dumps({"scenes": [
        {"index": i, "chapter_title": r.get("chapter_title"), "text": _plain(r.get("text_html", ""))}
        for i, r in enumerate(batch)
    ]}, ensure_ascii=False)
    expected = set(range(len(batch)))

    # Coverage check: exactly one item per scene index, no gaps / duplicates / extras.
    def validate(data: BatchEnrichment):
        idxs = [it.index for it in data.items]
        counts = Counter(idxs)
        s = set(idxs)
        missing = sorted(expected - s)
        extra = sorted(s - expected)
        dupes = sorted(i for i, n in counts.items() if n > 1)
        if not (missing or extra or dupes):
            return True, ""
        parts = []
        if missing: parts.append(f"missing indices {missing}")
        if dupes:   parts.append(f"duplicate indices {dupes}")
        if extra:   parts.append(f"indices not in input {extra}")
        return False, "; ".join(parts)

    data = _run_tool(payload, validate=validate)               # forced enrichment call + retry loop
    by_idx = {it.index: it for it in data.items}
    return [_item_to_dict(by_idx[i]) for i in range(len(batch))]


# Write one scene's enrichment onto its record (LLM fields only — svos / facets / vdi_curve are DERIVED later).
def _apply(rec: dict, enriched: dict):
    rec["summary"] = enriched["summary"]
    rec["moments"] = enriched.get("moments") or []   # .get keeps older checkpoints loadable
    rec["descriptors"] = enriched["descriptors"]
    rec["pov"] = enriched["pov"]
    rec["tense"] = enriched["tense"]
    rec["prose_word"] = enriched["prose_word"]
    rec["enriched"] = True
    rec["enrich_model"] = MODEL


# ** MAIN ** — tests + the Stage-3 driver enrich each book here
# Enrich every scene in one scenes json (resumable) and rewrite in place; returns the records.
def enrich_file(path: Path) -> list[dict]:
    log.step(f"enriching book {PurePath(path).name[2:-7]}")
    records = read_json(path, [])                              # read_write: the book's flat scene records
    if not records:
        return records

    code = records[0]["book_id"]
    ckpt = Checkpoint(SrcPaths.ENRICH_CKPT_DIR, f"pg{code}")   # per-scene resume cache (plain-dict payloads)
    print_lock = threading.Lock()

    # resume: apply anything already enriched or checkpointed; only the rest hit the LLM
    todo = []
    for r in records:
        if r.get("enriched") and r.get("summary"):
            continue
        cached = ckpt.load(r["scene_id"])                     # reuse a checkpointed result
        if cached is not None:
            try:
                _apply(r, cached)
                continue
            except Exception:
                pass  # corrupt checkpoint -> recompute
        todo.append(r)

    # One LLM call per batch; checkpoint each scene before mutating the record.
    def work(batch):
        results = _enrich_batch(batch)                        # forced enrichment call
        for r, res in zip(batch, results):
            ckpt.save(r["scene_id"], res)                     # persist before mutating
            _apply(r, res)
        with print_lock:
            log.info(f"batch ({len(batch)}): {', '.join(r['scene_id'] for r in batch)}")
        return batch

    batches = list(_batches(todo))                            # char/count-budget batches
    if batches:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            list(ex.map(work, batches))                       # parallel batch enrichment

    write_json(path, records)                                 # read_write: rewrite the enriched json in place
    ckpt.clear()   # book done: checkpoints no longer needed
    log.done(f"enriched {len(records)} scenes -> {path}")
    return records


# ---- schema drift guard (import-time) ----
# The hand-authored enrichment model is the user's tuning surface, so it stays hand-written — but its
# FIELD SET must match the registry, or editing scene_schema.json would silently desync what the index
# stores from what the LLM returns. Fail loudly here.
assert {n for n in SceneEnrichment.model_fields if n != "index"} == set(schema.LLM_FIELDS), (
    f"SceneEnrichment fields {sorted(n for n in SceneEnrichment.model_fields if n != 'index')} "
    f"!= schema LLM_FIELDS {sorted(schema.LLM_FIELDS)} — sync scene_schema.json or the model")
