# FOR CLAUDE — Stage 3: enrichment + indexing pipeline.
# -----------------------------------------------------------------------------
# in:  scenes/pg{code}-s.json  (flat records from process.py, enrichment fields null)
# out: same file, enriched in place  +  a Qdrant collection of scene points.
#
# scenes are enriched in BATCHES: several scenes are packed into ONE prompt (up to
# BATCH_CHAR_LIMIT of paragraph text), and one call returns, per scene and IN ORDER,
# first the flavor (dominant_tone, intensity, arc, descriptors) then a GENERAL
# one-sentence summary consistent with it. Then neighbor tones are denormalized
# (prev_tone/next_tone) for Mode-2 search, and each scene is upserted as one Qdrant
# point with TWO named vectors (summary, descriptors) — queryable alone or fused.
#
# OWNERSHIP: prompt/model tuning is the user's — edit BATCH_SYSTEM_PROMPT,
# ENRICH_EFFORT, BATCH_CHAR_LIMIT to retune (ENRICH_EFFORT is below the segmenter's
# "high" because classifying already-cut scenes is easier than cutting). Paths +
# atomic JSON IO come from storage.py; the Qdrant contract comes from search.py.
# -----------------------------------------------------------------------------
import sys, json, re, time, threading, shutil
from collections import Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, ValidationError, Field, field_validator
from openai import pydantic_function_tool
from qdrant_client import QdrantClient, models

# reuse the single OpenRouter client, model, tag-vocab enums, error policy
from process import CLIENT, MODEL, Tone, Intensity, Arc, _classify_error, SCHEMA_VERSION
# vector-store primitives shared with the read path (search.py owns them)
from search import COLLECTION, VECTOR_NAMES, embed as _embed, point_id as _point_id, open_client
# paths + atomic JSON IO
from storage import SCENES_PATH, ENRICH_CKPT_DIR, STATUS_PATH, read_json, write_json


# --- tuning constants (model/prompt surface — the user's to tune) --- #

ENRICH_WORKERS = 6                # concurrent batches in flight
ENRICH_EFFORT = "medium"          # reasoning effort for enrichment; tune as needed
BATCH_CHAR_LIMIT = 12000          # ~12-15k chars of paragraph text packed per prompt


# --- batch enrichment schema (tag vocab enums live in process.py) --- #

class SceneEnrichment(BaseModel):
    # one scene's full enrichment. `index` ties it back to its slot in the batch.
    index: int
    dominant_tone: Tone
    intensity: Intensity
    arc: Arc
    descriptors: list[str] = Field(min_length=1, max_length=3)
    summary: str

    @field_validator("descriptors")
    @classmethod
    def _norm_desc(cls, v: list[str]) -> list[str]:
        cleaned = [d.strip().lower() for d in v if d and d.strip()]
        if not (1 <= len(cleaned) <= 3):
            raise ValueError("descriptors must have 1-3 non-empty items")
        return cleaned

    @field_validator("summary")
    @classmethod
    def _clean(cls, v: str) -> str:
        v = re.sub(r"\s+", " ", v or "").strip()
        if not v:
            raise ValueError("summary must be non-empty")
        v = v[0].upper() + v[1:]     # uniform surface form: capital start ...
        v = v.rstrip(" .") + "."     # ... and exactly one trailing period
        return v


class BatchEnrichment(BaseModel):
    items: list[SceneEnrichment]


BATCH_TOOL = pydantic_function_tool(
    BatchEnrichment,
    name="output_enrichment",
    description="Return flavor tags + one general summary for EVERY scene in the batch.",
)


BATCH_SYSTEM_PROMPT = """
# ROLE
You enrich a BATCH of scenes. For EACH scene: FIRST find its single dominant
TONE, THEN write ONE general summary consistent with that tone. Output ONLY a
call to output_enrichment.

# INPUT
A JSON object {"scenes": [ {"index", "scene_title", "chapter_title", "text"}, ... ]}.
`text` is the full scene prose. `index` identifies the scene.

# OUTPUT
output_enrichment with "items": ONE object per input scene. Cover EVERY
index exactly once — no gaps, no duplicates, no indices that were not in input.
Each item:
- index: the scene's index from the input.
- dominant_tone: the ONE dominant feeling (controlled vocabulary; pick the single cleanest term).
- intensity: low (background) / moderate (visible) / high (dominates).
- arc: rising / steady / falling / turn (tone/conflict changes drastically).
- descriptors: 3-5 lowercase MODERN adjectives for the flavor (e.g. ["creeping","claustrophobic"]).
- summary: see below. Write it AFTER the flavor and keep it consistent with it.

# SUMMARY — GENERAL, ONE SITUATION, READABLE SENTENCE
The summary IS the search target: a writer types a short generic scene description
and it must match. Write it as ONE clear, natural, grammatical sentence that reads
well on its own.
- Describe the TYPE of scene: roles/archetypes plus one action or dynamic. Nothing unique to that book should be included.
- ONE situation: one actor or relationship in one situation doing one action. You may add a short clause that colours the SAME moment, but not a second moment or secoond set of characters (NOT "X does A while Y does B").
- NO feeling words (grief, dread, joy, desperate, tender, aching...). The emotion is already carried by dominant_tone + descriptors. State only the situation.
- ~10-18 words. Present tense. Start with a capital, end with a period. No quotes, no title, no book framing.
- Examples:
  - An authority figure instructs a young recruit at training grounds to trust reason over emotion.
  - A hunted man hides in a dark forest at night as his pursuer draws near.
  - A defeated challenger kneels in a grand hall to offer his sword to the victor.

# RULES
- A scene is ONE flavor. Choose the single strongest tone.
- Judge only the words; ignore any residual markup.
- Call output_enrichment and nothing else.
"""


# --- LLM helper --- #

def _run_tool(system_prompt: str, user_content: str, tool: dict, tool_name: str,
              model_cls, effort: str = ENRICH_EFFORT, validate=None,
              max_transient_retries: int = 6, max_validation_retries: int = 3):
    """One forced tool call validated into `model_cls`, using SceneBreaker's retry policy.

    Transient errors back off; bad/absent args get corrective feedback appended to
    history; temp bumps across attempts. Optional `validate(data) -> (ok, reason)`
    adds a semantic check (e.g. batch index coverage) on top of schema validation.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    transient_tries = validation_tries = attempt = 0

    while True:
        temp = 0 if attempt == 0 else min(0.6, 0.15 * attempt)
        try:
            response = CLIENT.chat.completions.create(
                model=MODEL, temperature=temp, tools=[tool],
                messages=messages,
                tool_choice={"type": "function", "function": {"name": tool_name}},
                extra_body={"provider": {"require_parameters": True},
                            "reasoning": {"effort": effort}},
            )
        except Exception as e:
            if _classify_error(e) == "fatal":
                raise RuntimeError(f"{tool_name} fatal (no retry): {e}") from e
            transient_tries += 1
            if transient_tries > max_transient_retries:
                raise RuntimeError(f"{tool_name} transient failure after "
                                   f"{max_transient_retries} retries: {e}") from e
            sleep = min(2 ** transient_tries, 30)
            print(f"  transient retry {transient_tries}/{max_transient_retries} "
                  f"(sleep {sleep}s): {e}")
            time.sleep(sleep)
            attempt += 1
            continue

        choices = response.choices
        msg = choices[0].message if choices else None
        if not msg or not msg.tool_calls:
            echo = (msg.content if msg else "") or "(empty response, no tool call)"
            correction = (f"You did NOT call {tool_name}. Respond ONLY with a call "
                          f"to {tool_name} and nothing else.")
        else:
            args = msg.tool_calls[0].function.arguments
            echo = args
            try:
                data = model_cls.model_validate_json(args)
            except (ValidationError, json.JSONDecodeError, ValueError) as e:
                correction = (f"Your {tool_name} arguments failed schema validation: {e}. "
                              f"Return arguments that exactly match the schema.")
            else:
                ok, why = validate(data) if validate else (True, "")
                if ok:
                    return data
                correction = (f"Your {tool_name} output is incomplete: {why}. "
                              f"Return one item per input scene, covering every index exactly once.")

        validation_tries += 1
        if validation_tries > max_validation_retries:
            raise RuntimeError(f"{tool_name} validation failure after "
                               f"{max_validation_retries} retries: {correction}")
        print(f"  validation retry {validation_tries}/{max_validation_retries}: {correction[:120]}")
        messages.append({"role": "assistant", "content": str(echo)})
        messages.append({"role": "user", "content": correction})
        attempt += 1


# --- enrichment --- #

def _plain(text_html: str) -> str:
    """Scene prose with markup stripped and whitespace collapsed (LLM input, not stored)."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text_html or "")).strip()


def _batches(records: list[dict], limit: int = BATCH_CHAR_LIMIT):
    """Group scenes into prompt-sized batches by text length; a scene is never split."""
    batch, size = [], 0
    for r in records:
        n = len(_plain(r.get("text_html", "")))
        if batch and size + n > limit:
            yield batch
            batch, size = [], 0
        batch.append(r)
        size += n
    if batch:
        yield batch


def _enrich_batch(batch: list[dict]) -> list[dict]:
    """Enrich one batch in a single call -> per-scene {"tags", "summary"} in batch order.

    Coverage-validated: exactly one item per scene, no gaps or duplicates.
    """
    payload = json.dumps({"scenes": [
        {"index": i, "scene_title": r.get("scene_title"),
         "chapter_title": r.get("chapter_title"), "text": _plain(r.get("text_html", ""))}
        for i, r in enumerate(batch)
    ]}, ensure_ascii=False)
    expected = set(range(len(batch)))

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

    data = _run_tool(BATCH_SYSTEM_PROMPT, payload, BATCH_TOOL, "output_enrichment",
                     BatchEnrichment, validate=validate)
    by_idx = {it.index: it for it in data.items}
    out = []
    for i in range(len(batch)):
        it = by_idx[i]
        out.append({
            "tags": {"dominant_tone": it.dominant_tone.value,
                     "intensity": it.intensity.value,
                     "arc": it.arc.value,
                     "descriptors": it.descriptors},
            "summary": it.summary,
        })
    return out


def _apply(rec: dict, enriched: dict):
    """Write one scene's enrichment onto its record (prev_tone/next_tone denormalized later)."""
    t = enriched["tags"]
    rec["dominant_tone"] = t["dominant_tone"]
    rec["intensity"] = t["intensity"]
    rec["arc"] = t["arc"]
    rec["descriptors"] = t["descriptors"]
    rec["summary"] = enriched["summary"]
    rec["enriched"] = True
    rec["enrich_model"] = MODEL


def enrich_file(path: Path) -> list[dict]:
    """Enrich every scene in one scenes json (resumable), denormalize tones, rewrite in place."""
    records = read_json(path, [])
    if not records:
        return records

    code = records[0]["book_id"]
    ckpt_dir = Path(ENRICH_CKPT_DIR) / f"pg{code}"
    print_lock = threading.Lock()

    # resume: apply anything already enriched or checkpointed; only the rest hit the LLM
    todo = []
    for r in records:
        if r.get("enriched") and r.get("summary"):
            continue
        cached = read_json(ckpt_dir / f"{r['scene_id']}.json")
        if cached is not None:
            try:
                _apply(r, cached)
                continue
            except Exception:
                pass  # corrupt checkpoint -> recompute
        todo.append(r)

    def work(batch):
        # one LLM call per batch; checkpoint each scene before mutating the record
        results = _enrich_batch(batch)
        for r, res in zip(batch, results):
            write_json(ckpt_dir / f"{r['scene_id']}.json", res, indent=None)
            _apply(r, res)
        with print_lock:
            print(f"**** BATCH ({len(batch)}) {', '.join(r['scene_id'] for r in batch)} ****")
        return batch

    batches = list(_batches(todo))
    if batches:
        with ThreadPoolExecutor(max_workers=ENRICH_WORKERS) as ex:
            list(ex.map(work, batches))

    # denormalize neighbor tones now that every dominant_tone is known (Mode-2)
    by_id = {r["scene_id"]: r for r in records}
    for r in records:
        p, n = r.get("prev_scene_id"), r.get("next_scene_id")
        r["prev_tone"] = by_id[p]["dominant_tone"] if p in by_id else None
        r["next_tone"] = by_id[n]["dominant_tone"] if n in by_id else None

    write_json(path, records)
    shutil.rmtree(ckpt_dir, ignore_errors=True)   # book done: checkpoints no longer needed
    print(f"enriched {len(records)} scenes -> {path}")
    return records


# --- qdrant index (write path; config/embedder/id come from search.py) --- #

def _ensure_collection(client: QdrantClient, dim: int):
    """Ensure the named-vector ("summary"/"descriptors") collection exists; rebuild if stale."""
    want = {n: models.VectorParams(size=dim, distance=models.Distance.COSINE)
            for n in VECTOR_NAMES}
    if client.collection_exists(COLLECTION):
        cfg = client.get_collection(COLLECTION).config.params.vectors
        if isinstance(cfg, dict) and set(cfg) == set(VECTOR_NAMES):
            return
        client.delete_collection(COLLECTION)   # single-vector / stale -> rebuild
    client.create_collection(COLLECTION, vectors_config=want)


def index_records(client: QdrantClient, records: list[dict]):
    """Embed summary + descriptors as two named vectors and upsert one point per scene.

    payload == the full flat record; id is the stable scene uuid, so re-runs overwrite.
    """
    ready = [r for r in records if r.get("summary")]
    if not ready:
        print("  no enriched summaries to index")
        return
    sum_vecs = _embed([r["summary"] for r in ready])
    # descriptors are 1-3 adjectives (schema-guaranteed); join to a vibe string.
    desc_vecs = _embed([", ".join(r.get("descriptors") or []) or r["summary"] for r in ready])
    _ensure_collection(client, len(sum_vecs[0]))
    points = [
        models.PointStruct(id=_point_id(r["scene_id"]),
                           vector={"summary": s, "descriptors": d}, payload=r)
        for r, s, d in zip(ready, sum_vecs, desc_vecs)
    ]
    client.upsert(COLLECTION, points=points)
    print(f"  indexed {len(ready)} points into '{COLLECTION}'")


# --- book completion status (skip finished books) --- #

def _load_status() -> dict:
    """Load the {book_id: status} map; missing file / key / null all mean "not done"."""
    return read_json(STATUS_PATH, {})


def _mark_status(book_id: str, value):
    """Persist one book's completion status so the next run can skip it."""
    status = _load_status()
    status[book_id] = value
    write_json(STATUS_PATH, status)


def main():
    """CLI: `embed.py [code ...]` enrich + index those books (default: all scenes files).

    A book marked "completed" in STATUS_PATH is skipped; set its status to null (or
    delete the file) to force a redo. Interactive tests live in tests.py.
    """
    if len(sys.argv) > 1:
        files = [Path(f"{SCENES_PATH}/pg{c}-s.json") for c in sys.argv[1:]]
    else:
        files = sorted(Path(SCENES_PATH).glob("pg*-s.json"))

    client = open_client()   # first run downloads the embed model
    status = _load_status()
    for f in files:
        if not f.exists():
            print(f"skip (missing): {f}")
            continue
        m = re.search(r"pg(\d+)-s\.json$", f.name)
        code = m.group(1) if m else f.stem
        if status.get(code) == "completed":
            print(f"skip (completed): {code}")
            continue
        records = enrich_file(f)
        index_records(client, records)
        _mark_status(code, "completed")   # persisted so the next run skips it
        print(f"marked {code} completed")


if __name__ == "__main__":
    main()
