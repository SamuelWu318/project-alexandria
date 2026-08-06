# enrichment + indexing pipeline
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
# NOTE (prompt/model tuning is the user's): edit BATCH_SYSTEM_PROMPT, ENRICH_EFFORT
# and BATCH_CHAR_LIMIT below to retune. ENRICH_EFFORT is lower than the segmenter's
# "high" because classifying already-cut scenes is easier than cutting.
# -----------------------------------------------------------------------------
import sys, json, re, time, threading, shutil
from collections import Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, ValidationError, Field, field_validator
from openai import pydantic_function_tool
from qdrant_client import QdrantClient, models

# reuse the single OpenRouter client, model, tag vocab enums, error policy
from process import CLIENT, MODEL, Tone, Intensity, Arc, _classify_error, SCHEMA_VERSION
# vector-store primitives shared with the read path (search.py owns them)
from search import COLLECTION, VECTOR_NAMES, embed as _embed, point_id as _point_id, open_client


# --- constants --- #

SCENES_PATH = "master/scenes"
ENRICH_CKPT_DIR = "master/checkpoints/enrich"
ENRICH_WORKERS = 4                 # concurrent batches in flight
ENRICH_EFFORT = "medium"          # reasoning effort for enrichment; tune as needed
BATCH_CHAR_LIMIT = 13000          # ~12-15k chars of paragraph text packed per prompt


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
You enrich a BATCH of scenes. For EACH scene: FIRST classify its single dominant
FLAVOR, THEN write ONE general summary consistent with that flavor. Output ONLY a
call to output_enrichment.

# INPUT
A JSON object {"scenes": [ {"index", "scene_title", "chapter_title", "text"}, ... ]}.
`text` is the full scene prose. `index` identifies the scene.

# OUTPUT
output_enrichment with "items": exactly ONE object per input scene. Cover EVERY
index exactly once — no gaps, no duplicates, no indices that were not in the input.
Each item:
- index: copy the scene's index from the input.
- dominant_tone: the ONE dominant feeling (controlled vocabulary; pick the single
  closest term, do not average two feelings).
- intensity: low (faint wash) / moderate (clearly colours the scene) / high (dominates every line).
- arc: rising (builds toward the end) / steady (holds one level) / falling (subsides) / turn (flips partway).
- descriptors: 1-3 lowercase MODERN adjectives for the flavor (e.g. ["creeping","claustrophobic"]).
- summary: see below. Write it AFTER the flavor and keep it consistent with it.

# SUMMARY — MUST BE GENERAL, ONE SENTENCE
The summary is matched against short, GENERIC scene descriptions a writer types, so
generalize hard. Describe the TYPE of scene: the roles/archetypes and the dynamic or
subject — NOT proper names, NOT plot specifics.
- Style to imitate: "conversation between an authority figure and a rebellious recruit about listening to authority".
- Style to imitate: "a hunted man hiding in the dark under the mounting threat of discovery".
- Exactly ONE sentence. Terse. No conjunctions or extra clauses that chain events or
  lengthen it. No names, no quotes, no title, no book framing.

# RULES
- A scene is ONE flavor. Choose the single strongest tone.
- Judge only the words; ignore any residual markup.
- Call output_enrichment and nothing else.
"""


# --- LLM helper --- #

def _run_tool(system_prompt: str, user_content: str, tool: dict, tool_name: str,
              model_cls, effort: str = ENRICH_EFFORT, validate=None,
              max_transient_retries: int = 6, max_validation_retries: int = 3):
    # one forced tool call, validated into model_cls. Mirrors SceneBreaker's retry
    # policy: transient (network/429/5xx) backs off; bad/absent args get corrective
    # feedback appended to history; temp bumps across attempts. `validate(data)` ->
    # (ok, reason) adds a semantic check (e.g. batch index coverage) on top of schema.
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
    # scene prose without markup, whitespace collapsed (LLM input, not stored)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text_html or "")).strip()


def _batches(records: list[dict], limit: int = BATCH_CHAR_LIMIT):
    # pack scenes into prompt-sized groups by paragraph-text length. A scene is
    # never split; one over the limit becomes its own (over-budget) batch.
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
    # one call enriches the whole batch; returns per-scene {"tags":{...},"summary"}
    # aligned to batch order. Coverage-validated: one item per scene, no gaps/dupes.
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
    # write enrichment onto the record (prev_tone/next_tone done later, once all known)
    t = enriched["tags"]
    rec["dominant_tone"] = t["dominant_tone"]
    rec["intensity"] = t["intensity"]
    rec["arc"] = t["arc"]
    rec["descriptors"] = t["descriptors"]
    rec["summary"] = enriched["summary"]
    rec["enriched"] = True
    rec["enrich_model"] = MODEL


def enrich_file(path: Path) -> list[dict]:
    # enrich every scene in one scenes json, resumable via per-scene checkpoints,
    # then denormalize neighbor tones and write the file back in place.
    records = json.loads(path.read_text(encoding="utf-8"))
    if not records:
        return records

    code = records[0]["book_id"]
    ckpt_dir = Path(ENRICH_CKPT_DIR) / f"pg{code}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print_lock = threading.Lock()

    # resume: apply anything already enriched or checkpointed; only the rest hit the LLM
    todo = []
    for r in records:
        if r.get("enriched") and r.get("summary"):
            continue
        cpath = ckpt_dir / f"{r['scene_id']}.json"
        if cpath.exists():
            try:
                _apply(r, json.loads(cpath.read_text(encoding="utf-8")))
                continue
            except Exception:
                pass  # corrupt checkpoint -> recompute
        todo.append(r)

    def work(batch):
        # one LLM call per batch; checkpoint each scene before mutating the record
        results = _enrich_batch(batch)
        for r, res in zip(batch, results):
            (ckpt_dir / f"{r['scene_id']}.json").write_text(
                json.dumps(res, ensure_ascii=False), encoding="utf-8")
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

    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.rmtree(ckpt_dir, ignore_errors=True)   # book done: checkpoints no longer needed
    print(f"enriched {len(records)} scenes -> {path}")
    return records


# --- qdrant index (write path; config/embedder/id come from search.py) --- #

def _ensure_collection(client: QdrantClient, dim: int):
    # named-vector collection: "summary" and "descriptors", queryable alone or fused.
    # self-heals: an existing collection with the wrong vector shape is rebuilt.
    want = {n: models.VectorParams(size=dim, distance=models.Distance.COSINE)
            for n in VECTOR_NAMES}
    if client.collection_exists(COLLECTION):
        cfg = client.get_collection(COLLECTION).config.params.vectors
        if isinstance(cfg, dict) and set(cfg) == set(VECTOR_NAMES):
            return
        client.delete_collection(COLLECTION)   # single-vector / stale -> rebuild
    client.create_collection(COLLECTION, vectors_config=want)


def index_records(client: QdrantClient, records: list[dict]):
    # embed BOTH the summary and the descriptor adjectives (separate named vectors)
    # and upsert one point per scene. payload == the full flat record (the DB row);
    # id is the stable scene uuid (re-runs overwrite).
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


def main():
    # embed.py [code ...]   enrich + index those books (default: all scenes files).
    # the Qdrant search test now lives in tests.py.
    if len(sys.argv) > 1:
        files = [Path(f"{SCENES_PATH}/pg{c}-s.json") for c in sys.argv[1:]]
    else:
        files = sorted(Path(SCENES_PATH).glob("pg*-s.json"))

    client = open_client()   # first run downloads the embed model
    for f in files:
        if not f.exists():
            print(f"skip (missing): {f}")
            continue
        records = enrich_file(f)
        index_records(client, records)


if __name__ == "__main__":
    main()
