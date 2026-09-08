import json, re, time, math, threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from openai import pydantic_function_tool
from pydantic import BaseModel, ValidationError
from typing import Literal

from data import gate_facts
from utils import (read_json, write_json, MODEL, MODEL_PARAMS, CLIENT, WORKERS, PROCESS_PROMPT,
                   classify_llm_error, Checkpoint, log, schema, inject_retry_notes, SrcPaths)

# ---- Stage 2: segmentation (label each paragraph -> reconstruct dramatic-unit scenes) ----
# The LLM reads a whole chunk (Chunk.scene_payload: read-only context + indexed paragraphs) and
# returns ONE boundary label per indexed paragraph — SCENE_START / CONTINUE / NOISE. Scenes, the
# cross-chunk stitch, and the noise drop all FALL OUT of that label stream: a run of CONTINUE after a
# SCENE_START is one scene; a scene whose CONTINUE run crosses a chunk boundary is stitched; NOISE is
# dropped and never breaks the open scene. A soft word-cap then splits any over-long scene at a
# paragraph break. segment_book is the single door: it runs the pre-gate, labels every chunk (parallel
# + checkpointed), reconstructs, and returns flat ingest-ready records (enrichment fields still null).
# The prompt (PROCESS_PROMPT, in utils/llm.py) and the retry/temperature policy are the USER'S tuning
# surface — do not touch unless asked. Behavior reference: PLAN §5.2 + Appendix A (process.py block).

# ---- LLM boundary-classification schema (forced output_labels tool) ----

SOFT_MAX_WORDS = 1500   # mechanical safety valve: split a longer scene at a paragraph break (tunable)


class ParagraphLabel(BaseModel):
    index: int                                          # the paragraph's GLOBAL index (from indexed_paragraphs)
    label: Literal["SCENE_START", "CONTINUE", "NOISE"]  # dramatic-unit boundary role

class ChunkLabels(BaseModel):
    # ParagraphLabel defined FIRST: this annotation is evaluated at class-definition time
    # (no `from __future__ import annotations`), so the forward name must already exist.
    labels: list[ParagraphLabel]

TOOL = pydantic_function_tool(
    ChunkLabels,
    name="output_labels",
    description="Return one boundary label for every indexed paragraph.",
)
# NON-STRICT: providers that don't enforce JSON schema (e.g. MiniMax) get dropped by
# require_parameters:True when strict is set. Pydantic + the label-retry loop validate.
TOOL["function"]["strict"] = False

# MODEL_PARAMS centralizes routing/reasoning but its tool_choice names the ENRICH tool; each stage
# overrides that shared tool_choice with its own tool, so force THIS stage's output_labels by name.
_TOOL_CHOICE = {"type": "function", "function": {"name": "output_labels"}}


# ---- coverage validation + retry-note assembly ----

# The indices the model must label: every indexed_paragraphs index (read-only context excluded).
def _expected_indices(payload: str) -> set[int]:
    obj = json.loads(payload)
    return {p["index"] for p in obj.get("indexed_paragraphs", [])}


# Verify every expected index is labelled exactly once -> (ok, reason for the model). Trivial now: one
# label per paragraph makes coverage automatic, so this only guards a malformed / short / stray answer.
def _validate_labels(data: ChunkLabels, expected: set[int]):
    got = [lab.index for lab in data.labels]
    counts = Counter(got)
    gotset = set(got)

    missing = sorted(expected - gotset)
    extra = sorted(gotset - expected)
    dupes = sorted(i for i, n in counts.items() if n > 1)

    if not (missing or extra or dupes):
        return True, ""

    parts = []
    if missing:
        parts.append(f"MISSING indices never labelled: {missing}")
    if dupes:
        parts.append(f"DUPLICATE indices labelled more than once: {dupes}")
    if extra:
        parts.append(f"indices NOT in the input: {extra}")
    return False, "; ".join(parts)


# The segmentation retry-reminder SECTION (prompt's `#`-section style), or "" on the first attempt.
# The generic slot-[1] splice lives in utils.llm.inject_retry_notes; this is just the segmenter's wording.
def _retry_note(notes: list[str]) -> str:
    if not notes:
        return ""
    lines = "\n".join(f"- previous attempt error (NEVER DO THIS AGAIN): {n}" for n in notes)
    return ("# RETRY — LABEL EVERY PARAGRAPH WHILE AVOIDING THESE ERRORS\n"
            "Earlier attempts on this section had these errors. FIX THIS: "
            f"\n{lines}\n")


# ---- SceneBreaker: one chunk -> paragraph labels, with the retry loop (retry/temperature policy is the user's) ----

class SceneBreaker:

    # ** MAIN ** — segment._label_book calls this once per chunk
    # Label one chunk via a forced output_labels call, retrying (fresh convo + note) until every
    # indexed paragraph is labelled exactly once; only a fatal API error raises.
    def break_chunk(self, file_code: str, chunk: str, chunk_index) -> ChunkLabels:
        TEMP_FREEZE_ATTEMPTS = 10   # attempts before the temperature stops climbing (hard cap)
        expected = _expected_indices(chunk)                 # indices this chunk must label
        notes = []                  # label misses from earlier attempts, replayed in the system note
        transient_tries = 0
        validation_tries = 0
        attempt = 0                 # drives temp bump across all retries

        while True:
            # temp 0, increase if attempts fail
            temp = 0 if attempt == 0 else min(0.75, math.log(attempt ** 0.20) + 0.15)
            # FRESH conversation every attempt: no chat history is carried; the paragraphs mislabelled
            # on earlier tries are replayed as a note appended to the system prompt.
            messages = [
                {"role": "system", "content": inject_retry_notes(PROCESS_PROMPT, notes, _retry_note)},  # splice the segmenter's retry wording
                {"role": "user", "content": chunk},
            ]
            try:
                log.info(f"book {file_code}: sending chunk {chunk_index} to {MODEL}")
                response = CLIENT.chat.completions.create(
                    model=MODEL, temperature=temp, tools=[TOOL],
                    messages=messages,
                    **{**MODEL_PARAMS, "tool_choice": _TOOL_CHOICE},   # keep routing/reasoning; force THIS stage's tool
                )
            except Exception as e:
                # only API/network failures land here; parse/coverage handled below
                if classify_llm_error(e) == "fatal":        # non-retryable 4xx -> abort
                    raise RuntimeError(f"break_chunk fatal (no retry): {e}") from e
                transient_tries += 1
                sleep = min(2 ** transient_tries, 30)
                # never give up: after the cap the temperature stops climbing and we keep retrying
                attempt = min(attempt + 1, TEMP_FREEZE_ATTEMPTS)
                log.warn(f"transient retry {transient_tries} (sleep {sleep}s, temp held ~{temp}): {e}")
                time.sleep(sleep)
                continue

            # inspect the response: no tool_call -> bad args -> incomplete labelling
            choices = response.choices
            msg = choices[0].message if choices else None
            if not msg or not msg.tool_calls:
                reason = "did not call the output_labels tool"
            else:
                args = msg.tool_calls[0].function.arguments
                try:
                    data = ChunkLabels.model_validate_json(args)   # schema-validate the tool args
                except (ValidationError, json.JSONDecodeError, ValueError) as e:
                    reason = f"arguments failed schema validation: {e}"
                else:
                    ok, why = _validate_labels(data, expected)      # every index once?
                    if ok:
                        return data
                    reason = why

            # remember the miss; the NEXT attempt is a fresh conversation whose system note
            # reminds the model to label the paragraphs it missed this time.
            validation_tries += 1
            attempt = min(attempt + 1, TEMP_FREEZE_ATTEMPTS)
            notes.append(reason)
            if len(notes) > 4:
                notes.pop(0)
            log.warn(f"validation retry {validation_tries} (fresh convo, temp held ~{temp}): {reason[:140]}")


# ---- label a whole book (parallel per-chunk, checkpointed) ----

# Label every chunk of a book (one forced LLM call each, parallel + per-chunk checkpointed) and merge
# into one global {paragraph index -> label} map. Chunk / label order is irrelevant — reconstruction
# sorts by the global index, so the cross-chunk stitch falls straight out of the merged stream.
def _label_book(book, checkpoint_base, workers: int) -> dict:
    ckpt = Checkpoint(checkpoint_base, f"pg{book.file_code}",       # per-book resume cache (typed codec)
                      load=ChunkLabels.model_validate,
                      dump=lambda d: d.model_dump(mode="json"))
    sb = SceneBreaker()
    log_lock = threading.Lock()

    # Label one chunk with a single LLM call, reusing a checkpoint if present.
    def work(chunk):
        key = f"chunk-{chunk.chunk_index}"
        cached = ckpt.load(key)                                     # resume: skip the LLM if cached
        if cached is not None:
            with log_lock:
                log.info(f"book {book.file_code}: chunk {chunk.chunk_index} cached (skip LLM)")
            return cached
        data = sb.break_chunk(book.file_code, chunk.scene_payload(), chunk.chunk_index)   # forced LLM label
        ckpt.save(key, data)   # persist before returning, so a crash survives
        with log_lock:
            log.info(f"book {book.file_code}: chunk {chunk.chunk_index} verified")
        return data

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(work, book.chunks))                  # parallel per-chunk labelling

    label_of = {lab.index: lab.label for data in results for lab in data.labels}
    ckpt.clear()   # book fully labelled: checkpoints no longer needed
    return label_of


# ---- reconstruct scenes from the label stream ----

# Group the global label stream into scenes: SCENE_START opens one, CONTINUE extends the open scene,
# NOISE is dropped (and never breaks it). Each scene = {indices, broken}, where `broken` flags a
# CONTINUE with no open scene — a dangling head at the very start of the stream.
def _scenes_from_labels(label_of: dict) -> list[dict]:
    scenes: list[dict] = []
    cur = None
    for idx in sorted(label_of):
        label = label_of[idx]
        if label == "NOISE":
            continue
        if label == "SCENE_START" or cur is None:
            cur = {"indices": [idx], "broken": label == "CONTINUE"}
            scenes.append(cur)
        else:                                   # CONTINUE with an open scene
            cur["indices"].append(idx)
    return scenes


# Split one scene's kept indices so each piece's word count stays <= SOFT_MAX_WORDS, cutting at the
# nearest paragraph break (a lone over-cap paragraph is kept whole, like data._pack).
def _cap_split(indices: list[int], word_of: dict) -> list[list[int]]:
    pieces, current, words = [], [], 0
    for idx in indices:
        w = word_of[idx]
        if current and words + w > SOFT_MAX_WORDS:
            pieces.append(current)
            current, words = [], 0
        current.append(idx)
        words += w
    if current:
        pieces.append(current)
    return pieces


# Derive one scene piece's stitch status from its chunk span: a broken head -> "broken_stitch"; a piece
# spanning >1 chunk -> "stitched" (a CONTINUE run crossed a chunk boundary); otherwise "complete".
def _stitch_status(indices: list[int], chunk_of: dict, broken_head: bool) -> str:
    if broken_head:
        return "broken_stitch"
    return "stitched" if len({chunk_of[i] for i in indices}) > 1 else "complete"


# ** MAIN ** — segment_book flattens a book's labels into ingest-ready records here
# Reconstruct scenes from the label stream, apply the soft word-cap, and flatten into flat records
# (one record == one future Qdrant point; enrichment fields start null). Every record starts from
# schema.blank_record() so the shape is defined ONCE in scene_schema.json; only what SEGMENTATION knows
# is filled. Vectors + the Qdrant envelope are added later — keep this DB-agnostic.
def _build_records(book, metadata: dict, label_of: dict) -> list[dict]:
    text_of, chapter_of, chunk_of, word_of = {}, {}, {}, {}
    for chunk in book.chunks:
        for p in chunk.paragraphs:
            text_of[p.index] = p.text
            chapter_of[p.index] = chunk.chapter_heading
            chunk_of[p.index] = chunk.chunk_index
            word_of[p.index] = len(re.sub(r"<[^>]+>", " ", p.text).split())

    # scenes -> soft-capped pieces; each piece becomes one record (broken flag rides the FIRST piece only)
    pieces: list[tuple[list[int], bool]] = []
    for scene in _scenes_from_labels(label_of):
        for k, idxs in enumerate(_cap_split(scene["indices"], word_of)):
            pieces.append((idxs, scene["broken"] and k == 0))

    code = book.file_code
    author = metadata.get("Author")
    language = metadata.get("Language")
    last = len(pieces) - 1
    records = []
    for i, (idxs, broken_head) in enumerate(pieces):
        start, end = idxs[0], idxs[-1]                          # min/max: kept indices are ascending
        # join only the KEPT paragraphs (interior noise already dropped from idxs)
        text = "<p>" + "</p><p>".join(text_of[j].strip() for j in idxs) + "</p>"
        word_count = len(re.sub(r"<[^>]+>", " ", text).split())
        rec = schema.blank_record()                            # null template from the registry
        rec.update({
            "scene_id": f"{code}-{i}",
            "book_id": code,
            "prev_scene_id": f"{code}-{i-1}" if i > 0 else None,
            "next_scene_id": f"{code}-{i+1}" if i < last else None,
            "chapter_title": chapter_of.get(start),
            "stitch_status": _stitch_status(idxs, chunk_of, broken_head),   # complete | stitched | broken_stitch
            "start_paragraph_index": start,
            "end_paragraph_index": end,
            "word_count": word_count,
            "author": author,
            "language": language,
            "book_metadata": metadata,          # already serialized (Subjects as list) by gate_facts
            "text_html": text,                  # render form; embed source is `summary`
        })
        records.append(rec)

    return records


# ---- pre-segmentation gate (which books must NOT be segmented) ----

EXCLUDE_SUBJECT_WORDS = ("poems", "poetry", "plays", "drama")   # non-prose by Gutenberg subject
EXCLUDED_BOOKS_FILE = "excluded-books.json"                     # running log of rejected books


# Append one rejected book to the running excluded-books json (auditable: reason + matched detail).
def _log_exclusion(exclude_dir, code, metadata: dict, reason, detail=None):
    path = f"{exclude_dir}/{EXCLUDED_BOOKS_FILE}"
    excluded = read_json(path, {})
    excluded[code] = {"reason": reason, "detail": detail, "metadata": metadata}
    write_json(path, excluded)


# Decide whether ONE book may be segmented from its pre-gate facts: US-public-domain (dc.rights) +
# non-prose (subject) gates. Logs the rejection and returns its reason, or None to proceed.
def _presegmentation_gate(code, facts: dict, exclude_dir) -> str | None:
    if facts["rights"] != "Public domain in the USA.":
        _log_exclusion(exclude_dir, code, facts["metadata"], "not public domain", facts["rights"])
        log.skip(f"pg{code}: dc.rights={facts['rights']!r} not US public domain — not segmenting "
                 f"(logged under {exclude_dir}/{EXCLUDED_BOOKS_FILE})")
        return "not public domain"

    matched = sorted(s for s in facts["subjects"]
                     if any(w in s.lower() for w in EXCLUDE_SUBJECT_WORDS))   # poetry/plays/drama
    if matched:
        _log_exclusion(exclude_dir, code, facts["metadata"], "non prose", matched)
        log.skip(f"pg{code}: subjects {matched} — not segmenting "
                 f"(logged under {exclude_dir}/{EXCLUDED_BOOKS_FILE})")
        return "non prose"

    return None


# ---- segment a whole book (the one door) ----

# ** MAIN ** — tests.segment_test segments each book through this single door
# Segment ONE whole book end to end: run the pre-gate, label every chunk (parallel + checkpointed),
# reconstruct dramatic-unit scenes from the labels, soft-cap, and return flat ingest-ready records
# (an empty list if the book is gated out). Keep DB-agnostic.
def segment_book(book, md: dict, checkpoint_base=SrcPaths.CHECKPOINT_DIR,
                 data_path=SrcPaths.DATA_DIR, exclude_dir=SrcPaths.RECALL_DIR,
                 workers: int = WORKERS) -> list[dict]:
    facts = gate_facts(book.file_code, md, data_path)      # data: rights + subjects + serialized metadata (one door)
    if _presegmentation_gate(book.file_code, facts, exclude_dir):
        return []                                          # excluded: no records

    label_of = _label_book(book, checkpoint_base, workers)          # forced LLM label per chunk -> {index: label}
    return _build_records(book, facts["metadata"], label_of)        # labels -> scenes -> flat records
