# FOR CLAUDE — Stage 2: segmentation (cut chunks into flavor-pure scenes).
# -----------------------------------------------------------------------------
# SceneBreaker.break_chunk sends one section (Chunk.scene_payload) to the LLM and
# forces an output_scenes tool call labelling every paragraph scene/noise, wrapped
# in a retry loop (transient backoff + corrective-feedback re-ask) and a coverage
# check (every input index covered exactly once). scenes_to_records then stitches
# open-ended scenes across chunks and flattens them into one-record-per-scene dicts
# for embed.py (enrichment fields start null).
#
# OWNERSHIP: the prompts (PROCESS_PROMPT, in utils/llm.py) and the retry/temperature policy are the
# user's tuning surface — do not touch unless asked. Plumbing (IO via storage.py,
# docstrings, assembly) is fair game. MODEL, the LLM client, the error policy, the
# tag-vocab enums and SCHEMA_VERSION moved to storage.py (shared) — still the user's to
# tune there.
#
# Shared foundation: storage.py provides CLIENT, MODEL, _classify_error, SCHEMA_VERSION,
# Tone / Intensity / Arc, .env loading and paths/IO. This stage imports them from there.
# -----------------------------------------------------------------------------
import os, json, re, time, math, threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from openai import pydantic_function_tool
from pydantic import BaseModel, ValidationError
from typing import Literal

from data import MetadataParser, parse_rights
from utils import (read_json, write_json, SCHEMA_VERSION, MODEL, MODEL_PARAMS, CLIENT, WORKERS, PROCESS_PROMPT,
                   classify_llm_error, Checkpoint, log, schema) # scene-record registry: blank_record() is the null template a scene starts as

# --- constants (MODEL / CLIENT / _classify_error / SCHEMA_VERSION / WORKERS / PROCESS_PROMPT now come from utils) --- #

class SceneData(BaseModel):
    # metadata can be added later.
    start_paragraph_index: int
    end_paragraph_index: int
    paragraph_type: Literal["scene", "noise"]      # noise filter only: "noise" is dropped
    content_form: Literal["prose", "other", "noise"]  # prose story / non-prose story (poem, play) / apparatus; powers the book-level majority-embed filter
    open_start_index: bool
    open_end_index: bool
    title: str

class MultiSceneData(BaseModel):
    # SceneData defined FIRST: this annotation is evaluated at class-definition time
    # (no `from __future__ import annotations`), so the forward name must already exist.
    scenes_data: list[SceneData]

TOOL = pydantic_function_tool(
    MultiSceneData,
    name="output_scenes",
    description="Force return of scenes in structure."
)
# NON-STRICT: providers that don't enforce JSON schema (e.g. MiniMax) get dropped by
# require_parameters:True when strict is set. Pydantic + the coverage-retry loop validate.
TOOL["function"]["strict"] = False

# --- enrichment tag vocabulary ---
# Tone / Intensity / Arc moved to storage.py (the shared foundation); both this
# stage and embed.py import them from there. Still the user's to tune — see storage.py.


def _expected_indices(payload: str) -> set[int]:
    """Indices the model must cover: every indexed_paragraphs index (context excluded)."""
    obj = json.loads(payload)
    return {p["index"] for p in obj.get("indexed_paragraphs", [])}


def _validate_coverage(data: MultiSceneData, expected: set[int]):
    """Verify every expected index is covered exactly once -> (ok, reason for the model).

    Runs BEFORE noise extraction, so missing / duplicate / extra indices all fail.
    """
    covered = []
    for s in data.scenes_data:
        covered.extend(range(s.start_paragraph_index, s.end_paragraph_index + 1))
    counts = Counter(covered)
    cset = set(covered)

    missing = sorted(expected - cset)
    extra = sorted(cset - expected)
    dupes = sorted(i for i, n in counts.items() if n > 1)

    if not (missing or extra or dupes):
        return True, ""

    parts = []
    if missing:
        parts.append(f"MISSING indices never covered by any scene: {missing}")
    if dupes:
        parts.append(f"DUPLICATE/overlapping indices covered more than once: {dupes}")
    if extra:
        parts.append(f"indices NOT in the input: {extra}")
    return False, "; ".join(parts)


def _retry_note(notes: list[str]) -> str:
    """The retry-reminder SECTION (rendered in the prompt's own `#`-section style), or ""
    on the first attempt. It fills slot [1] of the PROCESS_PROMPT parts list via
    _inject_retry_notes — placed among the instructions, NOT tacked onto the end."""
    if not notes:
        return ""
    lines = "\n".join(f"- previous attempt error (NEVER DO THIS AGAIN): {n}" for i, n in enumerate(notes))
    return ("# RETRY — SEGMENT WHILE AVOIDING THESE ERRORS\n"
            "Earlier attempts on this section had these errors. FIX THIS: "
            f"\n{lines}\n")


def _inject_retry_notes(prompt: list, notes: list[str]) -> str:
    """Rebuild the system prompt with the retry reminder in slot [1] of the parts list
    (a structured position among the instructions), empty on the first attempt. Copies
    the list first, so the module-level PROCESS_PROMPT is never mutated (thread-safe)."""
    temp = prompt.copy()
    temp[1] = _retry_note(notes)
    return "".join(temp)


class SceneBreaker:
    def break_chunk(self, file_code: str, chunk: str, chunk_index: str):
        """Segment one section via a forced output_scenes call, retrying until coverage passes.

        Retries NEVER abort. Each retry starts a FRESH conversation (no chat history is
        carried); the paragraphs missed on earlier attempts are replayed as a note added
        to the system prompt. Transient errors back off; the temperature climbs only
        until TEMP_FREEZE_ATTEMPTS then HOLDS. Only a fatal API error raises.
        """
        TEMP_FREEZE_ATTEMPTS = 10   # attempts before the temperature stops climbing (hard cap)
        expected = _expected_indices(chunk)
        notes = []                  # coverage misses from earlier attempts, replayed in the system note
        transient_tries = 0
        validation_tries = 0
        attempt = 0                 # drives temp bump across all retries

        while True:
            # temp 0, increase if attempts fail
            temp = 0 if attempt == 0 else min(0.75, math.log(attempt ** 0.20) + 0.15)
            # FRESH conversation every attempt: no chat history is carried; the paragraphs
            # missed on earlier tries are replayed as a note appended to the system prompt.
            messages = [
                {"role": "system", "content": _inject_retry_notes(PROCESS_PROMPT, notes)},
                {"role": "user", "content": chunk},
            ]
            try:
                log.info(f"book {file_code}: sending chunk {chunk_index} to {MODEL}")
                response = CLIENT.chat.completions.create(
                    model=MODEL, temperature=temp, tools=[TOOL],
                    messages=messages,
                    **MODEL_PARAMS,   # tool_choice + reasoning + routing prefs, centralized in utils/llm.py
                )
            except Exception as e:
                # only API/network failures land here; parse/coverage handled below
                if classify_llm_error(e) == "fatal":
                    raise RuntimeError(f"break_chunk fatal (no retry): {e}") from e
                transient_tries += 1
                sleep = min(2 ** transient_tries, 30)
                # never give up: after the cap the temperature stops climbing and we keep
                # retrying at that held value (only a fatal API error above aborts).
                attempt = min(attempt + 1, TEMP_FREEZE_ATTEMPTS)
                log.warn(f"transient retry {transient_tries} (sleep {sleep}s, temp held ~{temp}): {e}")
                time.sleep(sleep)
                continue

            # inspect the response: no tool_call -> bad args -> incomplete coverage
            choices = response.choices
            msg = choices[0].message if choices else None
            if not msg or not msg.tool_calls:
                reason = "did not call the output_scenes tool"
            else:
                args = msg.tool_calls[0].function.arguments
                try:
                    data = MultiSceneData.model_validate_json(args)
                except (ValidationError, json.JSONDecodeError, ValueError) as e:
                    reason = f"arguments failed schema validation: {e}"
                else:
                    ok, why = _validate_coverage(data, expected)
                    if ok:
                        return data
                    reason = why

            # remember the miss; the NEXT attempt is a fresh conversation whose system
            # note reminds the model to segment the paragraphs it missed this time.
            validation_tries += 1
            attempt = min(attempt + 1, TEMP_FREEZE_ATTEMPTS)
            notes.append(reason)
            if len(notes) > 4:
                notes.pop(0)
            log.warn(f"validation retry {validation_tries} (fresh convo, temp held ~{temp}): {reason[:140]}")


# --- pre-segmentation gates (which books must NOT be segmented) --- #

EXCLUDE_SUBJECT_WORDS = ("poems", "poetry", "plays", "drama")   # non-prose by Gutenberg subject
EXCLUDED_BOOKS_FILE = "excluded-books.json"                     # running log of rejected books


def _log_exclusion(exclude_dir, code, md, reason, detail=None):
    """Append one rejected book to the running excluded-books json under exclude_dir.

    reason is "non prose" or "not public domain"; detail is the matched subjects or the
    dc.rights string that triggered it. Kept auditable for later look-back.
    """
    path = f"{exclude_dir}/{EXCLUDED_BOOKS_FILE}"
    excluded = read_json(path, {})
    excluded[code] = {"reason": reason, "detail": detail,
                      "metadata": MetadataParser.to_dict(md)}
    write_json(path, excluded)


def presegmentation_gate(code, md, data_path, exclude_dir) -> str | None:
    """Decide whether ONE book may be segmented. Returns an exclusion reason (already
    logged + printed) if it must be skipped, else None to proceed.

    gate 1 — US public domain only: read dc.rights straight from the book HTML.
    gate 2 — non-prose by subject: poetry / plays / drama in the Gutenberg Subjects.
    """
    rights = parse_rights(code, data_path)
    if rights != "Public domain in the USA.":
        _log_exclusion(exclude_dir, code, md, "not public domain", rights)
        log.skip(f"pg{code}: dc.rights={rights!r} not US public domain — not segmenting "
                 f"(logged under {exclude_dir}/{EXCLUDED_BOOKS_FILE})")
        return "not public domain"

    matched = sorted(s for s in (md.get("Subjects") or [])
                     if any(w in s.lower() for w in EXCLUDE_SUBJECT_WORDS))
    if matched:
        _log_exclusion(exclude_dir, code, md, "non prose", matched)
        log.skip(f"pg{code}: subjects {matched} — not segmenting "
              f"(logged under {exclude_dir}/{EXCLUDED_BOOKS_FILE})")
        return "non prose"

    return None


def segment_book(book, checkpoint_base, workers: int = WORKERS):
    """Segment ONE whole book into scenes: one forced LLM call per chunk, run in
    parallel, each chunk checkpointed so a crash resumes instead of re-paying. Returns
    the ordered, flattened list of scene objects (noise scenes INCLUDED — the caller
    filters, tallies, and records them). Checkpoints are dropped once the book finishes.

    `checkpoint_base` is the directory the per-book resume cache lives under — the
    caller owns paths, but this module owns the typed codec (it owns MultiSceneData).
    """
    ckpt = Checkpoint(checkpoint_base, f"pg{book.file_code}",
                         load=MultiSceneData.model_validate,
                         dump=lambda d: d.model_dump(mode="json"))
    sb = SceneBreaker()
    log_lock = threading.Lock()

    def work(chunk):
        """Segment one chunk with a single LLM call, reusing a checkpoint if present."""
        key = f"chunk-{chunk.chunk_index}"
        cached = ckpt.load(key)
        if cached is not None:
            with log_lock:
                log.info(f"book {book.file_code}: chunk {chunk.chunk_index} cached (skip LLM)")
            return cached
        data = sb.break_chunk(book.file_code, chunk.scene_payload(), chunk.chunk_index)
        ckpt.save(key, data)   # persist before returning, so a crash survives
        with log_lock:
            log.info(f"book {book.file_code}: chunk {chunk.chunk_index} verified")
        return data

    # ex.map preserves input order, so scenes stay in chunk (reading) order
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(work, book.chunks))

    scenes = [scene for data in results for scene in data.scenes_data]
    ckpt.clear()   # book fully segmented: checkpoints no longer needed
    return scenes


def scenes_to_records(file_code, scenes, book, metadata):
    """Stitch cross-chunk scenes and flatten them into flat, ingest-ready record dicts.

    One record == one future Qdrant point; enrichment fields (tone/summary/...) start
    null and are filled in by embed.py.
    """
    text_of, chapter_of = {}, {}
    for chunk in book.chunks:
        for p in chunk.paragraphs:
            text_of[p.index] = p.text
            chapter_of[p.index] = chunk.chapter_heading

    # stitch scenes across chunks
    merged = []
    for s in scenes:
        if s.open_start_index and merged and merged[-1]["_open_end"]:
            prev = merged[-1]
            prev["end_paragraph_index"] = s.end_paragraph_index
            prev["_open_end"] = s.open_end_index
            prev["status"] = "stitched"        # tail + head joined cleanly
        else:
            start = s.start_paragraph_index
            status = "complete"
            # connect open_start_index with open_end_index, otherwise connect 1 paragraph
            if s.open_start_index:
                start = max(0, start - 1)
                status = "broken_stitch"       # head with no matching tail
            merged.append({
                "start": start,
                "end_paragraph_index": s.end_paragraph_index,
                "title": s.title,
                "_open_end": s.open_end_index,
                "status": status,
            })

    # no open_start_index; mark as broken
    for m in merged:
        if m["_open_end"]:
            m["status"] = "broken_stitch"

    PARA_BREAK = "</p><p>"
    last = len(merged) - 1
    # metadata stays a set in memory; JSON/Qdrant can't hold a set, so serialize
    # a list-Subjects copy here (the sink) without mutating the caller's dict.
    book_metadata = MetadataParser.to_dict(metadata)
    author = book_metadata.get("Author")
    language = book_metadata.get("Language")
    records = []
    for i, m in enumerate(merged):
        start, end = m["start"], m["end_paragraph_index"]
        text = "<p>" + PARA_BREAK.join(text_of[j].strip() for j in range(start, end + 1) if j in text_of) + "</p>"
        word_count = len(re.sub(r"<[^>]+>", " ", text).split())
        # FLAT, TYPED, INGEST-READY: one record == one Qdrant point. Enrichment
        # (embed.py) fills the null flavor fields + summary, then denormalizes
        # prev_tone/next_tone. Vectors + the Qdrant {id,vector,payload} envelope
        # are added at upsert, NOT stored here (keep this file DB-agnostic).
        # neighbor pointers: scenes are contiguous + book-ordered, so prev/next
        # power the small-to-big fetch and the box-to-box sequence walk.
        # Start from the registry's null template (all enrichment/frame/transition fields
        # None, enriched False, schema_version stamped) so the record shape is defined ONCE
        # in scene_schema.json; here we only fill the fields SEGMENTATION knows. Enrichment
        # (embed.py) fills the nulls + summary, then denormalizes prev_tone/next_tone.
        rec = schema.blank_record()
        rec.update({
            "scene_id": f"{file_code}-{i}",
            "book_id": file_code,
            "prev_scene_id": f"{file_code}-{i-1}" if i > 0 else None,
            "next_scene_id": f"{file_code}-{i+1}" if i < last else None,
            "scene_title": m["title"],
            "chapter_title": chapter_of.get(start),
            "stitch_status": m["status"],   # complete | stitched | broken_stitch
            "start_paragraph_index": start,
            "end_paragraph_index": end,
            "word_count": word_count,
            "author": author,
            "language": language,
            "book_metadata": book_metadata,
            "text_html": text,              # render form; embed source is `summary`
        })
        records.append(rec)

    return records
