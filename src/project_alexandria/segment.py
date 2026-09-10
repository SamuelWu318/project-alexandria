import json, re, time, math, threading
from collections import Counter
from openai import pydantic_function_tool
from pydantic import BaseModel, ValidationError
from typing import Literal

from data import gate_facts
from utils import (read_json, write_json, MODEL, MODEL_PARAMS, CLIENT, PROCESS_PROMPT, PROCESS_CONTINUE_NOTE,
                   SPLIT_PROMPT, classify_llm_error, Checkpoint, log, schema, inject_retry_notes, SrcPaths)

# ---- Stage 2: segmentation (SPARSE boundary labels -> reconstruct dramatic-unit scenes) ----
# The LLM reads a whole chunk (Chunk.scene_payload: read-only context + indexed paragraphs) and returns
# only the BOUNDARY paragraphs — SCENE_START (a new dramatic unit begins here), SCENE_CONTINUE (at most
# one, the LAST scene marker: this section's final scene runs open past the section's end), and NOISE
# (apparatus, dropped). Unlabelled paragraphs are IMPLICITLY part of the currently open scene. Scenes,
# the cross-chunk stitch, and the noise drop all FALL OUT of the merged global stream: a start opens a
# scene, unlabelled paras fill it to the next start, NOISE is dropped and never breaks the open scene, an
# open tail is rejoined by the next section's unlabelled opening. A book's chunks are labelled
# SEQUENTIALLY (the DRIVER runs books in parallel), so each chunk is told via PROCESS_CONTINUE_NOTE when
# the previous one left a scene open (a SCENE_CONTINUE); the model then withholds the first SCENE_START so
# the halves stitch. Any scene past the word ceiling is then FLAGGED and re-split by the LLM into
# ceil(words / SOFT_MAX_WORDS) contiguous pieces (mechanical paragraph-split as the fallback). segment_book is
# the single door: it runs the pre-gate, labels every chunk (sequential + checkpointed), reconstructs, and
# returns flat ingest-ready records (enrichment fields still null). The prompt (PROCESS_PROMPT, in
# utils/llm.py) and the retry/temperature policy are the USER'S tuning surface — do not touch unless asked.

# ---- LLM boundary-classification schema (forced output_labels tool) ----

SOFT_MAX_WORDS = 1000   # mechanical safety valve: split a longer scene at a paragraph break (tunable; matches the prompt's 1000-word ceiling)


class ParagraphLabel(BaseModel):
    index: int                                                   # the paragraph's GLOBAL index (from indexed_paragraphs)
    label: Literal["SCENE_START", "SCENE_CONTINUE", "NOISE"]     # scene begins / open tail / apparatus

class ChunkLabels(BaseModel):
    # ParagraphLabel defined FIRST: this annotation is evaluated at class-definition time
    # (no `from __future__ import annotations`), so the forward name must already exist.
    labels: list[ParagraphLabel]

TOOL = pydantic_function_tool(
    ChunkLabels,
    name="output_labels",
    description="Return only the boundary paragraphs: noise, scene beginnings, and one optional trailing scene-continue.",
)
# NON-STRICT: providers that don't enforce JSON schema (e.g. MiniMax) get dropped by
# require_parameters:True when strict is set. Pydantic + the label-retry loop validate.
TOOL["function"]["strict"] = False

# MODEL_PARAMS centralizes routing/reasoning ONLY (no tool_choice); each stage forces its own tool at
# the call. This stage's tool is output_labels.
_TOOL_CHOICE = {"type": "function", "function": {"name": "output_labels"}}


# ---- sparse-label validation + retry-note assembly ----

# The indices the model may label: every indexed_paragraphs index (read-only context excluded).
def _expected_indices(payload: str) -> set[int]:
    obj = json.loads(payload)
    return {p["index"] for p in obj.get("indexed_paragraphs", [])}


# Validate the SPARSE labels -> (ok, reason for the model). Coverage is NOT required (unlabelled paras
# are implicit continuation); this guards against out-of-range / duplicate labels and enforces the
# SCENE_CONTINUE contract: at most one, and it is the LAST scene marker (no SCENE_START after it).
def _validate_labels(data: ChunkLabels, expected: set[int]):
    got = [lab.index for lab in data.labels]
    extra = sorted(set(got) - expected)
    dupes = sorted(i for i, n in Counter(got).items() if n > 1)
    continues = sorted(lab.index for lab in data.labels if lab.label == "SCENE_CONTINUE")
    starts = [lab.index for lab in data.labels if lab.label == "SCENE_START"]

    parts = []
    if extra:
        parts.append(f"labelled indices NOT in this section: {extra}")
    if dupes:
        parts.append(f"an index labelled more than once: {dupes}")
    if len(continues) > 1:
        parts.append(f"more than one SCENE_CONTINUE (allowed at most one): {continues}")
    if continues and any(s > continues[0] for s in starts):
        parts.append("SCENE_CONTINUE must be the LAST scene marker — no SCENE_START may follow it")

    return (not parts), "; ".join(parts)


# The segmentation retry-reminder SECTION (prompt's `#`-section style), or "" on the first attempt.
# The generic slot-[1] splice lives in utils.llm.inject_retry_notes; this is just the segmenter's wording.
def _retry_note(notes: list[str]) -> str:
    if not notes:
        return ""
    lines = "\n".join(f"- previous attempt error (NEVER DO THIS AGAIN): {n}" for n in notes)
    return ("# RETRY — MARK THE BOUNDARIES WHILE AVOIDING THESE ERRORS\n"
            "Earlier attempts on this section had these errors. FIX THIS: "
            f"\n{lines}\n")


# ---- SceneBreaker: one chunk -> paragraph labels, with the retry loop (retry/temperature policy is the user's) ----

class SceneBreaker:

    # ** MAIN ** — segment._label_book calls this once per chunk, in order
    # Label one chunk's boundaries via a forced output_labels call, retrying (fresh convo + note) until
    # the sparse labels validate (in-range, no dupes, one trailing continue); only a fatal API error
    # raises. `pending_continue` (the previous chunk left a scene open) splices PROCESS_CONTINUE_NOTE so
    # the model continues, not restarts, at the section's opening.
    def break_chunk(self, file_code: str, chunk: str, chunk_index, pending_continue: bool = False) -> ChunkLabels:
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
            system = inject_retry_notes(PROCESS_PROMPT, notes, _retry_note)   # splice the segmenter's retry wording
            if pending_continue:
                system += "\n" + PROCESS_CONTINUE_NOTE                        # tell it the prior section left a scene open
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": chunk},
            ]
            try:
                log.info(f"book {file_code}: sending chunk {chunk_index} to {MODEL}")
                response = CLIENT.chat.completions.create(
                    model=MODEL, temperature=temp, tools=[TOOL],
                    messages=messages,
                    tool_choice=_TOOL_CHOICE,   # force THIS stage's tool
                    **MODEL_PARAMS,             # routing + reasoning only (no tool_choice)
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


# ---- label a whole book (SEQUENTIAL per-chunk, checkpointed) ----

# Label a book's chunks IN ORDER (one forced LLM call each, per-chunk checkpointed) and merge into one
# global, SPARSE {paragraph index -> label} map (only boundary paragraphs appear). Sequential is what
# carries the cross-section handshake: after each chunk, `pending` = it emitted a SCENE_CONTINUE, and that
# flag is fed to the NEXT chunk so the model continues (not restarts) the carried-over scene. The DRIVER
# runs whole books in parallel; within a book, order matters, so this stays single-threaded.
def _label_book(book, checkpoint_base) -> dict:
    ckpt = Checkpoint(checkpoint_base, f"pg{book.file_code}",       # per-book resume cache (typed codec)
                      load=ChunkLabels.model_validate,
                      dump=lambda d: d.model_dump(mode="json"))
    sb = SceneBreaker()
    label_of, pending = {}, False
    for chunk in book.chunks:
        key = f"chunk-{chunk.chunk_index}"
        data = ckpt.load(key)                                       # resume: skip the LLM if cached
        if data is not None:
            log.info(f"book {book.file_code}: chunk {chunk.chunk_index} cached (skip LLM)")
        else:
            data = sb.break_chunk(book.file_code, chunk.scene_payload(), chunk.chunk_index, pending)   # forced LLM label
            ckpt.save(key, data)                                    # persist before continuing, so a crash survives
            log.info(f"book {book.file_code}: chunk {chunk.chunk_index} verified")
        for lab in data.labels:
            label_of[lab.index] = lab.label
        pending = any(lab.label == "SCENE_CONTINUE" for lab in data.labels)   # open tail -> next chunk continues it

    ckpt.clear()   # book fully labelled: checkpoints no longer needed
    return label_of


# ---- reconstruct scenes from the sparse label stream ----

# Walk ALL paragraph indices in reading order and group them into scenes from the SPARSE labels:
# SCENE_START / SCENE_CONTINUE open a scene, NOISE is dropped (never breaks the open scene), and every
# UNLABELLED paragraph joins the currently open scene (implicit continuation) — this is what stitches an
# open tail to the next section's opening. Each scene = {indices, broken}, where `broken` flags an
# unlabelled run before any scene has opened (a dangling head at the very start of the stream).
def _scenes_from_labels(order: list[int], label_of: dict) -> list[dict]:
    scenes: list[dict] = []
    cur = None
    for idx in order:
        label = label_of.get(idx)               # None = unlabelled = continuation of the open scene
        if label == "NOISE":
            continue
        if label in ("SCENE_START", "SCENE_CONTINUE"):
            cur = {"indices": [idx], "broken": False}
            scenes.append(cur)
        elif cur is None:                        # unlabelled with nothing open yet -> dangling head
            cur = {"indices": [idx], "broken": True}
            scenes.append(cur)
        else:                                    # unlabelled -> extend the open scene
            cur["indices"].append(idx)
    return scenes


# MECHANICAL FALLBACK for the LLM re-split (SceneSplitter below): split one scene's kept indices so each
# piece's word count stays <= SOFT_MAX_WORDS, cutting at the nearest paragraph break (a lone over-cap
# paragraph is kept whole, like data._pack). Used only when the model cannot deliver a valid split.
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


# ---- oversize re-split: an over-cap scene -> N contiguous pieces (LLM, mechanical _cap_split fallback) ----

SPLIT_MAX_ATTEMPTS = 6   # total tries before falling back to mechanical _cap_split (retry policy is the user's)


# LLM split schema (forced output_splits tool): the first paragraph index of each piece.
class SplitPoints(BaseModel):
    piece_starts: list[int]   # GLOBAL paragraph index that begins each piece, reading order; first == scene start

SPLIT_TOOL = pydantic_function_tool(
    SplitPoints,
    name="output_splits",
    description="Return the first paragraph index of each contiguous piece the over-long scene is cut into.",
)
SPLIT_TOOL["function"]["strict"] = False   # non-strict, like TOOL above (providers drop strict under require_parameters)

_SPLIT_TOOL_CHOICE = {"type": "function", "function": {"name": "output_splits"}}


# Build the split call's user JSON: the target piece count + the scene's paragraphs in reading order.
def _split_payload(indices: list[int], n_pieces: int, text_of: dict) -> str:
    return json.dumps({
        "n_pieces": n_pieces,
        "indexed_paragraphs": [{"index": i, "text": text_of[i]} for i in indices],
    })


# Validate a split -> (ok, reason for the model): EXACTLY n starts, all inside the scene, strictly ascending,
# and the first is the scene's first paragraph (so the pieces partition the whole scene from the top).
def _validate_splits(data: SplitPoints, scene_indices: list[int], n: int):
    starts = data.piece_starts
    allowed = set(scene_indices)
    parts = []
    if len(starts) != n:
        parts.append(f"need EXACTLY {n} piece-start indices, got {len(starts)}")
    extra = sorted({s for s in starts if s not in allowed})
    if extra:
        parts.append(f"start indices not in this scene: {extra}")
    if any(b <= a for a, b in zip(starts, starts[1:])):
        parts.append("piece starts must be STRICTLY ascending (each piece begins later than the last)")
    if starts and starts[0] != scene_indices[0]:
        parts.append(f"the first piece must start at the scene's first paragraph {scene_indices[0]}")
    return (not parts), "; ".join(parts)


# The re-split retry-reminder SECTION (prompt slot-[1] splice), or "" on the first attempt.
def _split_retry_note(notes: list[str]) -> str:
    if not notes:
        return ""
    lines = "\n".join(f"- previous attempt error (NEVER DO THIS AGAIN): {n}" for n in notes)
    return ("# RETRY — CUT THIS SCENE WHILE AVOIDING THESE ERRORS\n"
            "Earlier attempts to split this scene had these errors. FIX THIS:\n"
            f"{lines}\n")


# Slice the scene's contiguous index list at the returned piece-start indices -> a list of index groups.
# starts are validated (ascending, in-scene, first == scene start), so every group is non-empty.
def _groups_from_starts(scene_indices: list[int], starts: list[int]) -> list[list[int]]:
    pos = {idx: k for k, idx in enumerate(scene_indices)}
    cuts = [pos[s] for s in starts]
    return [scene_indices[a:b] for a, b in zip(cuts, cuts[1:] + [len(scene_indices)])]


class SceneSplitter:

    # ** MAIN ** — segment._resplit_oversize calls this once per flagged (over-cap) scene
    # Cut ONE over-long scene into EXACTLY n_pieces contiguous pieces via a forced output_splits call,
    # retrying (fresh convo + replayed misses + climbing temp) until the split validates. BOUNDED: on
    # SPLIT_MAX_ATTEMPTS exhaustion or a fatal API error it falls back to the mechanical _cap_split, so a
    # book is never wedged by a stubborn scene. Returns index groups in reading order.
    def split_scene(self, file_code: str, indices: list[int], n_pieces: int,
                    word_of: dict, text_of: dict) -> list[list[int]]:
        if len(indices) < 2:                        # a lone paragraph cannot be cut
            return _cap_split(indices, word_of)
        n_pieces = min(n_pieces, len(indices))      # cannot make more pieces than there are paragraphs
        payload = _split_payload(indices, n_pieces, text_of)
        notes = []                                  # split misses from earlier attempts, replayed in the note

        for attempt in range(SPLIT_MAX_ATTEMPTS):
            temp = 0 if attempt == 0 else min(0.75, math.log(attempt ** 0.20) + 0.15)
            system = inject_retry_notes(SPLIT_PROMPT, notes, _split_retry_note)   # splice the splitter's retry wording
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": payload},
            ]
            try:
                log.info(f"book {file_code}: splitting scene {indices[0]}..{indices[-1]} into {n_pieces} (try {attempt + 1})")
                response = CLIENT.chat.completions.create(
                    model=MODEL, temperature=temp, tools=[SPLIT_TOOL],
                    messages=messages,
                    tool_choice=_SPLIT_TOOL_CHOICE,   # force THIS stage's tool
                    **MODEL_PARAMS,                   # routing + reasoning only (no tool_choice)
                )
            except Exception as e:
                if classify_llm_error(e) == "fatal":                  # non-retryable 4xx -> stop, fall back
                    log.warn(f"book {file_code}: split fatal ({e}) — mechanical fallback")
                    break
                sleep = min(2 ** (attempt + 1), 30)
                log.warn(f"book {file_code}: split transient retry (sleep {sleep}s): {e}")
                time.sleep(sleep)
                continue

            # inspect the response: no tool_call / bad args / invalid split -> replay as a note next attempt
            choices = response.choices
            msg = choices[0].message if choices else None
            if not msg or not msg.tool_calls:
                reason = "did not call the output_splits tool"
            else:
                args = msg.tool_calls[0].function.arguments
                try:
                    data = SplitPoints.model_validate_json(args)      # schema-validate the tool args
                except (ValidationError, json.JSONDecodeError, ValueError) as e:
                    reason = f"arguments failed schema validation: {e}"
                else:
                    ok, why = _validate_splits(data, indices, n_pieces)
                    if ok:
                        return _groups_from_starts(indices, data.piece_starts)
                    reason = why
            notes.append(reason)
            if len(notes) > 4:
                notes.pop(0)
            log.warn(f"book {file_code}: split validation retry {attempt + 1}: {reason[:140]}")

        return _cap_split(indices, word_of)   # LLM split unavailable/invalid -> mechanical safety valve


# Flag each reconstructed scene that overshot the word ceiling: scene["oversize"] = the number of pieces to
# cut it into (ceil(words / SOFT_MAX_WORDS), always >= 2), else 0. This is the word->piece rule the model is
# later TOLD (2x cap -> 2, 4x cap -> 4, ...). _resplit_oversize scans this flag.
def _flag_oversize(scenes: list[dict], word_of: dict):
    for scene in scenes:
        words = sum(word_of[i] for i in scene["indices"])
        scene["oversize"] = math.ceil(words / SOFT_MAX_WORDS) if words > SOFT_MAX_WORDS else 0


# Scan the flagged scenes and re-split each over-cap one (via the LLM) into its flagged piece count — the
# semantic replacement for the old blind paragraph-split. A within-cap scene passes straight through; a
# flagged scene becomes its N pieces (the `broken` head rides the FIRST piece only). SceneSplitter falls
# back to _cap_split when the LLM cannot deliver a valid split, so this always returns splittable scenes.
def _resplit_oversize(scenes: list[dict], word_of: dict, text_of: dict, file_code: str) -> list[dict]:
    splitter = SceneSplitter()
    out: list[dict] = []
    for scene in scenes:
        n = scene.get("oversize") or 0
        if n < 2:
            out.append(scene)
            continue
        groups = splitter.split_scene(file_code, scene["indices"], n, word_of, text_of)   # LLM cut (fallback inside)
        for k, idxs in enumerate(groups):
            out.append({"indices": idxs, "broken": scene["broken"] and k == 0})
    return out


# Derive one scene piece's stitch status from its chunk span: a broken head -> "broken_stitch"; a piece
# spanning >1 chunk -> "stitched" (its unlabelled fill crossed a chunk boundary); otherwise "complete".
def _stitch_status(indices: list[int], chunk_of: dict, broken_head: bool) -> str:
    if broken_head:
        return "broken_stitch"
    return "stitched" if len({chunk_of[i] for i in indices}) > 1 else "complete"


# ** MAIN ** — segment_book flattens a book's labels into ingest-ready records here
# Reconstruct scenes from the label stream, flag + LLM-re-split any over-cap scene (mechanical fallback),
# and flatten into flat records (one record == one future Qdrant point; enrichment fields start null).
# Every record starts from schema.blank_record() so the shape is defined ONCE in scene_schema.json; only
# what SEGMENTATION knows is filled. Vectors + the Qdrant envelope are added later — keep this DB-agnostic.
def _build_records(book, metadata: dict, label_of: dict) -> list[dict]:
    code = book.file_code
    text_of, chapter_of, chunk_of, word_of = {}, {}, {}, {}
    for chunk in book.chunks:
        for p in chunk.paragraphs:
            text_of[p.index] = p.text
            chapter_of[p.index] = chunk.chapter_heading
            chunk_of[p.index] = chunk.chunk_index
            word_of[p.index] = len(re.sub(r"<[^>]+>", " ", p.text).split())

    # labels -> scenes -> flag the over-cap ones -> LLM re-split them; each resulting piece becomes one record
    order = sorted(word_of)                     # every paragraph index, reading order (fills the sparse labels)
    scenes = _scenes_from_labels(order, label_of)               # sparse labels -> dramatic-unit scene groups
    _flag_oversize(scenes, word_of)                             # mark any scene past the word ceiling for re-split
    scenes = _resplit_oversize(scenes, word_of, text_of, code)  # LLM cut of the flagged scenes (fallback: _cap_split)
    pieces = [(scene["indices"], scene["broken"]) for scene in scenes]   # broken head already on the first piece

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
_EXCLUDE_LOCK = threading.Lock()                               # books segment in parallel; this file is shared


# Append one rejected book to the running excluded-books json (auditable: reason + matched detail).
# Locked: the driver segments books in parallel and this one file is shared (read-modify-write).
def _log_exclusion(exclude_dir, code, metadata: dict, reason, detail=None):
    path = f"{exclude_dir}/{EXCLUDED_BOOKS_FILE}"
    with _EXCLUDE_LOCK:
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

# ** MAIN ** — tests.segment_test segments each book through this single door (the driver runs books in parallel)
# Segment ONE whole book end to end: run the pre-gate, label its chunks in order (sequential +
# checkpointed, threading the continue-flag), reconstruct dramatic-unit scenes from the labels, flag +
# LLM-re-split any over-cap scene, and return flat ingest-ready records (an empty list if the book is
# gated out). Keep DB-agnostic.
def segment_book(book, md: dict, checkpoint_base=SrcPaths.CHECKPOINT_DIR,
                 data_path=SrcPaths.DATA_DIR, exclude_dir=SrcPaths.RECALL_DIR) -> list[dict]:
    facts = gate_facts(book.file_code, md, data_path)      # data: rights + subjects + serialized metadata (one door)
    if _presegmentation_gate(book.file_code, facts, exclude_dir):
        return []                                          # excluded: no records

    label_of = _label_book(book, checkpoint_base)                   # sequential forced LLM label per chunk -> {index: label}
    return _build_records(book, facts["metadata"], label_of)        # labels -> scenes -> flat records
