from data import MetadataParser, SceneParser
import os, json, re, time, math, threading, shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import openai
from openai import OpenAI, pydantic_function_tool
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from typing import Literal
from pathlib import Path
load_dotenv()


# --- metadata parser constants --- #

CATALOG_PATH = "pg_catalog.csv"
DATA_PATH = "data"
SCENES_PATH = "scenes"
CHECKPOINT_DIR = "checkpoints"
# two consecutive chunks share one LLM call when their combined payload
# (chars) stays under this ceiling. tunable knob; raise for more batching.
BATCH_CHAR_LIMIT = 45000

# --- model constants --- #

MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
SYSTEM_PROMPT = """
    # ROLE
    You segment one chunk of a book chapter into either: 
    scenes of prose/dialogue, or: noise from leftover html/licenses/footnotes/chapter titles/non-prose/headers.
    You label paragraphs by their index only, never rewriting or outputting the text directly.

    # INPUT
    You will receive EITHER one JSON object (a single section), OR a JSON array of two such objects (two sections batched into one call). Each section object has:
    - "chapter_title": the chapter the current section belongs to. use it as context for determining scenes/noise.
    - "section_within_chunk": "SECTION/TOTAL". SECTION is the 1-based index of the section within the chapter. (EX: 1/5 is FIRST SECTION, 5/5 is LAST SECTION) Use this to determine if some scenes were cut off during chunking.
    - "read_only_context_paragraphs": paragraphs from the PREVIOUS SECTION. Use it for additional context. They are never included in the output.
    - "number_of_indexed_paragraphs": number of paragraphs that MUST be segmented.
    - "indexed_paragraphs": the paragraphs themselves that MUST be segmented. Each is in form {"index": int, "text": str}. Ignore inline HTML, reason only about the words.

    When the input is an array of two sections, segment BOTH. Paragraph indices are globally unique across sections (they never repeat), so return ONE flat scenes_data list covering every indexed paragraph from every section, in ascending index order. Two adjacent sections may hold a single scene that spans the boundary between them: because their paragraphs are contiguous, emit it as ONE scene instead of splitting it with open flags. open_start_index / open_end_index still refer only to continuation BEYOND all paragraphs present in this call (the FIRST section's read_only_context on the start side, the LAST section's final paragraph on the end side).

    # TASK
    Return, via the output_scenes tool, an ordered list of segmented scenes covering every paragraph in "indexed_paragraphs".

    # OUTPUT 
    - start_paragraph_index: inclusive start index range for a scene from "indexed_paragraphs".
    - end_paragraph_index: inclusive end index range for a scene from "indexed_paragraphs".
    - paragraph_type: "scene" contains story prose with one central focus. "noise" contains anything not related to the story: footnotes, captions, table-of-contents, licenses, HEADERS, etc.
    - open_start_index: True ONLY if segment is FIRST "SCENE" segment, ONLY when TOTAL SECTIONS > 1, and ONLY if part of the scene is contained within "read_only_context_paragraphs". Otherwise, False.
    - open_end_index: True ONLY if segment is LAST "SCENE" segment, ONLY when TOTAL SECTIONS > 1, and ONLY if the scene obviously continues past the last paragraph. Otherwise, False.
    - title: a short, 4-10 word title for the scene. if noise, label it as "NOISE". 

    # RULES
    - only segment "indexed_paragraphs".
    - cover every index in "indexed_paragraphs" exactly once, in ascending order, no gaps, no overlaps. The first scene/noise segment begins at the smallest index; the last scene/noise segment the largest.
    - segments in sections that are not the first or last section will always have open_start_index and open_end_index set to False.
    - segments that are considered noise will always have open_start_index and open_end_index set to False.
    - tool-call output_scenes, and output nothing else.
    - attempt to keep scenes between 200-600 words. Do not feel restricted if a scene goes on longer or ends shorter, use best judgement to determine scene cut-off points.

    # EXAMPLE 1
    -- input --
    {
    "chapter_title": "A Mad Tea-Party",
    "section_within_chunk": "1/1",
    "read_only_context_paragraphs": [],
    "indexed_paragraphs": [
    { "index": 0, "text": "There was a table set out under a tree, and the March Hare and the Hatter were having tea."},
    { "index": 1, "text": "Alice sat down uninvited. The Hatter asked a riddle with no answer."},
    { "index": 2, "text": "They moved round the table, and Alice, quite exhausted, walked off into the wood."},
    { "index": 3, "text": "[7] 'Mad as a hatter' predates Carroll; hatters were poisoned by mercury."}
    ]
    }
    -- output_scenes --
    {"scenes_data": [
        {"start_paragraph_index": 0, "end_paragraph_index": 2, "paragraph_type": "scene", "open_start_index": False, "open_end_index": False, "title": "Tea with the Hatter and Hare"},
        {"start_paragraph_index": 3, "end_paragraph_index": 3, "paragraph_type": "noise", "open_start_index": False, "open_end_index": False, "title": "NOISE"}
    ]}

    Reasoning: section is 1/1, so this is one whole chapter with no broken scenes. The paragraph at index 3 is noise. Looking at the indices, 0-3 indices are covered exactly once.

    # EXAMPLE 2
    -- input --
    {
    "chapter_title": "BOOK IV",
    "section_within_chunk": "2/3",
    "read_only_context_paragraphs": [
    { "index": 7, "text": "Helen told of Troy while the hall listened."},
    { "index": 8, "text": "Then Menelaus began the long tale of his voyage home, and as night fell he spoke of the sea-god Proteus."}
    ],
    "indexed_paragraphs": [
    { "index": 9, "text": "The sea-god was the Old Man of the Sea, whom he wrestled at dawn to force the truth from him."},
    { "index": 10, "text": "The tale done, Telemachus rose to take his leave."},
    { "index": 11, "text": "\"Brothers!\" Telemachus announced."}
    ]
    }
    -- output_scenes --
    {"scenes_data":[
        {"start_paragraph_index": 9, "end_paragraph_index": 10, "paragraph_type": "scene", "open_start_index": True, "open_end_index": False, "title": "Menelaus finishes his tale"},
        {"start_paragraph_index": 11, "end_paragraph_index": 11, "paragraph_type": "scene", "open_start_index": False, "open_end_index": True, "title": "Telemachus speaks to his brothers"},
    ]}

    Reasoning: section is 2/3, so both open_start_index and open_end_index can be True. Using the read_only_context_paragraphs, we can see the first scene extends backward, so we set open_start_index to be True.
    Looking at the last scene available, it is obvious that the scene continues past that point, so we set open_end_index to be True. We also only use indices 9-11, as 7-8 are read_only_context_paragraphs.

    # EXAMPLE 3 (two sections batched in one call)
    -- input --
    [
      {
        "chapter_title": "Chapter 5",
        "section_within_chunk": "1/2",
        "read_only_context_paragraphs": [],
        "indexed_paragraphs": [
          { "index": 20, "text": "Rain hammered the inn roof as the traveler stamped in from the dark."},
          { "index": 21, "text": "He took the corner seat and called for wine, watching the door."}
        ]
      },
      {
        "chapter_title": "Chapter 5",
        "section_within_chunk": "2/2",
        "read_only_context_paragraphs": [
          { "index": 21, "text": "He took the corner seat and called for wine, watching the door."}
        ],
        "indexed_paragraphs": [
          { "index": 22, "text": "When it opened, the woman he had followed for a year stepped through."},
          { "index": 23, "text": "[3] The inn still stands today, rebuilt after the fire of 1812."}
        ]
      }
    ]
    -- output_scenes --
    {"scenes_data":[
        {"start_paragraph_index": 20, "end_paragraph_index": 22, "paragraph_type": "scene", "open_start_index": False, "open_end_index": False, "title": "The traveler waits at the inn"},
        {"start_paragraph_index": 23, "end_paragraph_index": 23, "paragraph_type": "noise", "open_start_index": False, "open_end_index": False, "title": "NOISE"}
    ]}

    Reasoning: two sections of the same chapter arrive together. Indices 20-23 are globally unique and contiguous, so they are covered exactly once, in ascending order, by ONE flat list. The scene that opens at 20 continues across the section boundary (21 into 22) into the second section, so it is emitted as ONE scene (20-22), NOT split with open flags. Index 23 is a footnote = noise. Although the sections are labeled 1/2 and 2/2, both are present in this call so nothing continues beyond the batch: every open flag is False.
    """

class MultiSceneData(BaseModel):
    scenes_data: list[SceneData]

class SceneData(BaseModel):
    # metadata can be added later.
    start_paragraph_index: int
    end_paragraph_index: int
    paragraph_type: Literal["scene", "noise"]
    open_start_index: bool
    open_end_index: bool
    title: str
    
TOOL = pydantic_function_tool(
    MultiSceneData,
    name="output_scenes",
    description="Force return of scenes in structure."
)
CLIENT = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_KEY"],
)
with open("test/test.json", "r") as f:
    TEST_CHUNK = json.dumps(json.load(f), ensure_ascii=False)


def _classify_error(e: Exception) -> str:
    # "transient": network/429/5xx -> worth retrying with backoff.
    # "fatal": 4xx (400/401/403 etc.) -> retrying won't help, raise now.
    if isinstance(e, (openai.APIConnectionError, openai.APITimeoutError)):
        return "transient"
    if isinstance(e, openai.APIStatusError):
        return "transient" if (e.status_code == 429 or e.status_code >= 500) else "fatal"
    return "transient"  # unknown network-ish -> limited retry


def _expected_indices(payload: str) -> set[int]:
    # every index in indexed_paragraphs across all sections in the payload.
    # (read_only_context_paragraphs are NOT expected in the output.)
    obj = json.loads(payload)
    sections = obj if isinstance(obj, list) else [obj]
    return {p["index"] for s in sections for p in s.get("indexed_paragraphs", [])}


def _validate_coverage(data: MultiSceneData, expected: set[int]):
    # runs BEFORE noise extraction, so EVERY expected index must be covered
    # exactly once. returns (ok, human-readable reason for the model).
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


class SceneBreaker:
    def break_chunk(self, chunk: str, max_transient_retries: int = 6,
                    max_validation_retries: int = 3):
        expected = _expected_indices(chunk)
        # conversation is kept across retries so corrective feedback (below) is
        # appended to real history instead of restarting cold each attempt.
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": chunk},
        ]
        transient_tries = 0
        validation_tries = 0
        attempt = 0  # drives temp bump across all retries

        while True:
            # temp 0, increase if attempts fail
            temp = 0 if attempt == 0 else math.log(attempt ** 0.15) + 0.15
            try:
                print("Sending chunk...")
                response = CLIENT.chat.completions.create(
                    model=MODEL, temperature=temp, tools=[TOOL],
                    messages=messages,
                    tool_choice={"type": "function", "function": {"name": "output_scenes"}},
                    extra_body={"provider":{"require_parameters":True},
                                "reasoning": {"effort": "high"}}
                )
            except Exception as e:
                # only API/network failures land here; parse/coverage handled below
                if _classify_error(e) == "fatal":
                    raise RuntimeError(f"break_chunk fatal (no retry): {e}") from e
                transient_tries += 1
                if transient_tries > max_transient_retries:
                    raise RuntimeError(
                        f"break_chunk transient failure after "
                        f"{max_transient_retries} retries: {e}") from e
                sleep = min(2 ** transient_tries, 30)
                print(f"  transient retry {transient_tries}/{max_transient_retries} "
                      f"(sleep {sleep}s): {e}")
                time.sleep(sleep)
                attempt += 1
                continue

            # inspect the response: no tool_call -> bad args -> incomplete coverage
            choices = response.choices
            msg = choices[0].message if choices else None
            echo = None       # what the model produced, echoed back into history
            correction = None  # what it did wrong, told to the model

            if not msg or not msg.tool_calls:
                echo = (msg.content if msg else "") or "(empty response, no tool call)"
                correction = ("You did NOT call the output_scenes tool. You must respond "
                              "ONLY with a call to output_scenes and nothing else.")
            else:
                args = msg.tool_calls[0].function.arguments
                echo = args
                try:
                    data = MultiSceneData.model_validate_json(args)
                except (ValidationError, json.JSONDecodeError, ValueError) as e:
                    correction = (f"Your output_scenes arguments failed schema validation: {e}. "
                                  f"Return arguments that exactly match the schema.")
                else:
                    ok, why = _validate_coverage(data, expected)
                    if ok:
                        return data
                    correction = (f"Your segmentation did not cover the paragraphs correctly. {why}. "
                                  f"Every index in indexed_paragraphs must be covered EXACTLY once, "
                                  f"in ascending order, no gaps and no overlaps. Redo the full segmentation.")

            # a correction is set: append to history and retry with the feedback
            validation_tries += 1
            if validation_tries > max_validation_retries:
                raise RuntimeError(
                    f"break_chunk validation failure after "
                    f"{max_validation_retries} retries: {correction}")
            print(f"  validation retry {validation_tries}/{max_validation_retries}: {correction[:140]}")
            messages.append({"role": "assistant", "content": str(echo)})
            messages.append({"role": "user", "content": correction})
            attempt += 1

def _load_checkpoint(path: Path):
    # returns cached MultiSceneData, or None if missing/corrupt (recompute).
    if not path.exists():
        return None
    try:
        return MultiSceneData.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_checkpoint(path: Path, data: MultiSceneData):
    # atomic: write temp then rename, so a crash mid-write can't leave partial json.
    tmp = path.with_suffix(".tmp")
    tmp.write_text(data.model_dump_json(), encoding="utf-8")
    os.replace(tmp, path)


def build_batches(chunks, char_limit: int = BATCH_CHAR_LIMIT):
    # greedily pair CONSECUTIVE chunks into one LLM call when their combined
    # payload stays under char_limit; otherwise a chunk goes alone. Deterministic
    # for a given book, so checkpoint keys stay stable across runs.
    batches = []
    i = 0
    while i < len(chunks):
        a = chunks[i]
        if i + 1 < len(chunks):
            b = chunks[i + 1]
            if len(a.scene_payload()) + len(b.scene_payload()) <= char_limit:
                batches.append((a, b))
                i += 2
                continue
        batches.append((a,))
        i += 1
    return batches


def _batch_payload(batch) -> str:
    # single chunk -> bare section object (matches EXAMPLES 1-2).
    # two chunks   -> JSON array of two sections (EXAMPLE 3).
    sections = [json.loads(c.scene_payload()) for c in batch]
    if len(sections) == 1:
        return json.dumps(sections[0], ensure_ascii=False)
    return json.dumps(sections, ensure_ascii=False)


def parsers():
    path = Path(DATA_PATH)
    
    metadata = {}
    mdp = MetadataParser()

    # search all files in path and create scenes
    books = {}
    sp = SceneParser()
    mdp = MetadataParser()

    for file in sorted(path.iterdir()):
        if not file.is_file(): continue

        match = re.search(r"pg(\d+)-h.zip", file.name)
        if not match: continue

        file_code = match.group(1)
        md = mdp.feed(file_code)
        metadata[file_code] = md
        book = sp.parse(file_code, str(path), md.get("Title"))

        books[book.file_code] = book
        # send to llm by doing books[file_code].chunks[chunk_num].scene_payload()

    return metadata, books


def scenes_to_records(file_code, scenes, book, metadata):
    # get all text at all indices and their chapter title
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

    PARA_BREAK = "\n\n"
    records = []
    for i, m in enumerate(merged):
        start, end = m["start"], m["end_paragraph_index"]
        text = PARA_BREAK.join(
            text_of[j].strip() for j in range(start, end + 1) if j in text_of
        )
        records.append({
            "scene_id": f"{file_code}-{i}",
            "chapter_title": chapter_of.get(start),
            "scene_title": m["title"],
            "stitch_status": m["status"],   # complete | stitched | broken_stitch
            "start_paragraph_index": start,
            "end_paragraph_index": end,
            "text": text,
            "book_metadata": metadata,
            "summary": None,   # second phase LLM fills later
            "tags": [],        # second phase LLM fills later
        })

    return records

def main():
    metadata, books = parsers()
    sb = SceneBreaker()

    desired_book = []
    desired = "11"
    book = books[desired]
    print_lock = threading.Lock()

    ckpt_dir = Path(CHECKPOINT_DIR) / f"pg{desired}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def work(batch):
        # one batch = 1 or 2 consecutive chunks answered by one LLM call
        key = "_".join(str(c.chunk_index) for c in batch)
        label = f"CHUNK {key}" if len(batch) == 1 else f"CHUNKS {key}"
        cpath = ckpt_dir / f"chunk-{key}.json"

        # resume: skip batches already segmented in a prior run
        cached = _load_checkpoint(cpath)
        if cached is not None:
            with print_lock:
                print(f"**** {label} CACHED (skip LLM) ****")
            return cached

        data = sb.break_chunk(_batch_payload(batch))
        _save_checkpoint(cpath, data)  # persist before printing, so it survives a crash

        # print as each batch finishes, real-time
        with print_lock:
            print(f"**** {label} VERIFIED ****")
            for scene in data.scenes_data:
                if scene.paragraph_type == "noise" or scene.title == "NOISE":
                    print(f"scene from {label} marked as noise")
                    continue
                print(f"scene from {label} passed")
                print(f"scene open? {scene.open_end_index} on end, {scene.open_start_index} on start.")
        return data

    batches = build_batches(book.chunks)

    # up to 6 concurrent LLM calls; results ordered by batch (book) position
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(work, batches))

    # ordered scene collection (silent). count paragraphs dropped as noise.
    kept_paras, noise_paras = 0, 0
    for data in results:
        for scene in data.scenes_data:
            span = scene.end_paragraph_index - scene.start_paragraph_index + 1
            if scene.paragraph_type == "noise" or scene.title == "NOISE":
                noise_paras += span
                continue
            kept_paras += span
            desired_book.append(scene)

    print(f"NOISE: dropped {noise_paras} paragraphs as noise; kept {kept_paras} "
          f"({noise_paras + kept_paras} total covered)")

    records = scenes_to_records(desired, desired_book, books[desired], metadata[desired])

    out_path = Path(f"{SCENES_PATH}/pg{desired}-s.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(records)} scenes to {out_path}")

    # book fully saved: drop its checkpoints, no longer needed for resume
    shutil.rmtree(ckpt_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
