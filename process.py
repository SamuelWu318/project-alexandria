from data import MetadataParser, SceneParser
import os, json, re, time, math, threading, shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import openai
from openai import OpenAI, pydantic_function_tool
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, Field, field_validator
from typing import Literal
from enum import Enum
from pathlib import Path
load_dotenv()


# --- metadata parser constants --- #

CATALOG_PATH = "pg_catalog.csv"
DATA_PATH = "data"
SCENES_PATH = "scenes"
CHECKPOINT_DIR = "checkpoints"

# --- model constants --- #

MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
SYSTEM_PROMPT = """
    # ROLE
    You split one book section into an ordered list of segments. Each segment is either:
    - "scene": story prose or dialogue.
    - "noise": non-story text — leftover HTML, licenses, footnotes, captions, table-of-contents, chapter titles, headers.
    Label paragraphs by index ONLY. Never rewrite or output the paragraph text.

    # INPUT
    You receive one JSON object (one section). It has:
    - "chapter_title": the chapter this section belongs to. Context for judging scene vs noise.
    - "section_within_chunk": "N/TOTAL" — this section's 1-based position in the chapter (1/5 = first, 5/5 = last). Use it to tell whether a scene was cut off at a section edge.
    - "read_only_context_paragraphs": paragraphs from the PREVIOUS section, for context only. Never segment or output them.
    - "number_of_indexed_paragraphs": how many paragraphs you must segment.
    - "indexed_paragraphs": the paragraphs to segment, each {"index": int, "text": str}. Ignore inline HTML; reason only about the words.

    The open flags are ONLY for beyond this section — before the first index, after the last.

    # TASK
    Call output_scenes with an ordered list of segments that covers every paragraph in "indexed_paragraphs" exactly once — ascending, no gaps, no overlaps.

    # OUTPUT (per segment)
    - start_paragraph_index / end_paragraph_index: inclusive index range, drawn from "indexed_paragraphs".
    - paragraph_type: "scene" or "noise".
    - title: 4-10 words naming the scene; "NOISE" for noise.
    - open_start_index: True only for the FIRST scene, and only when its opening lies in "read_only_context_paragraphs" (the scene began in an earlier section). Otherwise False.
    - open_end_index: True only for the LAST scene, and only when it clearly continues past the final indexed paragraph (into a later section). Otherwise False.
    - Noise segments and interior scenes (any scene that is neither first nor last) always have both flags False.

    # HOW TO CUT A SCENE
    A scene is ONE flavor: a single dominant tone/feeling, held from first line to last. TONAL PURITY IS THE HIGHEST PRIORITY — A scene never holds two feelings. The moment the dominant tone shifts, the scene ENDS and a new one begins.
    - PRIMARY seam 1: cut the instant the tone/feeling shifts drastically. This outranks every other consideration. 
    - PRIMARY seam 2: Treat size as equally important as tone; overly long scenes often hide two tones, overly short scenes cannot contain full tonal flavor. Aim for 300-600 words; ABSOLUTE floor ~250, ABSOLUTE ceiling ~800 words. 
    - SECONDARY seam: only when the tone holds steady across a long size do you cut where pov, setting, time, or the active conversation changes.

    # RULES
    - Segment only "indexed_paragraphs". The first segment starts at the smallest index; the last ends at the largest.
    - Call output_scenes and output nothing else.
    - ALWAYS first search for NOISE, as removing NOISE is the most important part.

    # EXAMPLE 1 (text is replaced with basic overview for example only)
        -- input --
        {
        "chapter_title": "A Dreadful Chapter",
        "section_within_chunk": "1/1",
        "read_only_context_paragraphs": [],
        "indexed_paragraphs": [
        { "index": 0, "text": "paragraph about building dread"},
        { "index": 1, "text": "paragraph about dread tension"},
        { "index": 2, "text": "paragraph about dread being realized"},
        { "index": 3, "text": "****** TABLE OF CONTENTS ***** Footnote: blah blah blah"}
        ]
        }
        -- output_scenes --
        {"scenes_data": [
            {"start_paragraph_index": 0, "end_paragraph_index": 2, "paragraph_type": "scene", "open_start_index": False, "open_end_index": False, "title": "title about dread"},
            {"start_paragraph_index": 3, "end_paragraph_index": 3, "paragraph_type": "noise", "open_start_index": False, "open_end_index": False, "title": "NOISE"}
        ]}
        Reasoning: section is 1/1 — one whole chapter, no cut-off scenes. Index 3 is a footnote = noise. Indices 0-3 are each covered once.
    
        # EXAMPLE 2 (text is replaced with basic overview for example only)
        -- input --
        {
        "chapter_title": "BOOK IV",
        "section_within_chunk": "2/3",
        "read_only_context_paragraphs": [
        { "index": 7, "text": "paragraph about the beginning of a story"},
        { "index": 8, "text": "paragraph about the middle of a story"}
        ],
        "indexed_paragraphs": [
        { "index": 9, "text": "paragraph about the end of a story"},
        { "index": 10, "text": "paragraph about someone leaving the room."},
        { "index": 11, "text": "paragraph about the beginning of a conversation."}
        ]
        }
        -- output_scenes --
        {"scenes_data":[
            {"start_paragraph_index": 9, "end_paragraph_index": 10, "paragraph_type": "scene", "open_start_index": True, "open_end_index": False, "title": "title about the story"},
            {"start_paragraph_index": 11, "end_paragraph_index": 11, "paragraph_type": "scene", "open_start_index": False, "open_end_index": True, "title": "title about the conversation"}
        ]}
        Reasoning: section is 2/3. The first scene's opening lies in read_only_context_paragraphs, so open_start_index is True. Lump in index 10 because it is transitional, doesn't hurt. The last scene clearly continues past index 11, so open_end_index is True. Only indices 9-11 are segmented; 7-8 are context.
        """
"""
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
    Reasoning: section is 1/1 — one whole chapter, no cut-off scenes. Index 3 is a footnote = noise. Indices 0-3 are each covered once.

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
        {"start_paragraph_index": 11, "end_paragraph_index": 11, "paragraph_type": "scene", "open_start_index": False, "open_end_index": True, "title": "Telemachus speaks to his brothers"}
    ]}
    Reasoning: section is 2/3. The first scene's opening lies in read_only_context (paras 7-8), so open_start_index is True. The last scene clearly continues past index 11, so open_end_index is True. Only indices 9-11 are segmented; 7-8 are context.
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


# ---- (enrichment) tags -----------------------------------------
# Rigid flavor tags for the goal: fetch scenes by emotional FLAVOR, then inject
# that flavor into the user's own prose. A scene is ONE tone (see HOW TO CUT), so
# these tags describe exactly ONE dominant feeling. The enrichment LLM call fills
# a SceneTags on the fully-stitched scene text; a record holds None until then.

class Tone(str, Enum):
    # CONTROLLED VOCABULARY — the single dominant feeling of a scene. This is the
    # rigid facet Mode-1 search filters on and the pair Mode-2 transitions are
    # built from. Edit this palette to retune search; keep values lowercase.
    DREAD = "dread"
    TERROR = "terror"
    MENACE = "menace"
    SUSPENSE = "suspense"
    UNEASE = "unease"
    GRIEF = "grief"
    MELANCHOLY = "melancholy"
    DESPAIR = "despair"
    LONELINESS = "loneliness"
    TENDERNESS = "tenderness"
    LOVE = "love"
    LONGING = "longing"
    NOSTALGIA = "nostalgia"
    BITTERSWEET = "bittersweet"
    JOY = "joy"
    DELIGHT = "delight"
    WONDER = "wonder"
    AWE = "awe"
    SERENITY = "serenity"
    WHIMSY = "whimsy"
    COMEDY = "comedy"
    SATIRE = "satire"
    IRONY = "irony"
    RAGE = "rage"
    DEFIANCE = "defiance"
    TRIUMPH = "triumph"
    SHAME = "shame"
    DISGUST = "disgust"


class Intensity(str, Enum):
    LOW = "low"            # a faint wash of the feeling, mostly beneath the surface
    MODERATE = "moderate"  # clearly present, colours the scene
    HIGH = "high"          # the feeling dominates every line


class Valence(str, Enum):
    DARK = "dark"          # negative / oppressive
    MIXED = "mixed"        # ambivalent (e.g. bittersweet, ironic)
    LIGHT = "light"        # positive / uplifting


class SceneTags(BaseModel):
    # STANDARD: exactly ONE dominant_tone (a scene is one flavor), ONE intensity,
    # ONE valence, and 1-3 lowercase modern descriptor adjectives. The enums are
    # the rigid search facets; `descriptors` carries modern vocabulary so a vibe
    # query ("claustrophobic dread") still matches 1865 prose that never says it.
    dominant_tone: Tone
    intensity: Intensity
    valence: Valence
    descriptors: list[str] = Field(
        min_length=1, max_length=3,
        description="1-3 lowercase modern adjectives for the flavor, e.g. "
                    "['creeping', 'claustrophobic', 'suffocating'].",
    )

    @field_validator("descriptors")
    @classmethod
    def _normalize(cls, v: list[str]) -> list[str]:
        cleaned = [d.strip().lower() for d in v if d and d.strip()]
        if not (1 <= len(cleaned) <= 3):
            raise ValueError("descriptors must have 1-3 non-empty items")
        return cleaned


TAGS_TOOL = pydantic_function_tool(
    SceneTags,
    name="output_tags",
    description="Return the rigid single-tone flavor tags for ONE scene.",
)
# -----------------------------------------------------------------------------

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
    # every index in indexed_paragraphs for the section.
    # (read_only_context_paragraphs are NOT expected in the output.)
    obj = json.loads(payload)
    return {p["index"] for p in obj.get("indexed_paragraphs", [])}


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

    PARA_BREAK = "</p><p>"
    last = len(merged) - 1
    records = []
    for i, m in enumerate(merged):
        start, end = m["start"], m["end_paragraph_index"]
        text = "<p>" + PARA_BREAK.join(text_of[j].strip() for j in range(start, end + 1) if j in text_of) + "</p>"
        # neighbor pointers: scenes are contiguous + book-ordered, so prev/next by
        # index power the small-to-big fetch (grab surrounding scenes after a hit)
        # and let Mode-2 transitions be built as a join over adjacent pairs.
        records.append({
            "scene_id": f"{file_code}-{i}",
            "prev_scene_id": f"{file_code}-{i-1}" if i > 0 else None,
            "next_scene_id": f"{file_code}-{i+1}" if i < last else None,
            "chapter_title": chapter_of.get(start),
            "scene_title": m["title"],
            "stitch_status": m["status"],   # complete | stitched | broken_stitch
            "start_paragraph_index": start,
            "end_paragraph_index": end,
            "text": text,
            "book_metadata": metadata,
            "summary": None,   # one-line flavor summary, enrichment LLM fills later
            "tags": None,      # SceneTags, enrichment (second-phase) LLM fills later
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

    def work(chunk):
        # one chunk answered by one LLM call
        key = str(chunk.chunk_index)
        label = f"CHUNK {key}"
        cpath = ckpt_dir / f"chunk-{key}.json"

        # resume: skip chunks already segmented in a prior run
        cached = _load_checkpoint(cpath)
        if cached is not None:
            with print_lock:
                print(f"**** {label} CACHED (skip LLM) ****")
            return cached

        data = sb.break_chunk(chunk.scene_payload())
        _save_checkpoint(cpath, data)  # persist before printing, so it survives a crash

        # print as each chunk finishes, real-time
        with print_lock:
            print(f"**** {label} VERIFIED ****")
            for scene in data.scenes_data:
                if scene.paragraph_type == "noise" or scene.title == "NOISE":
                    print(f"scene from {label} marked as noise")
                    continue
                print(f"scene from {label} passed")
                print(f"scene open? {scene.open_end_index} on end, {scene.open_start_index} on start.")
        return data

    # up to 6 concurrent LLM calls; results ordered by chunk (book) position
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(work, book.chunks))

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
