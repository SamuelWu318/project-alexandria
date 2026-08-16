# FOR CLAUDE — Stage 2: segmentation (cut chunks into flavor-pure scenes).
# -----------------------------------------------------------------------------
# SceneBreaker.break_chunk sends one section (Chunk.scene_payload) to the LLM and
# forces an output_scenes tool call labelling every paragraph scene/noise, wrapped
# in a retry loop (transient backoff + corrective-feedback re-ask) and a coverage
# check (every input index covered exactly once). scenes_to_records then stitches
# open-ended scenes across chunks and flattens them into one-record-per-scene dicts
# for embed.py (enrichment fields start null).
#
# OWNERSHIP: the prompts (SYSTEM_PROMPT), MODEL, the tag-vocab enums (Tone /
# Intensity / Arc / SceneTags) and the retry/temperature policy are the user's
# tuning surface — do not touch unless asked. Plumbing (IO via storage.py,
# docstrings, assembly) is fair game.
#
# Shared downstream: embed.py imports CLIENT, MODEL, Tone, Intensity, Arc,
# _classify_error, SCHEMA_VERSION from here. Paths + JSON IO live in storage.py.
# -----------------------------------------------------------------------------
from data import MetadataParser
import os, json, re, time, math
from collections import Counter
import openai
from openai import OpenAI, pydantic_function_tool
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, Field, field_validator
from typing import Literal
from enum import Enum
from pathlib import Path

from storage import read_text, write_text
load_dotenv()


# --- constants --- #

SCHEMA_VERSION = 1   # bump when the scene-record shape changes (embed.py reads it)

# --- model constants --- #

MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
SYSTEM_PROMPT = """
    # ROLE
    You split one book section into an ordered list of segments. Each segment is either:
    - "scene": story prose or dialogue.
    - "noise": non-story text — unnecessary HTML, licenses, footnotes, captions, table-of-contents, chapter titles, headers, images.
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

# --- enrichment classes --- #

# Rigid flavor tags for the goal: fetch scenes by emotional FLAVOR, then inject
# that flavor into the user's own prose. A scene is ONE tone (see HOW TO CUT), so
# these tags describe exactly ONE dominant feeling. The enrichment LLM call fills
# them on the fully-stitched scene text; a record holds None until then.

class Tone(str, Enum):
    # CONTROLLED VOCABULARY — the single dominant feeling of a scene, and the rigid
    # facet Mode-1 search filters on (Mode-2 transition pairs are built from it too).
    #
    # Laid out on the empirically-derived 4-D affective space of Fontaine, Scherer,
    # Roesch & Ellsworth (2007), "The World of Emotions is not Two-Dimensional"
    # (Psychological Science). The four axes, in order of importance, are VALENCE
    # (pleasant<->unpleasant), POTENCY/CONTROL (weak<->dominant), AROUSAL
    # (calm<->activated) and NOVELTY (expected<->sudden). Named tones are drawn from
    # Scherer's Geneva Emotion Wheel (20 emotion families on a valence x control
    # wheel), pruned + extended for literary scene-flavor.
    #
    # INTENSITY and ARC are SEPARATE facets, so a tone names the QUALITY of a
    # feeling, never its strength (magnitude lives in Intensity). The sections tile
    # the Valence x Arousal quadrants, then the Novelty axis, then blended tones.
    # Edit freely to retune search; keep values lowercase. NOTE: changing this
    # vocabulary invalidates already-enriched tones — re-enrich to stay consistent.

    # negative · high-arousal — threat, conflict, agitation
    DREAD = "dread"              # anticipatory fear (low potency)
    TERROR = "terror"           # acute, overwhelming fear
    ANXIETY = "anxiety"         # restless worry / unease
    MENACE = "menace"           # outward threat, intimidation (high potency)
    RAGE = "rage"               # hot anger (high potency)
    DEFIANCE = "defiance"       # hostile resistance (high potency)
    DISGUST = "disgust"         # revulsion
    CONTEMPT = "contempt"       # cold scorn / disdain

    # negative · low-arousal — loss, sorrow, withdrawal
    GRIEF = "grief"             # acute mourning
    MELANCHOLY = "melancholy"   # pensive, settled sadness
    DESPAIR = "despair"         # hopelessness
    LONELINESS = "loneliness"   # isolation
    SHAME = "shame"             # self-directed disgrace
    GUILT = "guilt"             # remorse over a wrong done
    REGRET = "regret"           # wishing the past undone
    RESIGNATION = "resignation" # bleak, defeated acceptance

    # positive · high-arousal — energy, uplift, victory
    JOY = "joy"                 # bright happiness
    DELIGHT = "delight"         # lively, playful pleasure
    EXCITEMENT = "excitement"   # eager anticipation / thrill
    TRIUMPH = "triumph"         # exultant victory (high potency)
    HOPE = "hope"               # forward-looking optimism
    PASSION = "passion"         # ardor, desire, romantic heat
    AMUSEMENT = "amusement"     # mirth, comic pleasure
    WONDER = "wonder"           # marvel at something new (novelty+)

    # positive · low-arousal — calm, warmth, connection
    SERENITY = "serenity"       # tranquil peace
    CONTENTMENT = "contentment" # settled satisfaction
    TENDERNESS = "tenderness"   # gentle, protective care
    AFFECTION = "affection"     # fond, steady love
    RELIEF = "relief"           # tension released
    GRATITUDE = "gratitude"     # thankfulness
    COMPASSION = "compassion"   # sympathy for another's pain
    PRIDE = "pride"             # quiet self-worth (high potency)

    # novelty axis — expectation violated (valence-ambiguous)
    SURPRISE = "surprise"       # sudden astonishment
    SUSPENSE = "suspense"       # tense, held-breath anticipation
    CURIOSITY = "curiosity"     # drawn-in intrigue / interest
    AWE = "awe"                 # reverent, overwhelmed vastness

    # complex / blended — mixed-valence literary flavors
    BITTERSWEET = "bittersweet" # joy and sorrow at once
    NOSTALGIA = "nostalgia"     # wistful ache for the past
    LONGING = "longing"         # yearning for the absent / distant
    FOREBODING = "foreboding"   # ominous sense of coming ill
    IRONY = "irony"             # detached incongruity
    SATIRE = "satire"           # mocking social critique
    WHIMSY = "whimsy"           # light, fanciful playfulness
    SOLEMNITY = "solemnity"     # grave, ceremonial dignity


class Intensity(str, Enum):
    LOW = "low"            # a faint wash of the feeling, mostly beneath the surface
    MODERATE = "moderate"  # clearly present, colours the scene
    HIGH = "high"          # the feeling dominates every line


class Arc(str, Enum):
    # trajectory of the feeling ACROSS the scene. Powers the box-to-box sequence
    # search ("as tension builds") and scaffolds a summary that names direction.
    RISING = "rising"      # feeling intensifies toward the end
    STEADY = "steady"      # feeling holds at one level throughout
    FALLING = "falling"    # feeling releases / subsides toward the end
    TURN = "turn"          # feeling flips or pivots partway through


class SceneTags(BaseModel):
    # STANDARD: exactly ONE dominant_tone (a scene is one flavor), ONE intensity,
    # ONE arc (its trajectory), and 1-3 lowercase modern descriptor adjectives. The
    # enums are the rigid search facets; `descriptors` carries modern vocabulary so
    # a vibe query ("claustrophobic dread") still matches 1865 prose that never says it.
    dominant_tone: Tone
    intensity: Intensity
    arc: Arc
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

# --- ai client --- #

CLIENT = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_KEY"],
)

def _classify_error(e: Exception) -> str:
    """Classify an API error: "transient" (retry with backoff) vs "fatal" (raise now).

    transient = network / 429 / 5xx; fatal = other 4xx where retrying won't help.
    """
    if isinstance(e, (openai.APIConnectionError, openai.APITimeoutError)):
        return "transient"
    if isinstance(e, openai.APIStatusError):
        return "transient" if (e.status_code == 429 or e.status_code >= 500) else "fatal"
    return "transient"  # unknown network-ish -> limited retry


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


class SceneBreaker:
    def break_chunk(self, chunk: str, max_transient_retries: int = 6,
                    max_validation_retries: int = 3):
        """Segment one section via a forced output_scenes call, retrying until coverage passes."""
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
    """Load a cached MultiSceneData, or None if missing/corrupt (forces recompute)."""
    raw = read_text(path)
    if raw is None:
        return None
    try:
        return MultiSceneData.model_validate_json(raw)
    except Exception:
        return None


def _save_checkpoint(path: Path, data: MultiSceneData):
    """Atomically checkpoint one chunk's segmentation (crash-safe via storage.write_text)."""
    write_text(path, data.model_dump_json())

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
        records.append({
            "scene_id": f"{file_code}-{i}",
            "book_id": file_code,
            "prev_scene_id": f"{file_code}-{i-1}" if i > 0 else None,
            "next_scene_id": f"{file_code}-{i+1}" if i < last else None,

            # --- flavor facets (Mode-1 filter); enrichment fills --- #
            "dominant_tone": None,
            "intensity": None,
            "arc": None,
            "descriptors": None,

            # --- transition facets (Mode-2); denormalized at enrichment --- #
            "prev_tone": None,
            "next_tone": None,

            # --- display / provenance --- #
            "scene_title": m["title"],
            "chapter_title": chapter_of.get(start),
            "stitch_status": m["status"],   # complete | stitched | broken_stitch
            "start_paragraph_index": start,
            "end_paragraph_index": end,
            "word_count": word_count,

            # --- book facets (flat filters) + full blob for display --- #
            "author": author,
            "language": language,
            "book_metadata": book_metadata,

            # --- content --- #
            "summary": None,        # flavor summary (embedded), enrichment fills
            "text_html": text,      # render form; embed source is `summary`

            # --- bookkeeping --- #
            "schema_version": SCHEMA_VERSION,
            "enriched": False,
            "enrich_model": None,
        })

    return records
