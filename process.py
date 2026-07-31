from data import MetadataParser, SceneParser
import os, json, re, time, math
from openai import OpenAI, pydantic_function_tool
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Literal
from pathlib import Path
load_dotenv()


# --- metadata parser constants --- #

CATALOG_PATH = "pg_catalog.csv"
DATA_PATH = "data"

# --- model constants --- #

MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
SYSTEM_PROMPT = """
    # ROLE
    You segment one chunk of a book chapter into either: 
    scenes of prose, or: noise from leftover html/licenses/footnotes/chapter titles/non-prose/headers.
    You label paragraphs by their index only, never rewriting or outputting the text directly.

    # INPUT
    You will receive one JSON object:
    - "chapter_title": the chapter the current section belongs to. use it as context for determining scenes/noise.
    - "section_within_chunk": "SECTION/TOTAL". SECTION is the 1-based index of the section within the chapter. (EX: 1/5 is FIRST SECTION, 5/5 is LAST SECTION) Use this to determine if some scenes were cut off during chunking.
    - "read_only_context_paragraphs": paragraphs from the PREVIOUS SECTION. Use it for additional context. They are never included in the output.
    - "number_of_indexed_paragraphs": number of paragraphs that MUST be segmented.
    - "indexed_paragraphs": the paragraphs themselves that MUST be segmented. Each is in form {"index": int, "text": str}. Ignore inline HTML, reason only about the words.

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
    - attempt to keep scenes between 3-5 paragraphs. But if there is dialogue, an extended scene, a shortened scenee, etc, do not feel restricted to only 3-5 paragraphs. 

    # EXAMPLE 1
    -- input --
    "chapter_title": "A Mad Tea-Party",
    "section_within_chunk": "1/1",
    "read_only_context_paragraphs": [],
    "indexed_paragraphs": [
    { "index": 0, "text": "There was a table set out under a tree, and the March Hare and the Hatter were having tea."},
    { "index": 1, "text": "Alice sat down uninvited. The Hatter asked a riddle with no answer."},
    { "index": 2, "text": "They moved round the table, and Alice, quite exhausted, walked off into the wood."},
    { "index": 3, "text": "[7] 'Mad as a hatter' predates Carroll; hatters were poisoned by mercury."}
    ]
    -- output_scenes --
    {"scenes_data": [
        {"start_paragraph_index": 0, "end_paragraph_index": 2, "paragraph_type": "scene", "open_start_index": False, "open_end_index": False, "title": "Tea with the Hatter and Hare"},
        {"start_paragraph_index": 3, "end_paragraph_index": 3, "paragraph_type": "noise", "open_start_index": False, "open_end_index": False, "title": "NOISE"}
    ]}

    Reasoning: section is 1/1, so this is one whole chapter with no broken scenes. The paragraph at index 3 is noise. Looking at the indices, 0-3 indices are covered exactly once.

    # EXAMPLE 2
    -- input --
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
    -- output_scenes --
    {"scenes_data":[
        {"start_paragraph_index": 9, "end_paragraph_index": 10, "paragraph_type": "scene", "open_start_index": True, "open_end_index": False, "title": "Menelaus finishes his tale"},
        {"start_paragraph_index": 11, "end_paragraph_index": 11, "paragraph_type": "scene", "open_start_index": False, "open_end_index": True, "title": "Telemachus speaks to his brothers"},
    ]}

    Reasoning: section is 2/3, so both open_start_index and open_end_index can be True. Using the read_only_context_paragraphs, we can see the first scene extends backward, so we set open_start_index to be True.
    Looking at the last scene available, it is obvious that the scene continues past that point, so we set open_end_index to be True. We also only use indices 9-11, as 7-8 are read_only_context_paragraphs.
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


class SceneBreaker:
    def break_chunk(self, chunk: str, max_retries: int = 10):
        last_err = None
        for attempt in range(max_retries):
            # temp 0, increase if attempts fail
            temp = 0 if attempt == 0 else math.log(attempt ** 0.15) + 0.15
            try:
                response = CLIENT.chat.completions.create(
                    model=MODEL, temperature=temp, tools=[TOOL],
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": chunk},
                    ],
                    tool_choice={"type": "function", "function": {"name": "output_scenes"}},
                    extra_body={"provider":{"require_parameters":True},
                                "reasoning": {"effort": "low"}}
                )

                choices = response.choices
                if not choices or not choices[0].message.tool_calls:
                    raise ValueError("no tool_call in response")

                args = choices[0].message.tool_calls[0].function.arguments
                return MultiSceneData.model_validate_json(args)

            except Exception as e:
                last_err = e
                print(f"  break_chunk retry {attempt + 1}/{max_retries}: {e}")
                time.sleep(min(2 ** attempt, 30))

        raise RuntimeError(f"break_chunk failed after {max_retries} tries: {last_err}")

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
    for code, book in books.items():
        if code != desired: continue
        for chunk in book.chunks:
            data = sb.break_chunk(chunk.scene_payload())
            print(f"**** CHUNK {chunk.chunk_index} VERIFIED ****")
            for scene in data.scenes_data:
                if scene.paragraph_type == "noise" or scene.title == "NOISE": 
                    print(f"scene from chunk {chunk.chunk_index} marked as noise")
                    continue
                print(f"scene from chunk {chunk.chunk_index} passed")
                desired_book.append(scene)

    records = scenes_to_records(desired, desired_book, books[desired], metadata[desired])

    out_path = Path("test/pg11-scenes.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(records)} scenes to {out_path}")

if __name__ == "__main__":
    main()
