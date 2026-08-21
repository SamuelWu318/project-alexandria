# FOR CLAUDE — Interactive test / smoke harness (run by hand, not pytest).
# -----------------------------------------------------------------------------
# Three entry points, called manually:
#   * search_test()       — the read path: run canned queries against the live
#                           Qdrant index and print hits. This is what main() runs.
#   * segment_test(code)  — the stage-2 driver: build the library, segment ONE book
#                           with the LLM (resumable via checkpoints), write its
#                           scenes json. Was process.main.
#   * payload_dump_test() — dump every book's chunk payloads for eyeballing. Was
#                           data.main.
# Paths + JSON IO come from storage.py.
# -----------------------------------------------------------------------------
from pathlib import Path

from data import build_library
from process import scenes_to_records, segment_book, presegmentation_gate
from embed import enrich_file, index_records
from storage import write_json, read_json
from qdrant_client import QdrantClient
import search, relational, subprocess, os, sys, time, re


# book-level embed gate: skip a book when non-prose "other" (poetry/plays) exceeds this fraction of its non-noise text
OTHER_SKIP_RATIO = 0.70
TEST_PATH = os.path.dirname(os.path.abspath(sys.argv[0])) + "/test"
DATA_PATH = TEST_PATH + "/data"                 # contains the raw zip files needed
SCENES_PATH = TEST_PATH + "/scenes"             # contains fully stitched scenes + fields before and after fill
SEGMENTS_PATH = TEST_PATH + "/segments"         # contains excluded book list + pre-processing chunks
RECALL_PATH = TEST_PATH + "/recall"             # contains books and metadata jsons for recall
CHECKPOINT_DIR = TEST_PATH + "/checkpoints"
QDRANT_PATH = TEST_PATH + "/qdrant_db"          # on-disk local Qdrant db (embed writes, search reads)
DB_PATH = TEST_PATH + "/scenes.db"              # on-disk SQLite relational mirror (embed writes, search reads)
STATUS_PATH = CHECKPOINT_DIR + "/status.json"   # {book_id: "completed"} -> embed_test skips it on rerun

# --- test file numbers --- #

FILE_IDS = [
    "64317",    # great gatsby
    "71865",    # mrs dalloway
    "4300",     # ulysses
    "2814",     # dubliners
    "5200",     # metamorphosis
    "215",      # call of the wild
    "55",       # wizard of oz
    "73",       # red badge of courage
    "75201",    # a farewell to arms
]

# --- qdrant search test --- #

# general, plot-free scene descriptions (how a writer searches) mapped to famous
# Odyssey (pg1727) scenes. Confirms the GENERAL summaries retrieve the right scene.
TEST_QUERIES = [
    "a man outsmarting a physically powerful opponent",          # Cyclops / Polyphemus
    "a hero resisting the lure of enchanting voices",            # Sirens
    "a sorceress turning men into animals",                      # Circe
    "a disguised ruler testing the loyalty of his household",    # beggar disguise
    "a contest of strength only the true master can win",        # stringing the bow
    "a long-separated husband and wife reunited",                # Odysseus & Penelope
    "a warrior killing the men who invaded his home",            # slaying the suitors
    "a living man journeying among the dead",                    # the Underworld
    "an old servant recognizing a hero by an old scar",          # Eurycleia & the scar
    "sailors doomed for eating forbidden cattle",                # Cattle of the Sun
]

# (summary, descriptors) pairs — exercise the fused two-vector search.
COMBINED_QUERIES = [
    ("a man outsmarting a physically powerful opponent", "cunning, triumphant, defiant"),
    ("a hero facing a monster at sea", "claustrophobic, terrifying, doomed"),
    ("a homecoming", "tender, bittersweet, joyful"),
]

# pure-descriptor (vibe-only) queries
DESCRIPTOR_QUERIES = [
    ["claustrophobic", "suffocating", "dread"],
    ["warm", "joyful", "relieved"]
]

# --- payload dump (was data.main) --- #

def payload_dump_test():
    """Dump every book's chunk payloads to SEGMENTS_PATH/pg{code}-p.json for inspection."""
    _, books = build_library(data_path=DATA_PATH, recall_path=RECALL_PATH)
    for book in books.values():
        print(book.file_code)
        book.to_json(SEGMENTS_PATH)


# --- segmentation run (was process.main) --- #

def segment_test(metadata: dict, books: dict, desired: str):
    """Segment ONE book with the LLM (resumable via checkpoints) and write its scenes json."""
    # do not segment if a completed scenes file already exists
    if Path(SCENES_PATH + f"/pg{desired}-s.json").is_file(): 
        print(f"*** BOOK OF CODE {desired} ALREADY IN SCENES (skip segmentation) ***")
        return

    desired_book = []
    book, md = books[desired], metadata[desired]

    # pre-segmentation gates (public-domain + non-prose)
    if presegmentation_gate(desired, md, DATA_PATH, SEGMENTS_PATH): return

    # LLM orchestration (parallel per-chunk calls + resumable checkpoints)
    scenes = segment_book(book, CHECKPOINT_DIR)

    # remove noise scenes, tally "other" scenes to check for poetry/plays to remove.
    kept_paras, noise_paras, other_paras = 0, 0, 0
    for scene in scenes:
        span = scene.end_paragraph_index - scene.start_paragraph_index + 1
        if scene.paragraph_type == "noise" or scene.title == "NOISE":
            noise_paras += span
            continue
        kept_paras += span
        if scene.content_form == "other":
            other_paras += span
        desired_book.append(scene)

    print(f"NOISE: dropped {noise_paras} paragraphs as noise; kept {kept_paras} "
          f"({noise_paras + kept_paras} total covered)")

    # book-level gate: if non-prose "other" (poetry/plays) is > 70% text, skip it.
    other_ratio = other_paras / kept_paras if kept_paras else 0.0
    if other_ratio > OTHER_SKIP_RATIO:
        print(f"SKIP pg{desired}: 'other' (poetry/plays) is {other_paras}/{kept_paras} "
              f"= {other_ratio:.0%} of non-noise text (> {OTHER_SKIP_RATIO:.0%}) — not embedding this book")
        return

    # the full stitched together segments
    records = scenes_to_records(desired, desired_book, books[desired], metadata[desired])

    out_path = f"{SCENES_PATH}/pg{desired}-s.json"
    write_json(out_path, records)
    print(f"wrote {len(records)} scenes to {out_path}")


# --- enrichment + indexing run (was embed.main) --- #

def _load_status() -> dict:
    """Load the test-root {book_id: status} map; missing file / key / null all mean 'not done'."""
    return read_json(STATUS_PATH, {})


def _mark_status(code: str, value="completed"):
    """Persist one book's completion status so the next embed_test run skips it."""
    status = _load_status()
    status[code] = value
    write_json(STATUS_PATH, status)
 

def embed_test(file_ids=None):
    """Enrich each scenes json under SCENES_PATH and index it into a LOCAL Qdrant db
    saved in the test root (TEST_PATH/qdrant_db). Test-tree mirror of embed.main.

    file_ids picks specific books; None enriches every pg*-s.json under SCENES_PATH.
    A book marked "completed" in STATUS_PATH is skipped; null its status (or delete the
    file) to force a redo.
    """
    if file_ids:
        files = [Path(f"{SCENES_PATH}/pg{c}-s.json") for c in file_ids if c]
    else:
        files = sorted(Path(SCENES_PATH).glob("pg*-s.json"))

    client = QdrantClient(path=QDRANT_PATH)   # on-disk local db in the test root, auto-created
    conn = relational.open_db(DB_PATH)        # test-root SQLite mirror, created/migrated on open
    status = _load_status()
    try:
        for f in files:
            if not f.exists():
                print(f"skip (missing): {f}")
                continue
            m = re.search(r"pg(\d+)-s\.json$", f.name)
            code = m.group(1) if m else f.stem
            if status.get(code) == "completed":
                print(f"skip (completed): {code}")
                continue
            records = enrich_file(f)              # LLM-enrich in place (resumable), rewrite the json
            index_records(client, records, conn)  # vectors + relational mirror, in lockstep
            _mark_status(code, "completed")        # persisted so the next run skips it
            print(f"marked {code} completed")
    finally:
        conn.close()
        client.close()


# --- search --- #

def _show(hits):
    """Print each hit: score, scene id, flavor tags, title, summary, descriptors."""
    for h in hits:
        p = h.payload
        print(f"  {round(h.score, 3)}  {p['scene_id']}  [{p.get('dominant_tone')}"
              f"/{p.get('intensity')}/{p.get('arc')}]  {p.get('scene_title')}")
        print(f"     {p.get('summary')}  << {p.get('descriptors')}")


def search_test(book_id: str, limit: int = 2):
    """Run summary-only, summary+descriptors (weighted), then pure-descriptor searches."""
    client = QdrantClient(path=QDRANT_PATH)   # read the test db that embed_test wrote

    flt = search.book_filter(book_id)

    try:
        print("===== SUMMARY ONLY =====")
        for q in TEST_QUERIES:
            print(f"\nQUERY: {q}")
            _show(search.search_summary(client, q, limit=limit, flt=flt))

        print("\n===== SUMMARY + DESCRIPTORS (weighted) =====")
        for summ, desc in COMBINED_QUERIES:
            print(f"\nQUERY: {summ!r}  +  descriptors {desc!r}")
            _show(search.search_summary(client, summ, descriptors=desc, limit=limit, flt=flt))

        print("\n===== DESCRIPTORS ONLY =====")
        for descriptors in DESCRIPTOR_QUERIES:
            print(f"\nQUERY: descriptors {descriptors!r}")
            _show(search.search_weighted_descriptors(client, descriptors))
    finally:
        client.close()


# --- steps --- #

def step_one_retrieval(file_ids):
    """Download each book's -h.zip into DATA_PATH. Launches the wgets in parallel,
    then waits for all of them so step two never reads a half-finished download."""
    Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
    procs = []

    for file_id in file_ids:
        if not file_id or Path(DATA_PATH + f"/pg{file_id}-h.zip").is_file(): continue

        cmd = ["wget", "-nc", "-nd", "-q", "--no-check-certificate", f"https://aleph.gutenberg.org/cache/epub/{file_id}/pg{file_id}-h.zip"]
        procs.append(subprocess.Popen(cmd, cwd=DATA_PATH))

    for p in procs:
        p.wait()

def step_two_processing(file_ids):
    """Segments all files into scenes, creating segment, scenes, and recall folders."""
    metadata, books = build_library(data_path=DATA_PATH, recall_path=RECALL_PATH)
    for file_id in file_ids:
        if not Path(DATA_PATH + f"/pg{file_id}-h.zip").is_file(): continue
        segment_test(metadata, books, file_id)

def step_three_embedding(file_ids):
    """Enrich + index each book's scenes into the local Qdrant db (test root),
    skipping any id with no scenes json yet (excluded or not segmented)."""
    exist_ids = []
    for file_id in file_ids:
        if not Path(SCENES_PATH + f"/pg{file_id}-s.json").is_file(): continue
        exist_ids.append(file_id)
    embed_test(exist_ids)

def main():
    #step_one_retrieval(FILE_IDS)
    #step_two_processing(FILE_IDS)
    step_three_embedding(FILE_IDS)
    #search_test()
    pass


if __name__ == "__main__":
    main()