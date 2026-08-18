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
import threading, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from data import build_library, MetadataParser, parse_rights
from process import SceneBreaker, scenes_to_records, _load_checkpoint, _save_checkpoint
from embed import enrich_file, index_records
from storage import write_json, read_json
from qdrant_client import QdrantClient
import search, subprocess, os, sys, time


# book-level embed gate: skip a book when non-prose "other" (poetry/plays) exceeds this fraction of its non-noise text
OTHER_SKIP_RATIO = 0.70
TEST_PATH = os.path.dirname(os.path.abspath(sys.argv[0])) + "/test"
DATA_PATH = TEST_PATH + "/data"                 # contains the raw zip files needed
SCENES_PATH = TEST_PATH + "/scenes"             # contains fully stitched scenes + fields before and after fill
SEGMENTS_PATH = TEST_PATH + "/segments"         # contains excluded book list + pre-processing chunks
RECALL_PATH = TEST_PATH + "/recall"             # contains books and metadata jsons for recall
CHECKPOINT_DIR = TEST_PATH + "/checkpoints"
QDRANT_PATH = TEST_PATH + "/qdrant_db"          # on-disk local Qdrant db (embed writes, search reads)

# a book whose metadata Subjects contain any of these words is treated as non-prose
# and excluded from segmentation up front; the exclusion is logged to EXCLUDED_BOOKS_FILE
EXCLUDE_SUBJECT_WORDS = ("poems", "poetry", "plays", "drama")
EXCLUDED_BOOKS_FILE = "excluded-books.json"

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
    "claustrophobic, suffocating, dread",
    "joyful, tender, warm",
    "cunning, clever, smart",
]

# --- payload dump (was data.main) --- #

def payload_dump_test():
    """Dump every book's chunk payloads to SEGMENTS_PATH/pg{code}-p.json for inspection."""
    _, books = build_library(data_path=DATA_PATH, recall_path=RECALL_PATH)
    for book in books.values():
        print(book.file_code)
        book.to_json(SEGMENTS_PATH)


# --- segmentation run (was process.main) --- #

def _exclude_book(code: str, md: dict, reason: str, detail=None):
    """Append one rejected book to the running excluded-books json under SEGMENT_PATH.

    reason is "non prose" or "not public domain"; detail is the matched subjects or
    the dc.rights string that triggered it. Kept auditable for later look-back.
    """
    path = f"{SEGMENTS_PATH}/{EXCLUDED_BOOKS_FILE}"
    excluded = read_json(path, {})
    excluded[code] = {
        "reason": reason,
        "detail": detail,
        "metadata": MetadataParser.to_dict(md),
    }
    write_json(path, excluded)


def segment_test(metadata: dict, books: dict, desired: str):
    """Segment ONE book with the LLM (resumable via checkpoints) and write its scenes json."""
    sb = SceneBreaker()

    desired_book = []
    book = books[desired]

    # pre-segmentation gates: reject + log books that must not be embedded.
    md = metadata[desired]

    # gate 1: US public domain only — read dc.rights straight from the book HTML
    rights = parse_rights(desired, DATA_PATH)
    if rights != "Public domain in the USA.":
        _exclude_book(desired, md, "not public domain", rights)
        print(f"EXCLUDE pg{desired}: dc.rights={rights!r} not US public domain — "
              f"not segmenting (logged to {SEGMENTS_PATH}/{EXCLUDED_BOOKS_FILE})")
        return

    # gate 2: non-prose by subject (poetry / plays / drama)
    matched = sorted(s for s in (md.get("Subjects") or [])
                     if any(w in s.lower() for w in EXCLUDE_SUBJECT_WORDS))
    if matched:
        _exclude_book(desired, md, "non prose", matched)
        print(f"EXCLUDE pg{desired}: subjects {matched} — not segmenting "
              f"(logged to {SEGMENTS_PATH}/{EXCLUDED_BOOKS_FILE})")
        return

    print_lock = threading.Lock()

    ckpt_dir = Path(CHECKPOINT_DIR) / f"pg{desired}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def work(chunk):
        """Segment one chunk with a single LLM call, reusing a checkpoint if present."""
        key = str(chunk.chunk_index)
        label = f"CHUNK {key}"
        cpath = ckpt_dir / f"chunk-{key}.json"

        cached = _load_checkpoint(cpath)
        if cached is not None:
            with print_lock:
                print(f"**** {label} CACHED (skip LLM) ****")
            return cached

        data = sb.break_chunk(chunk.scene_payload())
        _save_checkpoint(cpath, data)  # persist before printing, so it survives a crash

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

    # collect kept scenes in order; count paragraphs dropped as noise, and tally
    # "other" (poetry/plays) so a book that is mostly non-prose can be skipped
    kept_paras, noise_paras, other_paras = 0, 0, 0
    for data in results:
        for scene in data.scenes_data:
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

    # book-level gate: if non-prose "other" (poetry/plays) is > 70% of the non-noise
    # text, this is not a prose book — skip records/embedding entirely.
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

    # book fully saved: drop its checkpoints, no longer needed for resume
    shutil.rmtree(ckpt_dir, ignore_errors=True)

# --- enrichment + indexing run (was embed.main) --- #

def embed_test(file_ids=None):
    """Enrich each scenes json under SCENES_PATH and index it into a LOCAL Qdrant db
    saved in the test root (TEST_PATH/qdrant_db). Test-tree mirror of embed.main.

    file_ids picks specific books; None enriches every pg*-s.json under SCENES_PATH.
    """
    if file_ids:
        files = [Path(f"{SCENES_PATH}/pg{c}-s.json") for c in file_ids if c]
    else:
        files = sorted(Path(SCENES_PATH).glob("pg*-s.json"))

    client = QdrantClient(path=QDRANT_PATH)   # on-disk local db in the test root, auto-created
    try:
        for f in files:
            if not f.exists():
                print(f"skip (missing): {f}")
                continue
            records = enrich_file(f)         # LLM-enrich in place (resumable), rewrite the json
            index_records(client, records)   # upsert one point per scene into the local collection
    finally:
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
        for desc in DESCRIPTOR_QUERIES:
            print(f"\nQUERY: descriptors {desc!r}")
            _show(search.search_descriptors(client, desc, limit=limit, flt=flt))
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
    step_two_processing(FILE_IDS)
    #step_three_embedding(FILE_IDS)
    #search_test()


if __name__ == "__main__":
    main()