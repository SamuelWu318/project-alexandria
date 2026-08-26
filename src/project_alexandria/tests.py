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
import subprocess, os, sys, time, re, contextlib

from data import build_library
from process import scenes_to_records, segment_book, presegmentation_gate
from embed import enrich_file, index_records
import search

from utils import write_json, read_json, relational, log, SCHEMA_VERSION, SrcPaths
from qdrant_client import QdrantClient

# book-level embed gate: skip a book when non-prose "other" (poetry/plays) exceeds this fraction of its non-noise text
OTHER_SKIP_RATIO = 0.70
# TEST_PATH = os.path.dirname(os.path.abspath(sys.argv[0])) + "/test"
# DATA_PATH = TEST_PATH + "/data"                 # contains the raw zip files needed
# SrcPaths.SCENES_DIR = TEST_PATH + "/scenes"             # contains fully stitched scenes + fields before and after fill
# SEGMENTS_PATH = TEST_PATH + "/segments"         # contains excluded book list + pre-processing chunks
# RECALL_PATH = TEST_PATH + "/recall"             # contains books and metadata jsons for recall
# CHECKPOINT_DIR = TEST_PATH + "/checkpoints"
# SrcPaths.QDRANT_DIR = TEST_PATH + "/qdrant_db"          # on-disk local Qdrant db (embed writes, search reads)
# DB_PATH = TEST_PATH + "/scenes.db"              # on-disk SQLite relational mirror (embed writes, search reads)
# STATUS_PATH = CHECKPOINT_DIR + "/status.json"   # {book_id: "completed"} -> embed_test skips it on rerun

# --- test file numbers --- #

FILE_IDS = [
    "64317",    # great gatsby
    "71865",    # mrs dalloway
    "4300",     # ulysses
    "2814",     # dubliners
    "215",      # call of the wild
    "55",       # wizard of oz
    "73",       # red badge of courage
    "75201",    # a farewell to arms
    "2701",     # moby dick
    "1342",     # pride and prejudice
    "84",       # frankenstein
    "11",       # alice in wonderland
    "1661",     # sherlock holmes
    "345",      # dracula
    "98",       # tale of two cities
]

# --- qdrant search test --- #

# general, plot-free scene descriptions (how a writer searches), each mapped to a famous
# scene in one of the FILE_IDS books. Confirms GENERAL summaries retrieve the right scene.
TEST_QUERIES = [
    "A man is inspired by a wealthy host's extravagant party.",
    "An evil witch is killed.",              
    "A young soldier panics and flees from his first taste of battle.",
    "A man experiences a traumatic fight and narrowly wins or escapes."
]

# (summary, descriptors-list) pairs for search_combined: the summary gates the pool, the
# weighted-descriptor centroid reranks within it. descriptors is a LIST (equal-weighted).
COMBINED_QUERIES = [
    ("A crowd of glittering strangers drifts through a wealthy host's extravagant summer party.",
     ["festive", "glamorous", "hollow"]),                       # Gatsby (64317)
    ("A wounded officer and a nurse fall in love in a wartime hospital.",
     ["tender", "bittersweet", "yearning"]),                    # A Farewell to Arms (75201)
]

# pure-descriptor (vibe-only) queries — each a descriptor list for search_weighted_descriptors.
DESCRIPTOR_QUERIES = [
    ["festive", "glamorous", "restless"],           # Gatsby-party glitter (64317)
    ["claustrophobic", "absurd", "dehumanizing"],
    ["chaotic", "terrifying", "cowardly"],          # Red Badge battle-panic (73)
]

# --- payload dump (was data.main) --- #

def payload_dump_test():
    """Dump every book's chunk payloads to SEGMENTS_PATH/pg{code}-p.json for inspection."""
    _, books = build_library(data_path=SrcPaths.DATA_DIR, recall_path=SrcPaths.RECALL_DIR)
    for book in books.values():
        log.info(f"dumping payloads for book {book.file_code}")
        book.to_json(SrcPaths.SEGMENTS_DIR)


# --- segmentation run (was process.main) --- #

def segment_test(metadata: dict, books: dict, desired: str):
    """Segment ONE book with the LLM (resumable via checkpoints) and write its scenes json."""
    # do not segment if a completed scenes file already exists
    if (SrcPaths.SCENES_DIR / f"pg{desired}-s.json").is_file(): 
        log.skip(f"book {desired} already in scenes — skip segmentation")
        return

    log.step(f"segmenting book {desired}")

    desired_book = []
    book, md = books[desired], metadata[desired]

    # pre-segmentation gates (public-domain + non-prose)
    if presegmentation_gate(desired, md, SrcPaths.DATA_DIR, SrcPaths.RECALL_DIR): return

    # LLM orchestration (parallel per-chunk calls + resumable checkpoints)
    scenes = segment_book(book, SrcPaths.CHECKPOINT_DIR)

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

    log.info(f"noise: dropped {noise_paras} paragraphs, kept {kept_paras} "
             f"({noise_paras + kept_paras} total covered)")

    # book-level gate: if non-prose "other" (poetry/plays) is > 70% text, skip it.
    other_ratio = other_paras / kept_paras if kept_paras else 0.0
    if other_ratio > OTHER_SKIP_RATIO:
        log.skip(f"book {desired}: 'other' (poetry/plays) is {other_paras}/{kept_paras} "
                 f"= {other_ratio:.0%} of non-noise text (> {OTHER_SKIP_RATIO:.0%}) — not embedding")
        return

    # the full stitched together segments
    records = scenes_to_records(desired, desired_book, books[desired], metadata[desired])

    out_path = f"{SrcPaths.SCENES_DIR}/pg{desired}-s.json"
    write_json(out_path, records)
    log.done(f"book {desired}: recorded {len(records)} scenes -> {out_path}")


# --- enrichment + indexing run (was embed.main) --- #

def _load_status() -> dict:
    """Load the test-root {book_id: status} map; missing file / key / null all mean 'not done'."""
    return read_json(SrcPaths.STATUS_PATH, {})


def _mark_status(code: str, value="completed"):
    """Persist one book's completion status so the next embed_test run skips it."""
    status = _load_status()
    status[code] = value
    write_json(SrcPaths.STATUS_PATH, status)
 

def embed_test(file_ids=None):
    """Enrich each scenes json under SrcPaths.SCENES_DIR and index it into a LOCAL Qdrant db
    saved in the test root (TEST_PATH/qdrant_db). Test-tree mirror of embed.main.

    file_ids picks specific books; None enriches every pg*-s.json under SrcPaths.SCENES_DIR.
    A book marked "completed" in STATUS_PATH is skipped; null its status (or delete the
    file) to force a redo.
    """
    if file_ids:
        files = [Path(f"{SrcPaths.SCENES_DIR}/pg{c}-s.json") for c in file_ids if c]
    else:
        files = sorted(Path(SrcPaths.SCENES_DIR).glob("pg*-s.json"))

    client = QdrantClient(path=str(SrcPaths.QDRANT_DIR))   # on-disk local db in the test root, auto-created
    conn = relational.open_db(SrcPaths.DB_PATH)        # test-root SQLite mirror, created/migrated on open
    status = _load_status()
    try:
        for f in files:
            m = re.search(r"pg(\d+)-s\.json$", f.name)
            code = m.group(1) if m else f.stem
            if not f.exists():
                log.skip(f"book {code}: skip embedding (missing scenes json)")
                continue
            if status.get(code) == "completed":
                log.skip(f"book {code}: skip embedding (already completed)")
                continue
            records = enrich_file(f)              # LLM-enrich in place (resumable), rewrite the json
            index_records(client, records, conn)  # vectors + relational mirror, in lockstep
            _mark_status(code, "completed")        # persisted so the next run skips it
            log.done(f"book {code}: embedding finished")
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


def search_test(book_id: str = None, limit: int = 2):
    """Run summary-only, summary+descriptors (weighted), then pure-descriptor searches."""
    client = QdrantClient(path=str(SrcPaths.QDRANT_DIR))   # read the test db that embed_test wrote

    flt = search.book_filter(book_id)

    try:
        log.step("SUMMARY ONLY")
        for q in TEST_QUERIES:
            print(f"\nQUERY: {q}")
            _show(search.search_summary(client, q, limit=limit, flt=flt))

        log.step("SUMMARY + WEIGHTED DESCRIPTORS (search_combined)")
        for summ, desc in COMBINED_QUERIES:
            print(f"\nQUERY: {summ!r}  +  descriptors {desc!r}")
            _show(search.search_combined(client, summ, desc, limit=limit, flt=flt))

        log.step("DESCRIPTORS ONLY")
        for descriptors in DESCRIPTOR_QUERIES:
            print(f"\nQUERY: descriptors {descriptors!r}")
            _show(search.search_weighted_descriptors(client, descriptors))
    finally:
        client.close()


# --- keep-awake (macOS lid-close survival) --- #

def _pmset_disablesleep(value: int) -> bool:
    """Toggle macOS lid-close sleep via `sudo pmset -b disablesleep <value>` (needs admin;
    prompts for your password in the terminal). 1 = never sleep on lid close even on
    BATTERY; 0 = restore default. Returns True on success; safe warn-and-continue if it
    fails (no sudo rights, no tty, or non-macOS)."""
    try:
        subprocess.run(["sudo", "pmset", "-b", "disablesleep", str(value)], check=True)
        return True
    except (FileNotFoundError, OSError, subprocess.CalledProcessError) as e:
        log.warn(f"pmset disablesleep {value} failed ({e}) — lid-close sleep unchanged")
        return False


@contextlib.contextmanager
def stay_awake():
    """Keep the Mac awake for the whole block so a long process/embed run survives a
    closed lid, then automatically restore normal sleep on exit — including Ctrl-C.

    Two layers:
      * `sudo pmset -b disablesleep 1` — strong guard: no lid-close sleep even on
        BATTERY. Needs admin, so it prompts for your password when you run tests.py.
        Auto-reverted to 0 in `finally` (normal end, exception, OR Ctrl-C).
      * `caffeinate -imsw <pid>` — prevents idle/disk/system sleep, bound to this PID.

    Only a low-battery/critical sleep or a manual Ctrl-C stops the run. (A hard SIGKILL
    would skip the revert; if that ever happens, rerun `sudo pmset -b disablesleep 0`.)
    """
    proc = None
    disabled = _pmset_disablesleep(1)   # admin prompt; reverted in finally below
    try:
        proc = subprocess.Popen(["caffeinate", "-i", "-m", "-s", "-w", str(os.getpid())])
        log.info("staying awake: lid-close safe (Ctrl-C or low battery stops it)")
    except (FileNotFoundError, OSError):
        log.warn("caffeinate unavailable — relying on pmset only")
    try:
        yield
    finally:
        if proc is not None:
            proc.terminate()
        if disabled:
            _pmset_disablesleep(0)      # restore normal sleep on Ctrl-C / end / error


# --- steps --- #

def step_one_retrieval(file_ids):
    """Download each book's -h.zip into DATA_PATH. Launches the wgets in parallel,
    then waits for all of them so step two never reads a half-finished download."""
    Path(SrcPaths.DATA_DIR).mkdir(parents=True, exist_ok=True)
    procs = []

    for file_id in file_ids:
        if not file_id or (SrcPaths.DATA_DIR / f"pg{file_id}-h.zip").is_file(): continue

        cmd = ["wget", "-nc", "-nd", "-q", "--no-check-certificate", f"https://aleph.gutenberg.org/cache/epub/{file_id}/pg{file_id}-h.zip"]
        procs.append(subprocess.Popen(cmd, cwd=SrcPaths.DATA_DIR))
        log.info(f"book {file_id}: downloading")

    for p in procs:
        p.wait()

def step_two_processing(file_ids):
    """Segments all files into scenes, creating segment, scenes, and recall folders."""
    with stay_awake():   # process runs long — survive a closed lid
        metadata, books = build_library(data_path=SrcPaths.DATA_DIR, recall_path=SrcPaths.RECALL_DIR)
        for file_id in file_ids:
            if not (SrcPaths.DATA_DIR / f"pg{file_id}-h.zip").is_file(): 
                log.warn(f"book {file_id}: download is not a zip")
                continue
            segment_test(metadata, books, file_id)

def step_three_embedding(file_ids):
    """Enrich + index each book's scenes into the local Qdrant db (test root),
    skipping any id with no scenes json yet (excluded or not segmented)."""
    with stay_awake():   # embed runs long — survive a closed lid
        exist_ids = []
        for file_id in file_ids:
            if not (SrcPaths.SCENES_DIR / f"pg{file_id}-s.json").is_file(): continue
            exist_ids.append(file_id)
        embed_test(exist_ids)


def distill_and_search(sentence: str, limit: int = 5, book_id: str = None):
    """TEMPORARY Phase-4 driver: distil a raw writer query into a frame, run search_fused,
    print the frame + hits. Eyeball the whole read path on the test index."""
    from embed import distill_query
    frame = distill_query(sentence)
    log.step(f"FRAME for {sentence!r}")
    for k in ("summary", "subject", "verb", "object", "setting", "descriptors"):
        print(f"  {k:11}: {frame[k]!r}")
    client = QdrantClient(path=str(SrcPaths.QDRANT_DIR))
    try:
        hits = search.search_fused(client, frame, limit=limit,
                                   flt=search.book_filter(book_id))
        _show(hits)
    finally:
        client.close()


def main():
    step_one_retrieval(FILE_IDS)
    step_two_processing(FILE_IDS)
    step_three_embedding(FILE_IDS)
    #search_test()
    pass


if __name__ == "__main__":
    main()