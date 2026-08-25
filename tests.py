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
from dotenv import load_dotenv
from process import scenes_to_records, segment_book, presegmentation_gate
from embed import enrich_file, index_records, reset_all, reset_enriched
from storage import write_json, read_json
from qdrant_client import QdrantClient
import search, relational, subprocess, os, sys, time, re, contextlib
import log

load_dotenv()

# book-level embed gate: skip a book when non-prose "other" (poetry/plays) exceeds this fraction of its non-noise text
OTHER_SKIP_RATIO = 0.70
TEST_PATH = os.path.dirname(os.path.abspath(sys.argv[0])) + "/test"
DATA_PATH = TEST_PATH + "/data"                 # raw pg{code}-h.zip archives
SCENES_PATH = TEST_PATH + "/scenes"             # per-book scene records, pg{code}-s.json
SEGMENTS_PATH = TEST_PATH + "/segments"         # chunk payload dumps, pg{code}-p.json
RECALL_PATH = TEST_PATH + "/recall"             # parse cache: metadata.json + books.json + excluded-books.json
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
    # The Great Gatsby (64317)
    "A man stares across a dark bay toward a small green light, aching for a lost love.",       # famous
    "Almost no mourners attend the lavish funeral of a rich, once-celebrated host.",           # obscure
    # Mrs Dalloway (71865)
    "A woman sets out through the city on a summer morning to buy flowers for her evening party.",  # famous
    "A traumatized war veteran leaps from a high window to his death rather than be taken away.",   # obscure
    # Ulysses (4300)
    "A woman lies awake in bed at night, her unpunctuated thoughts flowing to a final whispered yes.",  # famous
    "A man watches from the rocks as a young woman leans back on the beach while fireworks burst overhead.",  # obscure
    # Dubliners (2814)
    "Snow falls across the whole country as a man learns his wife long mourned a boy who died for her.",  # famous
    "A young woman grips the dock railing, frozen, unable to sail away with her lover.",        # obscure
    # The Call of the Wild (215)
    "A domesticated dog finally answers the wild and runs off to join a wolf pack.",            # famous
    "A newly captured dog is beaten with a club into obedience, learning the law of force.",    # obscure
    # The Wonderful Wizard of Oz (55)
    "A girl flings a bucket of water on a wicked witch, who melts away to nothing.",            # famous
    "Travelers collapse into sleep crossing a vast field of deadly scarlet poppies.",          # obscure
    # The Red Badge of Courage (73)
    "A young soldier panics and flees from his first taste of battle.",                         # famous
    "A fleeing soldier is struck on the head by a comrade's rifle and passes the gash off as a battle wound.",  # obscure
    # A Farewell to Arms (75201)
    "A man walks back to his hotel alone in the rain after his lover dies giving birth.",       # famous
    "During a chaotic retreat a lieutenant is nearly executed by his own side and escapes by diving into a river.",  # obscure
    # Moby-Dick (2701)
    "A captain nails a gold coin to the mast and swears vengeance on a great white whale.",     # famous
    "Two strangers forced to share an inn's bed become unlikely close friends.",               # obscure
    # Pride and Prejudice (1342)
    "A proud suitor proposes clumsily and is indignantly refused by the woman he had insulted.",  # famous
    "A woman touring a grand country estate is startled to meet its owner and begins to see him differently.",  # obscure
    # Frankenstein (84)
    "On a dreary night a scientist brings a stitched corpse to life, then flees in horror.",    # famous
    "A lonely, hidden creature watches a poor family through a wall and teaches himself to speak and read.",  # obscure
    # Alice's Adventures in Wonderland (11)
    "A bored girl follows a waistcoated rabbit down a hole into a nonsensical world.",          # famous
    "A wailing baby handed to a girl gradually turns into a grunting pig.",                     # obscure
    # The Adventures of Sherlock Holmes (1661)
    "A master detective is outwitted by a clever woman and keeps her photograph in esteem.",    # famous
    "A red-haired shopkeeper is hired to copy books for a sham society while thieves tunnel beneath his shop.",  # obscure
    # Dracula (345)
    "A trapped guest watches his aristocratic host crawl face-down down the sheer castle wall.",  # famous
    "A storm drives an abandoned ship ashore, its crew vanished and the dead captain lashed to the wheel.",  # obscure
    # A Tale of Two Cities (98)
    "A man calmly takes a condemned stranger's place at the guillotine, at peace with his sacrifice.",  # famous
    "A man freed after long years in prison compulsively makes shoes, his mind broken.",        # obscure
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

# Book-agnostic writer queries: archetypal scenes a writer hunts for across the WHOLE
# corpus, not mapped to any one FILE_ID. Phrased in the index's physical-action register
# (grounded on-page action, light emotional framing) rather than interpretive gloss, so
# they exercise cross-corpus recall the way a real query pool would. Not wired into
# search_test by default — swap into the SUMMARY-ONLY loop to eyeball breadth.
GENERIC_QUERIES = [
    "A tense farewell at a train station as one lover boards and the other stays behind.",
    "A lone figure stands at a graveside in the rain.",
    "A heated argument that ends a long friendship for good.",
    "A first kiss interrupted at the worst possible moment.",
    "A soldier writes a last letter home the night before battle.",
    "A child overhears their parents fighting through a bedroom wall.",
    "Two former friends face each other in a duel at dawn.",
    "A shipwreck survivor washes up on an unfamiliar shore.",
    "A quiet confession of love that goes unanswered.",
    "A character wakes disoriented in an unfamiliar room.",
    "Two strangers lock eyes across a crowded ballroom.",
    "A deathbed reconciliation between an estranged parent and child.",
    "A character finds a hidden letter that changes everything.",
    "A storm forces travelers to shelter in a stranger's house for the night.",
    "A public humiliation in front of a watching crowd.",
    "A character carries a child out of a burning building.",
    "A tense negotiation where one side is quietly bluffing.",
    "Two siblings reunite after many years apart.",
    "A character swears revenge over a body.",
    "A lone figure watches the city from a rooftop at night.",
]

# --- payload dump (was data.main) --- #

def payload_dump_test():
    """Dump every book's chunk payloads to SEGMENTS_PATH/pg{code}-p.json for inspection."""
    _, books = build_library(data_path=DATA_PATH, recall_path=RECALL_PATH)
    for book in books.values():
        log.info(f"dumping payloads for book {book.file_code}")
        book.to_json(SEGMENTS_PATH)


# --- segmentation run (was process.main) --- #

def segment_test(metadata: dict, books: dict, desired: str):
    """Segment ONE book with the LLM (resumable via checkpoints) and write its scenes json."""
    # do not segment if a completed scenes file already exists
    if Path(SCENES_PATH + f"/pg{desired}-s.json").is_file(): 
        log.skip(f"book {desired} already in scenes — skip segmentation")
        return

    log.step(f"segmenting book {desired}")

    desired_book = []
    book, md = books[desired], metadata[desired]

    # pre-segmentation gates (public-domain + non-prose)
    if presegmentation_gate(desired, md, DATA_PATH, RECALL_PATH): return

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

    out_path = f"{SCENES_PATH}/pg{desired}-s.json"
    write_json(out_path, records)
    log.done(f"book {desired}: recorded {len(records)} scenes -> {out_path}")


# --- enrichment + indexing run (was embed.main) --- #

def _load_status() -> dict:
    """Load the test-root {book_id: status} map; missing file / key / null all mean 'not done'."""
    return read_json(STATUS_PATH, {})


def _mark_status(code: str, value="completed"):
    """Persist one book's completion status so the next embed_test run skips it."""
    status = _load_status()
    status[code] = value
    write_json(STATUS_PATH, status)


def _clear_status(code: str):
    """Drop one book's key from STATUS_PATH so embed_test stops treating it as completed.
    No-op (no rewrite) if the book has no status entry."""
    status = _load_status()
    if status.pop(code, None) is not None:
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
                log.skip(f"book {code}: skip embedding (missing scenes json)")
                continue
            m = re.search(r"pg(\d+)-s\.json$", f.name)
            code = m.group(1) if m else f.stem
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


def reset_test(file_ids=None):
    """Clear all resettable enrichment fields (tone, next_tone, prev_tone, descriptors,
    intensity, arc) AND revert the enrich gate (summary/enriched/enrich_model) on every
    book's scenes json, rewriting each in place — a full revert to the unenriched state.

    file_ids picks specific books (default FILE_IDS). Any id whose scenes json is missing
    is skipped. Also drops each book from STATUS_PATH, so the next embed_test fully
    re-enriches + re-indexes it instead of skipping it as completed.
    """
    ids = file_ids or FILE_IDS
    for code in ids:
        if not code:
            continue
        path = Path(f"{SCENES_PATH}/pg{code}-s.json")
        if not path.is_file():
            log.skip(f"book {code}: no scenes json — skip reset")
            continue
        records = read_json(path, [])
        reset_all(records)
        n = reset_enriched(records)
        write_json(path, records)
        _clear_status(code)   # drop the "completed" mark so embed_test redoes it
        log.done(f"book {code}: reset {n} scenes")


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
    client = QdrantClient(path=QDRANT_PATH)   # read the test db that embed_test wrote

    flt = search.book_filter(book_id)

    try:
        log.step("SUMMARY ONLY")
        for q in TEST_QUERIES:
            print(f"\nQUERY: {q}")
            _show(search.search_summary(client, q, limit=limit, flt=flt))

        log.step("GENERIC TESTING")
        for q in GENERIC_QUERIES:
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
    Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
    procs = []

    for file_id in file_ids:
        if not file_id or Path(DATA_PATH + f"/pg{file_id}-h.zip").is_file(): continue

        cmd = ["wget", "-nc", "-nd", "-q", "--no-check-certificate", f"https://aleph.gutenberg.org/cache/epub/{file_id}/pg{file_id}-h.zip"]
        procs.append(subprocess.Popen(cmd, cwd=DATA_PATH))
        log.info(f"book {file_id}: downloading")

    for p in procs:
        p.wait()

def step_two_processing(file_ids):
    """Segments all files into scenes, creating segment, scenes, and recall folders."""
    with stay_awake():   # process runs long — survive a closed lid
        metadata, books = build_library(data_path=DATA_PATH, recall_path=RECALL_PATH)
        for file_id in file_ids:
            if not Path(DATA_PATH + f"/pg{file_id}-h.zip").is_file(): 
                log.warn(f"book {file_id}: download is not a zip")
                continue
            segment_test(metadata, books, file_id)

def step_three_embedding(file_ids):
    """Enrich + index each book's scenes into the local Qdrant db (test root),
    skipping any id with no scenes json yet (excluded or not segmented)."""
    with stay_awake():   # embed runs long — survive a closed lid
        exist_ids = []
        for file_id in file_ids:
            if not Path(SCENES_PATH + f"/pg{file_id}-s.json").is_file(): continue
            exist_ids.append(file_id)
        embed_test(exist_ids)

def main():
    #step_one_retrieval(FILE_IDS)
    #step_two_processing(FILE_IDS)
    #step_three_embedding(FILE_IDS)
    reset_test(FILE_IDS)
    #search_test()
    pass


if __name__ == "__main__":
    main()