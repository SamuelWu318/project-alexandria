from pathlib import Path
import subprocess, os, sys, time, re, contextlib, zipfile

from data import build_library, ensure_book
from process import scenes_to_records, segment_book, presegmentation_gate
from embed import enrich_file, index_records
import embed
import search

from utils import write_json, read_json, relational, subjects, log, SCHEMA_VERSION, SrcPaths, llm, llm_ready_up
from qdrant_client import QdrantClient

# ---- interactive test / smoke harness (run by hand, not pytest) ----
# The manually-run drivers that exercise the whole pipeline: step_one/two/three_retrieval/processing/embedding
# (download -> segment -> enrich + index), plus search_test / manual_search (read path) and the subject-tree
# tests. Each ** ENTRY ** function is one you run yourself; their internal calls are annotated inline.

# book-level embed gate: skip a book when non-prose "other" (poetry/plays) exceeds this fraction of its non-noise text
OTHER_SKIP_RATIO = 0.70

# ---- test book ids (uncomment a line to include that book) ----

FILE_IDS = [
    "64317",    # great gatsby
    #"71865",    # mrs dalloway
    "4300",     # ulysses
    #"2814",     # dubliners
    "215",      # call of the wild
    #"55",       # wizard of oz
    #"73",       # red badge of courage
    #"75201",    # a farewell to arms
    "2701",     # moby dick
    "1342",     # pride and prejudice
    "84",       # frankenstein
    #"11",       # alice in wonderland
    "1661",     # sherlock holmes
    #"345",      # dracula
    "98",       # tale of two cities
    #"43",       # jekyll and hyde
    #"2554",     # crime and punishment
    #"8492",     # the king in yellow
    "2147",     # edgar allan poe 1
    #"2148",     # edgar allan poe 2
    #"175",      # phantom of the opera
    #"68283",    # the call of cthulhu
    "103"       # around the world in eighty days
]

# ---- query sets (how a writer searches; each maps to a famous scene in a FILE_IDS book) ----

# general, plot-free scene descriptions -> summary-only retrieval.
TEST_QUERIES = [
    "A man is inspired by a wealthy host's extravagant party.",
    "An evil witch is killed.",
    "A young soldier panics and flees from his first taste of battle.",
    "A man experiences a traumatic fight and narrowly wins or escapes."
]

# (summary, descriptors-list) pairs -> summary channel + weighted-descriptor centroid, RRF-merged.
COMBINED_QUERIES = [
    ("A crowd of glittering strangers drifts through a wealthy host's extravagant summer party.",
     ["festive", "glamorous", "hollow"]),                       # Gatsby (64317)
    ("A wounded officer and a nurse fall in love in a wartime hospital.",
     ["tender", "bittersweet", "yearning"]),                    # A Farewell to Arms (75201)
]

# (summary, moment-sentence list) pairs -> what-happens two-channel search (summary vs svos, max).
MOMENTS_QUERIES = [
    ("A young soldier breaks and runs from his first battle.",
     ["A terrified soldier flees the battlefield.", "The soldier throws down his rifle and runs."]),  # Red Badge (73)
    ("An evil witch is destroyed by water.",
     ["A girl hurls a bucket of water over a witch.", "The witch melts into nothing."]),              # Wizard of Oz (55)
]

# pure-descriptor (vibe-only) queries — each a descriptor list.
DESCRIPTOR_QUERIES = [
    ["festive", "glamorous", "restless"],           # Gatsby-party glitter (64317)
    ["claustrophobic", "absurd", "dehumanizing"],
    ["chaotic", "terrifying", "cowardly"],          # Red Badge battle-panic (73)
]

# ---- subject tree: SQL table + subject_paths payload ----

# ** ENTRY ** — build the subject table from metadata, then walk + recall down a branch it discovers.
def subject_sql_test():
    conn = subjects.open_db(SrcPaths.DB_PATH)
    try:
        n = subjects.build_from_recall(conn, SrcPaths.RECALL_DIR)   # table from metadata.json alone
        log.step(f"built book_subject_path: {n} rows in {SrcPaths.DB_PATH}")

        # walk a path DOWN the tree the table generated — nothing hardcoded
        path = []
        while True:
            kids = subjects.children(conn, path)                    # browse one level
            log.info(f"children of {path or '[root]'}: {kids[:8]}{' ...' if len(kids) > 8 else ''}")
            if not kids:
                break
            path.append("Fiction" if not path and "Fiction" in kids else kids[0])

        # recall at the deepest branch reached
        parent = path[:-1]                     # last hop had no children; step back to a real branch
        log.step(f"RECALL  in={parent}")
        log.done(f"  branch slice -> {subjects.branch(conn, parent)}")   # child terms + book ids
    finally:
        conn.close()
    return


# ** ENTRY ** — one-off migration: stamp `subject_paths` onto EXISTING indexed points WITHOUT re-embedding.
def backfill_subject_paths(file_ids=None):
    files = ([Path(f"{SrcPaths.SCENES_DIR}/pg{c}-s.json") for c in file_ids if c] if file_ids
             else sorted(Path(SrcPaths.SCENES_DIR).glob("pg*-s.json")))
    client = QdrantClient(path=str(SrcPaths.QDRANT_DIR))
    try:
        embed._ensure_subject_index(client)          # inert locally, live on server Qdrant
        total = 0
        for f in files:
            if not f.exists():
                continue
            recs = read_json(str(f), [])
            subj = next((r.get("book_metadata") for r in recs if r.get("book_metadata")), None) or {}
            paths = subjects.suffixes(subj.get("Subjects") or [])   # right-anchored branch labels
            ids = [embed._point_id(r["scene_id"]) for r in recs
                   if r.get("summary") and r.get("scene_id")]        # only enriched scenes are indexed
            if not ids or not paths:
                log.skip(f"{f.name}: no indexed points or no subjects — skip")
                continue
            client.set_payload(embed.COLLECTION,
                               payload={embed.SUBJECT_PATHS_FIELD: paths}, points=ids)   # payload-only stamp
            total += len(ids)
            log.info(f"{f.name}: stamped {len(ids)} points with {len(paths)} labels")
        log.done(f"backfilled subject_paths onto {total} points across {len(files)} books")
    finally:
        client.close()
    return


# ---- payload dump (was data.main) ----

# ** ENTRY ** — dump every book's chunk payloads to SEGMENTS_DIR/pg{code}-p.json for inspection.
def payload_dump_test():
    metadata, _ = build_library(data_path=SrcPaths.DATA_DIR, recall_path=SrcPaths.RECALL_DIR)   # metadata cache
    for code in metadata:                        # books are lazy now — parse/load each shard
        book = ensure_book(code, SrcPaths.DATA_DIR, SrcPaths.RECALL_DIR, metadata[code].get("Title"))
        log.info(f"dumping payloads for book {book.file_code}")
        book.to_json(SrcPaths.SEGMENTS_DIR)      # lossy chunk payloads


# ---- segmentation run (was process.main) ----

# ** ENTRY ** — segment ONE book with the LLM (resumable) and write its scenes json.
def segment_test(metadata: dict, books: dict, desired: str):
    # do not segment if a completed scenes file already exists
    if (SrcPaths.SCENES_DIR / f"pg{desired}-s.json").is_file():
        log.skip(f"book {desired} already in scenes — skip segmentation")
        return

    # build_library only carries books with a usable source; a missing/invalid zip is absent
    # from metadata, so skip rather than KeyError.
    if desired not in metadata:
        log.warn(f"book {desired}: no metadata (missing / invalid source) — skip")

    log.step(f"segmenting book {desired}")

    desired_book = []
    md = metadata[desired]
    # recall is lazy + per-book: ensure_book parses + writes the shard if it does not exist.
    if desired not in books:
        books[desired] = ensure_book(desired, SrcPaths.DATA_DIR, SrcPaths.RECALL_DIR, md.get("Title"))
    book = books[desired]

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

    # stitch + flatten the kept scenes into ingest-ready records
    records = scenes_to_records(desired, desired_book, books[desired], metadata[desired])

    out_path = f"{SrcPaths.SCENES_DIR}/pg{desired}-s.json"
    write_json(out_path, records)
    log.done(f"book {desired}: recorded {len(records)} scenes -> {out_path}")


# ---- enrichment + indexing run (was embed.main) ----

# Load the test-root {book_id: status} map; missing file / key / null all mean 'not done'.
def _load_status() -> dict:
    return read_json(SrcPaths.STATUS_PATH, {})


# Persist one book's completion status so the next embed_test run skips it.
def _mark_status(code: str, value="completed"):
    status = _load_status()
    status[code] = value
    write_json(SrcPaths.STATUS_PATH, status)


# ** ENTRY ** — enrich each scenes json and index it into the local Qdrant db (skips books marked 'completed').
def embed_test(file_ids=None):
    if file_ids:
        files = [Path(f"{SrcPaths.SCENES_DIR}/pg{c}-s.json") for c in file_ids if c]
    else:
        files = sorted(Path(SrcPaths.SCENES_DIR).glob("pg*-s.json"))

    client = QdrantClient(path=str(SrcPaths.QDRANT_DIR))   # on-disk local db, auto-created
    conn = relational.open_db(SrcPaths.DB_PATH)        # SQLite mirror, created/migrated on open
    subjects.ensure_table(conn)                        # sibling subject table in the same scenes.db
    subjects.build_from_recall(conn, SrcPaths.RECALL_DIR)  # book-level subject trie, from metadata alone
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


# ---- search ----

# Print each hit: score, scene id, flavor tags, title, summary, descriptors.
def _show(hits):
    for h in hits:
        p = h.payload
        print(f"  {round(h.score, 3)}  {p['scene_id']}  [{p.get('dominant_tone')}"
              f"/{p.get('intensity')}/{p.get('arc')}]  {p.get('scene_title')}")
        print(f"     {p.get('summary')}  << {p.get('descriptors')}")


# ** ENTRY ** — smoke the unified search(): summary-only, +moments, +descriptors, then pure-descriptor.
def search_test(book_id: str = None, limit: int = 2):
    client = QdrantClient(path=str(SrcPaths.QDRANT_DIR))   # read the test db that embed_test wrote

    flt = search.book_filter(book_id)                      # optional one-book pre-filter

    try:
        log.step("SUMMARY ONLY")
        for q in TEST_QUERIES:
            print(f"\nQUERY: {q}")
            _show(search.search(client, summary=q, limit=limit, flt=flt))

        log.step("SUMMARY + MOMENTS (what-happens: summary vs svos, greatest single match)")
        for summ, moments in MOMENTS_QUERIES:
            print(f"\nQUERY: {summ!r}  +  moments {moments!r}")
            _show(search.search(client, summary=summ, moments=moments, limit=limit, flt=flt))

        log.step("SUMMARY + DESCRIPTORS (RRF merge)")
        for summ, desc in COMBINED_QUERIES:
            print(f"\nQUERY: {summ!r}  +  descriptors {desc!r}")
            _show(search.search(client, summary=summ, descriptors=desc, limit=limit, flt=flt))

        log.step("DESCRIPTORS ONLY")
        for descriptors in DESCRIPTOR_QUERIES:
            print(f"\nQUERY: descriptors {descriptors!r}")
            _show(search.search(client, descriptors=descriptors, limit=limit, flt=flt))
    finally:
        client.close()


# ---- keep-awake (macOS lid-close survival) ----

# Toggle macOS lid-close sleep via `sudo pmset -b disablesleep <value>` (needs admin; safe warn-and-continue on failure).
def _pmset_disablesleep(value: int) -> bool:
    try:
        subprocess.run(["sudo", "pmset", "-b", "disablesleep", str(value)], check=True)
        return True
    except (FileNotFoundError, OSError, subprocess.CalledProcessError) as e:
        log.warn(f"pmset disablesleep {value} failed ({e}) — lid-close sleep unchanged")
        return False


# Keep the Mac awake for the whole block (pmset + caffeinate), restoring normal sleep on exit (incl. Ctrl-C).
@contextlib.contextmanager
def stay_awake():
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


# ---- pipeline steps ----

# ** ENTRY ** — step 1: download each book's -h.zip into DATA_DIR (parallel wgets, then validate + rebuild bad ones).
def step_one_retrieval(file_ids, force=False):
    if not file_ids: return

    Path(SrcPaths.DATA_DIR).mkdir(parents=True, exist_ok=True)
    procs = []

    for file_id in file_ids:
        # skip regardless if file_id does not exist; skip only if file exists and not force
        if not file_id or ((SrcPaths.DATA_DIR / f"pg{file_id}-h.zip").is_file() and not force): continue

        current_zip = (SrcPaths.DATA_DIR / f"pg{file_id}-h.zip")
        if current_zip.is_file(): current_zip.unlink()

        cmd = ["wget", "-nc", "-nd", "-q", "--no-check-certificate", f"https://aleph.gutenberg.org/cache/epub/{file_id}/pg{file_id}-h.zip"]
        procs.append(subprocess.Popen(cmd, cwd=SrcPaths.DATA_DIR))   # launch the download
        log.info(f"book {file_id}: downloading")
        time.sleep(2)

    # validation
    for file_id in file_ids:
        rebuild = []
        if not zipfile.is_zipfile(SrcPaths.DATA_DIR / f"pg{file_id}-h.zip"):
            log.warn(f"book {file_id}: missing or invalid zip — rebuild")
            rebuild.append(file_id)
    step_one_retrieval(rebuild, force=True)      # re-download the bad ones

    for p in procs:
        p.wait()                                 # block until every download finishes

# ** ENTRY ** — step 2: segment every downloaded book into scenes (build library, then segment_test each).
def step_two_processing(file_ids):
    if not llm_ready_up(): sys.exit("LLM issue")     # fail fast if the LLM is unreachable

    with stay_awake():   # process runs long — survive a closed lid
        metadata, books = build_library(data_path=SrcPaths.DATA_DIR, recall_path=SrcPaths.RECALL_DIR)
        for file_id in file_ids:
            # is_zipfile is False for BOTH a missing file and a corrupt one — the same books
            # build_library skipped, so this keeps segment_test's metadata lookup from KeyError-ing.
            if not zipfile.is_zipfile(SrcPaths.DATA_DIR / f"pg{file_id}-h.zip"):
                log.warn(f"book {file_id}: missing or invalid zip — skip")
            segment_test(metadata, books, file_id)   # segment one book

# ** ENTRY ** — step 3: enrich + index each book that has a scenes json.
def step_three_embedding(file_ids):
    with stay_awake():   # embed runs long — survive a closed lid
        exist_ids = []
        for file_id in file_ids:
            if not (SrcPaths.SCENES_DIR / f"pg{file_id}-s.json").is_file(): continue
            exist_ids.append(file_id)
        embed_test(exist_ids)                    # enrich + index the ones present


# ** ENTRY ** — hand-driven read path: run the unified search() over a manual summary/moments/descriptors and print hits.
def manual_search(summary: str = "", moments=None, descriptors=None,
                  limit: int = 5, book_id: str = None):
    log.step(f"SEARCH  summary={summary!r}  moments={moments!r}  descriptors={descriptors!r}")
    client = QdrantClient(path=str(SrcPaths.QDRANT_DIR))
    try:
        hits = search.search(client, summary=summary or None, moments=moments,
                             descriptors=descriptors, limit=limit,
                             flt=search.book_filter(book_id))
        _show(hits)
    finally:
        client.close()


# ** ENTRY ** — full pipeline: download -> segment -> enrich/index -> subject-tree smoke.
def main():
    #step_one_retrieval(FILE_IDS)         # download
    #step_two_processing(FILE_IDS)        # segment
    step_three_embedding(FILE_IDS)       # enrich + index
    subject_sql_test()                   # subject-tree smoke
    #search_test()
    pass


if __name__ == "__main__":
    main()
