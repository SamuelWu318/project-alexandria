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

from data import build_library
from process import SceneBreaker, scenes_to_records, _load_checkpoint, _save_checkpoint
from storage import TEST_PATH, SCENES_PATH, CHECKPOINT_DIR, write_json
import search


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


def _show(hits):
    """Print each hit: score, scene id, flavor tags, title, summary, descriptors."""
    for h in hits:
        p = h.payload
        print(f"  {round(h.score, 3)}  {p['scene_id']}  [{p.get('dominant_tone')}"
              f"/{p.get('intensity')}/{p.get('arc')}]  {p.get('scene_title')}")
        print(f"     {p.get('summary')}  << {p.get('descriptors')}")


def search_test(book_id: str = "1727", limit: int = 2):
    """Run summary-only, summary+descriptors (weighted), then pure-descriptor searches."""
    client = search.open_client()
    flt = search.book_filter(book_id)

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


# --- payload dump (was data.main) --- #

def payload_dump_test():
    """Dump every book's chunk payloads to TEST_PATH/pg{code}-p.json for inspection."""
    _, books = build_library()
    for book in books.values():
        print(book.file_code)
        book.to_json(TEST_PATH)


# --- segmentation run (was process.main) --- #

def segment_test(desired: str = "1727"):
    """Segment ONE book with the LLM (resumable via checkpoints) and write its scenes json."""
    metadata, books = build_library()
    sb = SceneBreaker()

    desired_book = []
    book = books[desired]
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

    # collect kept scenes in order; count paragraphs dropped as noise
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

    out_path = f"{SCENES_PATH}/pg{desired}-s.json"
    write_json(out_path, records)
    print(f"wrote {len(records)} scenes to {out_path}")

    # book fully saved: drop its checkpoints, no longer needed for resume
    shutil.rmtree(ckpt_dir, ignore_errors=True)


def main():
    """Default run: the Qdrant search test."""
    search_test()


if __name__ == "__main__":
    main()
