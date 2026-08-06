# ad-hoc test / smoke harness
# -----------------------------------------------------------------------------
# main() runs ONLY the Qdrant search test. The other two methods are the old
# driver/leftover mains pulled out of data.py (payload dump) and process.py
# (segmentation run) — call them by hand when needed.
# -----------------------------------------------------------------------------
import json, threading, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from data import build_library, TEST_PATH
from process import (SceneBreaker, scenes_to_records, _load_checkpoint,
                     _save_checkpoint, CHECKPOINT_DIR, SCENES_PATH)
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
    "cunning, triumphant, clever",
]


def _show(hits):
    for h in hits:
        p = h.payload
        print(f"  {round(h.score, 3)}  {p['scene_id']}  [{p.get('dominant_tone')}"
              f"/{p.get('intensity')}/{p.get('arc')}]  {p.get('scene_title')}")
        print(f"     {p.get('summary')}  << {p.get('descriptors')}")


def search_test(book_id: str = "1727", limit: int = 2):
    # summary-only, then summary+descriptors (weighted rerank), then pure descriptors.
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
    # dump every book's chunk payloads to TEST_PATH/pg{code}-p.json for inspection.
    _, books = build_library()
    for book in books.values():
        print(book.file_code)
        book.to_json(TEST_PATH)


# --- segmentation run (was process.main) --- #

def segment_test(desired: str = "1727"):
    # segment ONE book with the LLM (resumable via checkpoints), write its scenes json.
    metadata, books = build_library()
    sb = SceneBreaker()

    desired_book = []
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


def main():
    search_test()


if __name__ == "__main__":
    main()
