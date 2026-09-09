import warnings
from pathlib import Path, PurePath
from qdrant_client import QdrantClient, models

# derive is the ONE feature-file door index leans on: the idempotent Stage-3b pass that fills every
# `source:"derived"` field (svos / the four facet lists / vdi_curve / prose_register / dialogue_ratio /
# arc). Called as a pre-embed safety net so a scene is never indexed with an un-derived frame/curve (PLAN §8).
from derive import derive_records
# vector-store contract — the ONE home of the Qdrant primitives, shared with search.py's read path.
from utils.vectorstore import (COLLECTION, VECTOR_NAMES, MULTIVECTOR_NAMES, SUBJECT_PATHS_FIELD,
                               embed, open_client, point_id, _as_terms)
from utils import relational   # SQLite scene mirror: the exact-match / navigation store beside the vectors
from utils import subjects     # subject-path expansion for the filterable payload label
from utils import SrcPaths, log, read_json, write_json

# ---- Stage 3c: indexing (build the stores from enriched + derived scene records) ----
# in:  scenes/pg{code}-s.json (records after enrich.py + derive.py — LLM fields + the derived payload filled)
# out: a Qdrant collection of scene points (SEVEN named vectors: summary + descriptors single, svos +
#      subject/verb/object/setting multivector) + a SQLite mirror + the book subject trie.
# One point per ENRICHED scene (payload == the full record, stamped with subject_paths + carrying the
# hard/soft facets pov/tense/prose_register/dialogue_ratio/vdi_curve); EVERY record (enriched or not)
# mirrors into SQLite. Rebuild is EXPLICIT — importing this module has NO side effects; call index_scenes().
# The vector-store contract (collection / named vectors / embedder / stable id) lives in utils/vectorstore.py.

# ---- qdrant collection config (mechanical; the named-vector set is derived from the registry) ----

# ** LOCKED **
# VectorParams for one named vector — a MAX_SIM multivector for a list field, a single vector otherwise.
def _vec_params(name: str, dim: int) -> models.VectorParams:
    mv = (models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM)
          if name in MULTIVECTOR_NAMES else None)
    return models.VectorParams(size=dim, distance=models.Distance.COSINE, multivector_config=mv)


# ** LOCKED **
# Ensure the named-vector collection exists; drop + rebuild if the vector NAME set or any field's multivector-ness disagrees with the registry.
def _ensure_collection(client: QdrantClient, dim: int) -> None:
    want = {n: _vec_params(n, dim) for n in VECTOR_NAMES}
    if client.collection_exists(COLLECTION):
        cfg = client.get_collection(COLLECTION).config.params.vectors
        names_ok = isinstance(cfg, dict) and set(cfg) == set(VECTOR_NAMES)
        mv_ok = names_ok and all(
            (getattr(cfg[n], "multivector_config", None) is not None) == (n in MULTIVECTOR_NAMES)
            for n in VECTOR_NAMES)
        if names_ok and mv_ok:
            return
        log.warn(f"'{COLLECTION}' vector config stale (name/multivector mismatch) — dropping + rebuilding")
        client.delete_collection(COLLECTION)   # stale vector set -> rebuild from scratch
    client.create_collection(COLLECTION, vectors_config=want)
    log.info(f"built '{COLLECTION}' with {len(want)} vectors: {', '.join(want)}")


# ** LOCKED **  ** MAIN ** — tests.backfill_subject_paths ensures this index too
# Keyword payload index on `subject_paths` so a branch filter is an inverted-index lookup (inert in local Qdrant, live on server; idempotent).
def _ensure_subject_index(client: QdrantClient) -> None:
    try:
        with warnings.catch_warnings():          # local Qdrant warns the index is inert — harmless
            warnings.simplefilter("ignore")
            client.create_payload_index(COLLECTION, field_name=SUBJECT_PATHS_FIELD,
                                        field_schema=models.PayloadSchemaType.KEYWORD)
    except Exception as e:                        # never let index setup break an index run
        log.warn(f"subject_paths payload index: {type(e).__name__}: {e}")


# Embed one multivector field for every ready scene -> a per-scene MATRIX (an empty field falls back to a 1-row matrix from the summary).
def _multivector_field(ready: list[dict], field: str) -> list[list[list[float]]]:
    per_terms = [(_as_terms(r.get(field)) or [r["summary"]]) for r in ready]   # terms per scene (summary fallback)
    flat = [t for terms in per_terms for t in terms]
    vecs = embed(flat) if flat else []                         # vectorstore: one batched embed over all terms
    out, k = [], 0
    for terms in per_terms:
        out.append(vecs[k:k + len(terms)])                     # regroup the flat vectors back per scene
        k += len(terms)
    return out


# ** MAIN ** — tests.embed_test + index_scenes build points here
# Index one book's records: derive the payload (safety net), mirror EVERY record into SQLite, then embed the
# seven named vectors for the ENRICHED scenes as one point each. payload == the full record; id is the stable
# uuid5 (a re-run overwrites its point — no dupes).
def index_records(client: QdrantClient, records: list[dict],
                  conn: "relational.sqlite3.Connection | None" = None) -> None:
    derive_records(records)          # derive.py: fill svos/facets/vdi_curve/... in place (idempotent — PLAN §8 safety net)

    if conn is not None:
        n = relational.sql_upsert(conn, records)               # SQLite mirror: EVERY record, enriched or not
        log.info(f"mirrored {n} rows into the relational store")

    ready = [r for r in records if r.get("summary")]           # only enriched scenes become vector points
    if not ready:
        log.skip("no enriched summaries to index")
        return

    sum_vecs = embed([r["summary"] for r in ready])            # the holistic summary vector
    # descriptors are 3-5 adjectives (schema-guaranteed); join to one vibe string (summary fallback).
    desc_vecs = embed([", ".join(r.get("descriptors") or []) or r["summary"] for r in ready])
    # the five multivector fields: one MATRIX per scene (a vector per term), scored by MAX_SIM.
    mv = {f: _multivector_field(ready, f) for f in MULTIVECTOR_NAMES}

    # stamp the filterable subject label onto each payload (right-anchored prefixes of the book's subjects).
    for r in ready:
        subj = (r.get("book_metadata") or {}).get("Subjects") or []
        r[SUBJECT_PATHS_FIELD] = subjects.suffixes(subj)       # subjects.py: every branch prefix -> one keyword match

    _ensure_collection(client, len(sum_vecs[0]))               # (re)build the collection if the vector set drifted
    _ensure_subject_index(client)                              # keyword index on subject_paths
    points = [
        models.PointStruct(
            id=point_id(r["scene_id"]),                        # vectorstore: stable uuid5 -> overwrite on re-run
            vector={"summary": sum_vecs[i], "descriptors": desc_vecs[i],
                    **{f: mv[f][i] for f in MULTIVECTOR_NAMES}},
            payload=r)
        for i, r in enumerate(ready)
    ]
    client.upsert(COLLECTION, points=points)
    log.info(f"indexed {len(ready)} points into '{COLLECTION}'")


# ---- rebuild driver: (re)build the stores from the enriched + derived scene jsons ----

# ** MAIN ** — tests rebuilds the stores here (run after wiping scenes.db + the qdrant dir)
# Rebuild the stores from the scene jsons: per book derive + persist the payload, then index (seven named
# vectors into Qdrant + the SQLite mirror), one client/conn for the whole run. file_ids=None = every
# pg*-s.json. Returns files indexed.
def index_scenes(file_ids=None) -> int:
    if file_ids:
        files = [Path(f"{SrcPaths.SCENES_DIR}/pg{c}-s.json") for c in file_ids if c]
    else:
        files = sorted(Path(SrcPaths.SCENES_DIR).glob("pg*-s.json"))

    client = open_client()                 # vectorstore: the on-disk Qdrant client (single-process)
    conn = relational.open_db()            # relational: scenes.db, created/migrated on open
    n = 0
    try:
        for f in files:
            if not f.exists():
                log.skip(f"index: skip {PurePath(f).name} (missing)")
                continue
            records = read_json(f, [])
            if not records:
                log.skip(f"index: skip {PurePath(f).name} (empty)")
                continue
            derive_records(records)        # derive.py: fill the derived payload in place ...
            write_json(f, records)         # ... and persist it BEFORE the subject_paths stamp (keeps the json clean)
            index_records(client, records, conn)   # SQLite mirror + named vectors, in lockstep (re-derives as its safety net)
            n += 1
        log.done(f"indexed {n} scene files -> '{COLLECTION}' + relational store")
        return n
    finally:
        conn.close()
        client.close()
