# read path: query the scene vector DB.
# -----------------------------------------------------------------------------
# Import THIS from the app / API. It pulls in only qdrant + fastembed — NO LLM,
# NO segmentation — so the query path stays light and fast. It also owns the
# vector-store primitives (collection config, embedder, id, filters) that the
# build-once write path (data.py / process.py / embed.py) imports to index.
# -----------------------------------------------------------------------------
import uuid
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding


# --- vector store config (shared with embed.py's indexer) --- #

QDRANT_PATH = "master/qdrant_db"          # local on-disk Qdrant (no server needed)
COLLECTION = "master/scenes"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"    # MUST match the model the index was built with
VECTOR_NAMES = ("summary", "descriptors")  # two named vectors per scene point

# stable per-scene Qdrant id: uuid5(NAMESPACE, scene_id) -> same scene, same point
NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "projectalexandria.scenes")


def point_id(scene_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, scene_id))


# --- embedder (same model for index + query) --- #

_EMBEDDER = None

def _embedder() -> TextEmbedding:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = TextEmbedding(model_name=EMBED_MODEL)
    return _EMBEDDER


def embed(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in _embedder().embed(texts)]


# --- client + filters --- #

def open_client() -> QdrantClient:
    return QdrantClient(path=QDRANT_PATH)


def book_filter(book_id: str | None) -> models.Filter | None:
    # restrict search to one book (collection is multi-book once more are indexed)
    if not book_id:
        return None
    return models.Filter(must=[models.FieldCondition(
        key="book_id", match=models.MatchValue(value=book_id))])


# --- search --- #

def search_summary(client: QdrantClient, summary: str, descriptors: str | None = None,
                   limit: int = 5, flt: models.Filter | None = None,
                   w_summary: float = 0.7):
    # summary REQUIRED, descriptors OPTIONAL. Without descriptors: rank by the
    # "summary" vector. With descriptors: summary GATES the candidate pool (and
    # carries the heavier weight); descriptors only rerank within it, so a
    # descriptor-only match can't inject an off-topic scene.
    #   score = w_summary * summary_cos + (1 - w_summary) * descriptor_cos
    if not summary:
        raise ValueError("search_summary needs a summary")
    sv = embed([summary])[0]

    if not descriptors:
        return client.query_points(COLLECTION, query=sv, using="summary", limit=limit,
                                   query_filter=flt, with_payload=True).points

    pool = max(limit * 5, 50)   # candidate pool from summary before descriptor rerank
    cands = client.query_points(COLLECTION, query=sv, using="summary", limit=pool,
                                query_filter=flt, with_payload=True).points
    if not cands:
        return cands

    # descriptor cosine for exactly those candidates (restrict by their ids)
    dv = embed([descriptors])[0]
    ids = [c.id for c in cands]
    dhits = client.query_points(
        COLLECTION, query=dv, using="descriptors", limit=len(ids),
        query_filter=models.Filter(must=[models.HasIdCondition(has_id=ids)]),
        with_payload=False,
    ).points
    dscore = {h.id: h.score for h in dhits}

    for c in cands:
        c.score = w_summary * c.score + (1 - w_summary) * dscore.get(c.id, 0.0)
    cands.sort(key=lambda c: c.score, reverse=True)
    return cands[:limit]


def search_descriptors(client: QdrantClient, descriptors: str, limit: int = 5,
                       flt: models.Filter | None = None):
    # pure descriptor-vector search (vibe words only, no summary).
    if not descriptors:
        raise ValueError("search_descriptors needs descriptors")
    dv = embed([descriptors])[0]
    return client.query_points(COLLECTION, query=dv, using="descriptors", limit=limit,
                               query_filter=flt, with_payload=True).points
