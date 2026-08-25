# FOR CLAUDE — Read path: query the scene vector DB.
# -----------------------------------------------------------------------------
# Import THIS from the app / API. It pulls in only qdrant + fastembed — NO LLM, NO
# segmentation — so the query path stays light and fast. It also OWNS the
# vector-store primitives (collection name, vector config, embedder, point id,
# filters) that the build-once write path (embed.py) imports to index.
#
# By design this config lives here, NOT in storage.py: storage.py centralizes
# plain-JSON file IO, whereas COLLECTION / VECTOR_NAMES / EMBED_MODEL / QDRANT_PATH
# are one cohesive Qdrant contract shared between read and write. Both sides MUST
# agree on them, so they have a single home.
#
# Invariants:
#   * EMBED_MODEL must match the model the index was built with, or scores are junk.
#   * point_id is a stable uuid5 of scene_id, so re-indexing a scene overwrites its
#     existing point instead of creating a duplicate.
#   * bge is asymmetric: only the summary QUERY is prefixed (QUERY_PREFIX); indexed
#     passages and descriptor queries stay raw.
# -----------------------------------------------------------------------------
import uuid
import numpy as np
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding


# --- vector store config (shared with embed.py's indexer) --- #

QDRANT_PATH = "master/qdrant_db"           # local on-disk Qdrant (no server needed)
COLLECTION = "master/scenes"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"     # MUST match the model the index was built with
VECTOR_NAMES = ("summary", "descriptors")  # two named vectors per scene point

# bge query-side instruction prefix (summary path only). Set to "" to A/B without
# re-indexing — passages are untouched, so the index stays valid either way.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# stable per-scene Qdrant id namespace: uuid5(NAMESPACE, scene_id) -> same point
NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "projectalexandria.scenes")


def point_id(scene_id: str) -> str:
    """Stable Qdrant point id for a scene (uuid5, so re-index overwrites)."""
    return str(uuid.uuid5(NAMESPACE, scene_id))


# --- embedder (same model for index + query) --- #

_EMBEDDER = None

def _embedder() -> TextEmbedding:
    """Lazily construct and cache the shared TextEmbedding model."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = TextEmbedding(model_name=EMBED_MODEL)
    return _EMBEDDER


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts into plain float lists."""
    return [v.tolist() for v in _embedder().embed(texts)]


# --- client + filters --- #

def open_client() -> QdrantClient:
    """Open the on-disk Qdrant client (first call downloads the embed model)."""
    return QdrantClient(path=QDRANT_PATH)


def book_filter(book_id: str | None) -> models.Filter | None:
    """Filter that restricts a search to one book, or None for all books."""
    if not book_id:
        return None
    return models.Filter(must=[models.FieldCondition(
        key="book_id", match=models.MatchValue(value=book_id))])


# --- search --- #

def search_summary(client: QdrantClient, summary: str, descriptors: str | None = None,
                   limit: int = 5, flt: models.Filter | None = None,
                   w_summary: float = 0.7):
    """Search by summary (required); descriptors (optional) only rerank within the summary pool.

    Without descriptors: rank by the "summary" vector. With descriptors: summary
    GATES the candidate pool and carries the heavier weight, descriptors rerank
    inside it — so a descriptor-only match can't inject an off-topic scene.
        score = w_summary * summary_cos + (1 - w_summary) * descriptor_cos
    """
    if not summary:
        raise ValueError("search_summary needs a summary")
    sv = embed([QUERY_PREFIX + summary])[0]   # query-side prefix (index stays raw)

    if not descriptors:
        return client.query_points(COLLECTION, query=sv, using="summary", limit=limit,
                                   query_filter=flt, with_payload=True).points

    pool = max(limit * 5, 50)   # summary candidate pool, reranked by descriptors below
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

# --- weighted + negative descriptor search --- #
# Per-descriptor weighting is a QUERY-time operation: nothing extra is stored. Instead
# of embedding one joined "a, b, c" string (search_summary above), each descriptor
# is embedded on its own and combined by weight, so the caller can lean the search
# ("0.7 melancholy, 0.3 eerie") and push AWAY from anti-descriptors by subtraction.

WEIGHT_TOL = 1e-6   # how far a weight list may drift from summing to 1.00


def _unit(v) -> np.ndarray:
    """L2-normalize a vector; a zero vector is returned unchanged (guards divide-by-zero)."""
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n else v


def _check_weights(terms: list[str], weights: list[float], label: str):
    """Validate a (terms, weights) pair: non-empty, equal length, non-negative, sum≈1.00."""
    if not terms:
        raise ValueError(f"{label}: need at least one term")
    if len(terms) != len(weights):
        raise ValueError(f"{label}: {len(terms)} terms but {len(weights)} weights")
    if any(w < 0 for w in weights):
        raise ValueError(f"{label}: weights must be non-negative")
    total = sum(weights)
    if abs(total - 1.0) > WEIGHT_TOL:
        raise ValueError(f"{label}: weights must sum to 1.00 (got {total:.6f})")


def weighted_vector(terms: list[str], weights: list[float]) -> np.ndarray:
    """Weighted centroid of the terms' embeddings, returned as a unit vector.

    Each term is embedded and L2-normalized FIRST, so no single word dominates by raw
    magnitude; the per-term unit vectors are then summed by weight and the result is
    re-normalized. Caller validates lengths/weights via _check_weights.
    """
    vecs = embed(terms)                         # one batched embed call for all terms
    acc = np.zeros(len(vecs[0]), dtype=np.float32)
    for v, w in zip(vecs, weights):
        acc += w * _unit(v)
    return _unit(acc)


def search_weighted_descriptors(
    client: QdrantClient,
    descriptors: list[str], 
    weights: list[float] = None,
    *,
    anti_descriptors: list[str] | None = None,
    anti_weights: list[float] | None = None,
    anti_strength: float = 1.0,
    limit: int = 5,
    flt: models.Filter | None = None,
):
    """Descriptor search with per-descriptor weights and optional anti-descriptors.

    `descriptors`/`weights` are equal-length; `weights` are the writer's percentages and
    MUST sum to 1.00. The query is a weighted centroid of the INDIVIDUAL descriptor
    embeddings (not one joined string), so "0.7 melancholy, 0.3 eerie" leans the search
    toward melancholy while still feeling the eerie pull.

    If `anti_descriptors`/`anti_weights` are supplied (same contract — equal length, sum
    to 1.00, given together), their weighted centroid is SUBTRACTED from the query,
    tilting results away from that flavor with no extra positive descriptor needed.
    `anti_strength` scales the push (1.0 == equal weight to the positive direction).
    Note: subtraction TILTS in cosine space, it does not hard-exclude; for a clean cut
    on the controlled tags use a payload `must_not` filter instead.

    Returns Qdrant ScoredPoints (payload == the full scene record), ranked best-first.
    """
    if not weights: weights = [1.00 / len(descriptors) for _ in descriptors]
    
    _check_weights(descriptors, weights, "descriptors")
    q = weighted_vector(descriptors, weights)

    if anti_descriptors or anti_weights:
        if not (anti_descriptors and anti_weights):
            raise ValueError("anti_descriptors and anti_weights must be given together")
        _check_weights(anti_descriptors, anti_weights, "anti_descriptors")
        q = q - anti_strength * weighted_vector(anti_descriptors, anti_weights)
        if float(np.linalg.norm(q)) < 1e-8:
            raise ValueError("positive and anti descriptors cancel out; lower anti_strength")
        q = _unit(q)

    return client.query_points(
        COLLECTION, query=q.tolist(), using="descriptors",
        limit=limit, query_filter=flt, with_payload=True,
    ).points


# --- combined: summary gate + weighted-descriptor rerank --- #

def search_combined(
    client: QdrantClient,
    summary: str,
    descriptors: list[str],
    weights: list[float] | None = None,
    *,
    anti_descriptors: list[str] | None = None,
    anti_weights: list[float] | None = None,
    anti_strength: float = 1.0,
    limit: int = 5,
    flt: models.Filter | None = None,
    w_summary: float = 0.7,
):
    """Fused search using BOTH named vectors: the summary GATES + weights the candidate
    pool, and a WEIGHTED-descriptor centroid reranks within it.

    Joins search_summary's precision gate (a stray descriptor can't drag in an off-topic
    scene) with search_weighted_descriptors' per-term weighting + optional anti-
    descriptors — instead of the single joined descriptor string that search_summary
    takes. `descriptors`/`weights` follow the same contract as search_weighted_descriptors
    (equal length, weights sum to 1.00; equal weights when omitted).
        score = w_summary * summary_cos + (1 - w_summary) * descriptor_cos
    Returns Qdrant ScoredPoints (payload == full scene record), best-first.
    """
    if not summary:
        raise ValueError("search_combined needs a summary")
    if not descriptors:
        raise ValueError("search_combined needs descriptors")

    # descriptor side: weighted centroid of the INDIVIDUAL descriptor embeddings
    if not weights:
        weights = [1.0 / len(descriptors) for _ in descriptors]
    _check_weights(descriptors, weights, "descriptors")
    dv = weighted_vector(descriptors, weights)
    if anti_descriptors or anti_weights:
        if not (anti_descriptors and anti_weights):
            raise ValueError("anti_descriptors and anti_weights must be given together")
        _check_weights(anti_descriptors, anti_weights, "anti_descriptors")
        dv = dv - anti_strength * weighted_vector(anti_descriptors, anti_weights)
        if float(np.linalg.norm(dv)) < 1e-8:
            raise ValueError("positive and anti descriptors cancel out; lower anti_strength")
        dv = _unit(dv)

    # summary side: gate the candidate pool (query-side prefix; index stays raw)
    sv = embed([QUERY_PREFIX + summary])[0]
    pool = max(limit * 5, 50)
    cands = client.query_points(COLLECTION, query=sv, using="summary", limit=pool,
                                query_filter=flt, with_payload=True).points
    if not cands:
        return cands

    # weighted-descriptor cosine for exactly those candidates (restrict by their ids)
    ids = [c.id for c in cands]
    dhits = client.query_points(
        COLLECTION, query=dv.tolist(), using="descriptors", limit=len(ids),
        query_filter=models.Filter(must=[models.HasIdCondition(has_id=ids)]),
        with_payload=False,
    ).points
    dscore = {h.id: h.score for h in dhits}

    for c in cands:
        c.score = w_summary * c.score + (1 - w_summary) * dscore.get(c.id, 0.0)
    cands.sort(key=lambda c: c.score, reverse=True)
    return cands[:limit]
