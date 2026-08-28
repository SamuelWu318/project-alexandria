# FOR CLAUDE — Read path: query the scene vector DB.
# -----------------------------------------------------------------------------
# Import THIS from the app / API. It pulls in only qdrant + fastembed — NO LLM, NO
# segmentation — so the query path stays light and fast. It also OWNS the
# vector-store primitives (COLLECTION name, vector config, embedder, point id,
# filters) that the build-once write path (embed.py) imports to index.
#
# By design this config lives here, NOT in storage.py: storage.py centralizes
# plain-JSON file IO, whereas COLLECTION / VECTOR_NAMES / EMBED_MODEL / QDRANT_DIR
# are one cohesive Qdrant contract shared between read and write. Both sides MUST
# agree on them, so they have a single home.
#
# Invariants:
#   * EMBED_MODEL must match the model the index was built with, or scores are junk.
#   * point_id is a stable uuid5 of scene_id, so re-indexing a scene overwrites its
#     existing point instead of creating a duplicate.
#   * bge is asymmetric: the summary and frame-phrase QUERIES are prefixed (QUERY_PREFIX);
#     indexed passages and descriptor queries stay raw.
# -----------------------------------------------------------------------------
import uuid
import numpy as np
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from utils import SrcPaths, schema # scene-record registry (single source of truth) — drives the named vectors


# --- vector store config (shared with embed.py's indexer) --- #

# Qdrant collection name — the read/write join key: embed.py writes points here,
# search.py queries here. Deliberately DECOUPLED from SrcPaths.SCENES_DIR (the on-disk
# scenes/ directory): a collection name is a store identifier, not a filesystem path.
COLLECTION = "scenes"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"     # MUST match the model the index was built with
# named vectors == the registry's vector:true fields (holistic summary + flavor descriptors
# + the decomposed frame). Edit scene_schema.json to add/drop a vector, then re-index.
VECTOR_NAMES = schema.VECTOR_NAMES
# multivector:true fields (subject/verb/object) store a LIST of per-term vectors and score
# by MAX_SIM: for a query term the field's score is the MAX cosine over the scene's terms
# (max-pooling). A specific query locks onto a specific facet; a general one onto a general
# facet — neither averaged away. These fields MUST be queried with a matrix (list of
# vectors), even a 1-row one — a single flat vector is rejected by the multivector index.
MULTIVECTOR_NAMES = frozenset(schema.MULTIVECTOR_NAMES)

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
    return QdrantClient(path=str(SrcPaths.QDRANT_DIR))


def book_filter(book_id: str | None) -> models.Filter | None:
    """Filter that restricts a search to one book, or None for all books."""
    if not book_id:
        return None
    return models.Filter(must=[models.FieldCondition(
        key="book_id", match=models.MatchValue(value=book_id))])


def subject_filter(branch) -> models.Filter | None:
    """Restrict a search to one subject branch via the indexed `subject_paths` payload label.

    One exact keyword term the payload index resolves to its points directly — instead of an
    id-set of every book under the branch. `branch` is a reversed nav list (["Fiction",
    "Italy"]) OR the ready suffix ("Italy -- Fiction"); None/empty -> no restriction (all books).
    """
    if not branch:
        return None
    suffix = branch if isinstance(branch, str) else " -- ".join(reversed(list(branch)))
    return models.Filter(must=[models.FieldCondition(
        key="subject_paths", match=models.MatchValue(value=suffix))])


def _search_params(exact: bool):
    """Pick the retrieval STRATEGY for a filtered search.

    exact=True  -> SearchParams(exact=True): skip the HNSW graph and compare the query
                   against EVERY point the filter allows (Strategy 2 — brute force over the
                   isolated set). Fast + exact when that set is small; the caller decides
                   "small" (few allowed books).
    exact=False -> None: the normal filtered HNSW walk (Strategy 1) — the graph traversal
                   that stays efficient when the allowed set is broad.
    Qdrant makes this same call automatically via full_scan_threshold (measured in POINTS);
    passing it explicitly lets the caller trip it on a BOOK count it already knows.
    """
    return models.SearchParams(exact=True) if exact else None


# --- search --- #

def search_summary(client: QdrantClient, summary: str, descriptors: str | None = None,
                   limit: int = 5, flt: models.Filter | None = None,
                   w_summary: float = 0.7, exact: bool = False):
    """Search by summary (required); descriptors (optional) only rerank within the summary pool.

    Without descriptors: rank by the "summary" vector. With descriptors: summary
    GATES the candidate pool and carries the heavier weight, descriptors rerank
    inside it — so a descriptor-only match can't inject an off-topic scene.
        score = w_summary * summary_cos + (1 - w_summary) * descriptor_cos
    """
    if not summary:
        raise ValueError("search_summary needs a summary")
    sv = embed([QUERY_PREFIX + summary])[0]   # query-side prefix (index stays raw)

    sp = _search_params(exact)   # gate strategy: brute-force the filtered set vs HNSW walk
    if not descriptors:
        return client.query_points(COLLECTION, query=sv, using="summary", limit=limit,
                                   query_filter=flt, search_params=sp, with_payload=True).points

    pool = max(limit * 5, 50)   # summary candidate pool, reranked by descriptors below
    cands = client.query_points(COLLECTION, query=sv, using="summary", limit=pool,
                                query_filter=flt, search_params=sp, with_payload=True).points
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
# of embedding one joined "a, b, c" string (search_descriptors above), each descriptor
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
    exact: bool = False,
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
        limit=limit, query_filter=flt, search_params=_search_params(exact), with_payload=True,
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
    exact: bool = False,
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
                                query_filter=flt, search_params=_search_params(exact),
                                with_payload=True).points
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


# --- fused frame search: weighted cosine over every named vector, no hard filter --- #

# Starting field weights for search_fused. The holistic summary + the decisive `verb`
# lead; setting is nearly incidental. Query-time, fields ABSENT from the frame are
# dropped and the rest renormalized, so an unspecified field simply does not vote.
# Nothing here is a hard filter — a mislabelled scene is penalised, never excluded.
# Tune these against a retrieval eval.
DEFAULT_FIELD_WEIGHTS = dict(schema.DEFAULT_WEIGHTS)   # per-vector `weight` from the registry

# frame fields that are short descriptive PHRASES — embedded like the summary, so the
# bge query prefix applies (index raw, query prefixed). `descriptors` is an adjective
# LIST (weighted centroid, raw); `summary` is the gate. subject/verb/object are MULTIVECTOR
# fields (a list of terms, queried as a matrix); `setting` stays single-valued.
_PHRASE_FIELDS = ("subject", "verb", "object", "setting")


def _as_terms(v) -> list[str]:
    """Normalize a frame field value to a clean list of query terms.

    Accepts either a bare string (legacy / single-valued fields like setting, and gold
    entries authored before the frame went multi-valued) or a list of strings, and returns
    the non-empty trimmed terms. An empty/None field yields [] (the field simply abstains).
    """
    if v is None:
        return []
    items = [v] if isinstance(v, str) else list(v)
    return [t.strip() for t in items if isinstance(t, str) and t.strip()]


def _normalize_pool(raw: dict, ids: list, method: str | None) -> dict:
    """Rescale ONE field's cosines across the current candidate pool so its WEIGHT — not
    its accidental cosine spread — governs how much it moves the fused ranking.

    bge cosines sit in narrow, field-specific bands (a summary field may spread 0.60-0.88,
    `setting` 0.80-0.86). A raw weighted sum lets the wider-spread field dominate at any
    weight and the narrow one barely vote, so DEFAULT_FIELD_WEIGHTS don't mean what they
    say. Normalizing each field over the pool fixes that: weights become the real dial.

    Normalized over `ids` ONLY — ranking is relative to these candidates. A candidate
    missing from `raw` is imputed to the field mean (neutral), NOT 0.0, so a missing vector
    doesn't crater a scene the way a raw-0.0 cosine outlier would after scaling. A field
    with no spread over the pool can't rank anything, so every candidate gets 0.0: it
    contributes a constant and effectively abstains.
        method=None -> raw cosines (missing -> 0.0): the pre-norm behavior, kept for A/B.
        "zscore"    -> (x - mean)/std  (equalizes variance; the recommended default)
        "minmax"    -> (x - min)/(max - min) into [0,1] (outlier-sensitive)
    """
    if method is None:
        return {i: raw.get(i, 0.0) for i in ids}
    present = [raw[i] for i in ids if i in raw]
    if not present:
        return {i: 0.0 for i in ids}
    arr = np.asarray(present, dtype=np.float32)
    mean = float(arr.mean())
    if method == "zscore":
        std = float(arr.std())
        if std < 1e-9:
            return {i: 0.0 for i in ids}                 # no spread -> abstain
        return {i: (float(raw.get(i, mean)) - mean) / std for i in ids}
    if method == "minmax":
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo < 1e-9:
            return {i: 0.0 for i in ids}                 # no spread -> abstain
        return {i: (float(raw.get(i, mean)) - lo) / (hi - lo) for i in ids}
    raise ValueError(f"unknown normalize method {method!r} (use 'zscore', 'minmax', or None)")


def _fused_pool(client: QdrantClient, frame: dict, weights: dict, *,
                limit: int, flt: models.Filter | None, exact: bool = False):
    """Summary-gate the candidate pool and pull every present field's RAW cosines over it.

    The EXPENSIVE half of search_fused (query embeds + Qdrant round-trips); the fuse math
    on top (_fuse) is pure Python. Split out so a weight sweep can collect each query's
    pool ONCE and then re-fuse any number of weightings for free (see evals.collect_pools).
    Only fields present in the frame AND carrying positive weight are scored; summary is
    always scored (it is the gate). Returns (cands, ids, scores) where scores is
    {field: {point_id: raw_cosine}}, or ([], [], {}) when the gate pool is empty.
    """
    summ = (frame.get("summary") or "").strip()
    if not summ:
        raise ValueError("search_fused needs frame['summary'] (the gate)")

    present = {"summary": summ}
    for f in _PHRASE_FIELDS:
        terms = _as_terms(frame.get(f))                     # str OR list -> clean term list
        if terms:
            present[f] = terms
    if frame.get("descriptors"):
        present["descriptors"] = frame["descriptors"]
    fetch = {f for f in present if weights.get(f, 0) > 0}
    fetch.add("summary")                                    # the gate always scores

    sv = embed([QUERY_PREFIX + summ])[0]
    pool = max(limit * 5, 50)
    cands = client.query_points(COLLECTION, query=sv, using="summary", limit=pool,
                                query_filter=flt, search_params=_search_params(exact),
                                with_payload=True).points
    if not cands:
        return [], [], {}
    ids = [c.id for c in cands]
    id_filter = models.Filter(must=[models.HasIdCondition(has_id=ids)])

    scores = {"summary": {c.id: c.score for c in cands}}   # gate cosine == summary cosine
    for f in _PHRASE_FIELDS:                                # phrase fields (prefixed query)
        if f in fetch and f in present:
            terms = present[f]
            if f in MULTIVECTOR_NAMES:
                # multivector field: query the whole term list as a MATRIX. Qdrant MAX_SIM
                # then scores each candidate by, per query term, its best-matching stored
                # term (max-pool) — summed over query terms. One term -> plain max-pool.
                qv = embed([QUERY_PREFIX + t for t in terms])          # list of vectors
            else:                                                       # single-valued (setting)
                qv = embed([QUERY_PREFIX + ", ".join(terms)])[0]        # one flat vector
            hits = client.query_points(COLLECTION, query=qv, using=f, limit=len(ids),
                                       query_filter=id_filter, with_payload=False).points
            scores[f] = {h.id: h.score for h in hits}
    if "descriptors" in fetch and "descriptors" in present:  # adjective centroid (raw)
        desc = present["descriptors"]
        dv = weighted_vector(desc, [1.0 / len(desc)] * len(desc)).tolist()
        hits = client.query_points(COLLECTION, query=dv, using="descriptors", limit=len(ids),
                                   query_filter=id_filter, with_payload=False).points
        scores["descriptors"] = {h.id: h.score for h in hits}
    return cands, ids, scores


def _fuse(ids: list, scores: dict, weights: dict, *,
          normalize: str | None, limit: int) -> list:
    """Fuse cached per-field cosines under a weight vector -> ranked [(point_id, score)].

    The pure-math half of search_fused: present fields are those in `scores` carrying
    positive weight, renormalized to sum 1.00; each is normalized across the pool
    (_normalize_pool), then weighted-summed. No embeds, no Qdrant — cheap to sweep.
    """
    w = {f: weights[f] for f in scores if weights.get(f, 0) > 0}
    total = sum(w.values())
    if total <= 0:
        raise ValueError("no positive weight over the present frame fields")
    w = {f: x / total for f, x in w.items()}
    normed = {f: _normalize_pool(scores[f], ids, normalize) for f in w}
    fused = [(i, sum(wt * normed[f][i] for f, wt in w.items())) for i in ids]
    fused.sort(key=lambda t: t[1], reverse=True)
    return fused[:limit]


def search_fused(client: QdrantClient, frame: dict,
                 weights: dict | None = None, *,
                 limit: int = 5, flt: models.Filter | None = None,
                 normalize: str | None = "zscore", exact: bool = False):
    """Fuse EVERY named vector by weighted cosine — the general form of search_combined.

    `frame` is the query distilled into the same shape the index stores: any of
    {summary, subject, verb, object, setting} as strings + descriptors as a list.
    The summary GATES the candidate pool (widest recall net) and is required; every
    other present field is scored over that pool and blended in by weight:
        score = Σ  w_field * field_score
    where field_score is a cosine for single vectors and a MAX_SIM max-pool for the
    multivector frame fields (subject/verb/object) — a list field's query term takes its
    best-matching stored term, so a specific query matches a specific facet without the
    other listed terms diluting it. Fields present as lists accept a bare string too.
    Only fields actually present in `frame` vote (their weights renormalized to 1.00),
    so an unspecified setting/subject simply drops out. No field is a hard filter — a
    wrong label costs a scene rank, never its place in the pool. `weights` overrides
    DEFAULT_FIELD_WEIGHTS. Returns Qdrant ScoredPoints (payload == record), best-first.

    `normalize` rescales each field's cosines ACROSS THE POOL before the weighted sum
    (see _normalize_pool) so the weights govern influence instead of each field's raw
    cosine spread; "zscore" (default) is recommended, None reproduces the old raw-cosine
    sum for A/B. NOTE: with normalization the returned `.score` is a fused, pool-relative
    score (z-scores can be negative), NOT a cosine — good for ranking within one query,
    but not comparable across queries or thresholdable as a similarity.
    """
    src = weights or DEFAULT_FIELD_WEIGHTS
    cands, ids, scores = _fused_pool(client, frame, src, limit=limit, flt=flt, exact=exact)
    if not cands:
        return cands
    ranked = _fuse(ids, scores, src, normalize=normalize, limit=limit)
    by_id = {c.id: c for c in cands}
    out = []
    for i, sc in ranked:                        # reattach fused score, emit in fused order
        c = by_id[i]
        c.score = sc
        out.append(c)
    return out
