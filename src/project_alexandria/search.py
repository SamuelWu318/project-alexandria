import uuid
import numpy as np
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from utils import SrcPaths, schema # scene-record registry (single source of truth) — drives the named vectors

# ---- Read path: query the scene vector DB (import THIS from the app / API) ----
# Pulls in only qdrant + fastembed — NO LLM, NO segmentation — so the query path stays light. It also
# OWNS the vector-store primitives (COLLECTION, vector config, embedder, point id, filters) that the
# write path (embed.py) imports to index. Invariants (see CLAUDE.md): EMBED_MODEL must match the index;
# point_id is a stable uuid5 so re-index overwrites; bge is asymmetric (queries are prefixed).

# ---- vector store config (shared with embed.py's indexer) ----

# Qdrant collection name — the read/write join key (embed.py writes here, search.py queries here).
COLLECTION = "scenes"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"     # MUST match the model the index was built with
# named vectors == the registry's vector:true fields (summary + descriptors + the svos multivector).
VECTOR_NAMES = schema.VECTOR_NAMES
# multivector:true fields (svos): a LIST of per-item vectors scored by MAX_SIM (max-pool). MUST be
# queried with a matrix (list of vectors), even a 1-row one — a flat vector is rejected by the index.
MULTIVECTOR_NAMES = frozenset(schema.MULTIVECTOR_NAMES)

# bge query-side instruction prefix (summary path only). Set to "" to A/B without re-indexing.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# stable per-scene Qdrant id namespace: uuid5(NAMESPACE, scene_id) -> same point
NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "projectalexandria.scenes")


# ** LOCKED **  ** MAIN ** — embed.py + schema.sync_qdrant address points by this id
# Stable Qdrant point id for a scene (uuid5, so re-index overwrites).
def point_id(scene_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, scene_id))


# ---- embedder (same model for index + query) ----

_EMBEDDER = None

# ** LOCKED **
# Lazily construct and cache the shared TextEmbedding model.
def _embedder() -> TextEmbedding:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = TextEmbedding(model_name=EMBED_MODEL)
    return _EMBEDDER


# ** LOCKED **  ** MAIN ** — embed.py imports this as `_embed` to build every vector
# Embed a batch of texts into plain float lists.
def embed(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in _embedder().embed(texts)]


# ---- client + filters ----

# ** MAIN ** — evals + schema.sync_qdrant open the on-disk store here
# Open the on-disk Qdrant client (first call downloads the embed model).
def open_client() -> QdrantClient:
    return QdrantClient(path=str(SrcPaths.QDRANT_DIR))


# ** MAIN ** — tests + webtest restrict a search to one book here
# Filter that restricts a search to one book, or None for all books.
def book_filter(book_id: str | None) -> models.Filter | None:
    if not book_id:
        return None
    return models.Filter(must=[models.FieldCondition(
        key="book_id", match=models.MatchValue(value=book_id))])


# ** MAIN ** — webtest restricts a search to one subject branch here
# Restrict a search to one subject branch via the indexed `subject_paths` payload label (one exact keyword term).
def subject_filter(branch) -> models.Filter | None:
    if not branch:
        return None
    suffix = branch if isinstance(branch, str) else " -- ".join(reversed(list(branch)))
    return models.Filter(must=[models.FieldCondition(
        key="subject_paths", match=models.MatchValue(value=suffix))])


# ** LOCKED **
# Pick the retrieval STRATEGY: exact brute-force over the filtered set (few books) vs the filtered HNSW walk (broad).
def _search_params(exact: bool):
    return models.SearchParams(exact=True) if exact else None


# ---- weighted + negative descriptor search (per-descriptor weighting is a QUERY-time op) ----

WEIGHT_TOL = 1e-6   # how far a weight list may drift from summing to 1.00


# ** LOCKED **
# L2-normalize a vector; a zero vector is returned unchanged (guards divide-by-zero).
def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n else v


# ** LOCKED **
# Validate a (terms, weights) pair: non-empty, equal length, non-negative, sum≈1.00.
def _check_weights(terms: list[str], weights: list[float], label: str):
    if not terms:
        raise ValueError(f"{label}: need at least one term")
    if len(terms) != len(weights):
        raise ValueError(f"{label}: {len(terms)} terms but {len(weights)} weights")
    if any(w < 0 for w in weights):
        raise ValueError(f"{label}: weights must be non-negative")
    total = sum(weights)
    if abs(total - 1.0) > WEIGHT_TOL:
        raise ValueError(f"{label}: weights must sum to 1.00 (got {total:.6f})")


# ** LOCKED **
# Weighted centroid of the terms' embeddings as a unit vector (each term unit-normalized first, so no word dominates by magnitude).
def weighted_vector(terms: list[str], weights: list[float]) -> np.ndarray:
    vecs = embed(terms)                         # one batched embed call for all terms
    acc = np.zeros(len(vecs[0]), dtype=np.float32)
    for v, w in zip(vecs, weights):
        acc += w * _unit(v)
    return _unit(acc)


# ** MAIN ** — search() runs the flavor channel through here; evals A/Bs it
# Descriptor search with per-descriptor weights and optional anti-descriptors (weighted centroid of INDIVIDUAL embeddings, anti = subtraction). Returns ScoredPoints best-first.
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
    if not weights: weights = [1.00 / len(descriptors) for _ in descriptors]   # equal-weight default

    _check_weights(descriptors, weights, "descriptors")
    q = weighted_vector(descriptors, weights)                  # positive centroid

    if anti_descriptors or anti_weights:
        if not (anti_descriptors and anti_weights):
            raise ValueError("anti_descriptors and anti_weights must be given together")
        _check_weights(anti_descriptors, anti_weights, "anti_descriptors")
        q = q - anti_strength * weighted_vector(anti_descriptors, anti_weights)   # tilt away from the anti-flavor
        if float(np.linalg.norm(q)) < 1e-8:
            raise ValueError("positive and anti descriptors cancel out; lower anti_strength")
        q = _unit(q)

    return client.query_points(
        COLLECTION, query=q.tolist(), using="descriptors",
        limit=limit, query_filter=flt, search_params=_search_params(exact), with_payload=True,
    ).points


# ---- field weights (registry) — LEGACY, not a search knob ----
# Per-vector `weight` from scene_schema.json. NOTHING in the read path consumes these anymore:
# search_scenes fuses summary vs svos by MAX (no weights) and search() blends channels by weighted
# RRF (method_weights, NOT these). Kept only so the schema<->search drift check stays meaningful.
# TO TUNE retrieval, use search()'s knobs (method_weights, normalize), NOT this constant.
DEFAULT_FIELD_WEIGHTS = dict(schema.DEFAULT_WEIGHTS)   # per-vector `weight` from the registry (legacy)


# ** LOCKED **  ** MAIN ** — embed.py imports this to normalize each multivector field
# Normalize a multivector field value to a clean list of items (bare string or list -> non-empty trimmed items; None -> []).
def _as_terms(v) -> list[str]:
    if v is None:
        return []
    items = [v] if isinstance(v, str) else list(v)
    return [t.strip() for t in items if isinstance(t, str) and t.strip()]


# ** LOCKED **
# Rescale ONE channel's cosines across the candidate pool so a cross-channel MAX compares RELATIVE strength (missing -> channel mean; no-spread -> abstain).
def _normalize_pool(raw: dict, ids: list, method: str | None) -> dict:
    # method=None -> raw cosines (missing -> 0.0); "zscore" -> (x-mean)/std; "minmax" -> into [0,1].
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


# ---- unified search: orchestrate the specific retrievers + merge ----

# Default RRF balance of what-happens (scenes) vs flavor (descriptors); only the ratio matters (RRF is rank-based).
DEFAULT_METHOD_WEIGHTS = {"scenes": 0.7, "flavor": 0.3}


# ** LOCKED **
# AND several optional filters into one (merge their `must` conditions); None if empty.
def _and_filters(*filters: models.Filter | None) -> models.Filter | None:
    musts: list = []
    for f in filters:
        if f is not None and f.must:
            musts.extend(f.must)
    return models.Filter(must=musts) if musts else None


# Manual query moments -> clean clause SENTENCES (accepts a string, list of strings, or {"sentence":...} dicts).
def _moment_sentences(moments) -> list[str]:
    if not moments:
        return []
    if isinstance(moments, str):
        moments = [moments]
    out = []
    for m in moments:
        s = m.get("sentence") if isinstance(m, dict) else m
        if isinstance(s, str) and s.strip():
            out.append(s.strip())
    return out


# ** MAIN ** — search() runs the what-happens channel here; evals.collect_channels calls it directly
# What-happens search: the general `summary` and the `svos` moment-clauses fused by the GREATEST SINGLE MATCH (each channel z-normalized over the union pool). Returns ScoredPoints, pool-relative score.
def search_scenes(client: QdrantClient, *, summary: str | None = None, moments=None,
                  limit: int = 5, flt: models.Filter | None = None,
                  prefetch: int | None = None, normalize: str | None = "zscore",
                  exact: bool = False):
    summ = (summary or "").strip()
    sents = _moment_sentences(moments)                         # query clause sentences
    if not (summ or sents):
        raise ValueError("search_scenes needs a summary or at least one moment sentence")
    prefetch = prefetch or max(limit * 5, 50)

    cand: dict = {}
    sv = qmat = None
    if summ:                                                   # summary channel prefetch
        sv = embed([QUERY_PREFIX + summ])[0]
        for p in client.query_points(COLLECTION, query=sv, using="summary", limit=prefetch,
                                     query_filter=flt, search_params=_search_params(exact),
                                     with_payload=True).points:
            cand[p.id] = p
    if sents:                                                  # svos channel prefetch (MAX_SIM)
        qmat = embed([QUERY_PREFIX + s for s in sents])
        for p in client.query_points(COLLECTION, query=qmat, using="svos", limit=prefetch,
                                     query_filter=flt, search_params=_search_params(exact),
                                     with_payload=True).points:
            cand.setdefault(p.id, p)
    if not cand:
        return []
    ids = list(cand)                                           # the union candidate pool
    idflt = models.Filter(must=[models.HasIdCondition(has_id=ids)])

    scores: dict = {}                                          # score BOTH channels over the union
    if sv is not None:
        hits = client.query_points(COLLECTION, query=sv, using="summary", limit=len(ids),
                                   query_filter=idflt, with_payload=False).points
        scores["summary"] = {h.id: h.score for h in hits}
    if qmat is not None:
        hits = client.query_points(COLLECTION, query=qmat, using="svos", limit=len(ids),
                                   query_filter=idflt, with_payload=False).points
        scores["svos"] = {h.id: h.score for h in hits}

    normed = {ch: _normalize_pool(sc, ids, normalize) for ch, sc in scores.items()}   # per-channel z-norm
    fused = sorted(((i, max(n[i] for n in normed.values())) for i in ids),            # greatest single match
                   key=lambda t: t[1], reverse=True)
    out = []
    for i, sc in fused[:limit]:
        p = cand[i]
        p.score = sc
        out.append(p)
    return out


# ** MAIN ** — per-facet search over ONE frame multivector (subject/verb/object/setting/svos)
# Query ONE frame multivector on its own as a MAX_SIM matrix, so a scene's score is its best-matching stored facet term.
def search_frame(client: QdrantClient, field: str, terms, *, limit: int = 5,
                 flt: models.Filter | None = None, exact: bool = False):
    if field not in MULTIVECTOR_NAMES:
        raise ValueError(f"{field!r} is not a multivector field (have {sorted(MULTIVECTOR_NAMES)})")
    items = _as_terms(terms)
    if not items:
        raise ValueError("search_frame needs at least one term")
    qmat = embed([QUERY_PREFIX + t for t in items])
    return client.query_points(COLLECTION, query=qmat, using=field, limit=limit,
                               query_filter=flt, search_params=_search_params(exact),
                               with_payload=True).points


# ** LOCKED **
# Weighted Reciprocal Rank Fusion of several ranked ScoredPoint lists -> one ranking (rank-based, so heterogeneous scores reconcile without a shared scale).
def _rrf(rankings: dict, weights: dict, k: int, limit: int) -> list:
    total: dict = {}
    seen: dict = {}
    for m, pts in rankings.items():
        w = weights.get(m, 1.0)
        for rank, p in enumerate(pts):
            total[p.id] = total.get(p.id, 0.0) + w / (k + rank + 1)
            seen[p.id] = p
    ranked = sorted(total, key=lambda i: total[i], reverse=True)[:limit]
    out = []
    for i in ranked:
        p = seen[i]
        p.score = total[i]
        out.append(p)
    return out


# ** MAIN ** — the ONE search entry: imported by tests, evals, webtest, embed's read side
# Unified scene search: orchestrate the active retrievers over one filter and MERGE their rankings by weighted RRF. Two knobs — method_weights (RRF balance) + normalize. Returns ScoredPoints best-first.
def search(client: QdrantClient, *, summary: str | None = None, moments=None,
           descriptors: list[str] | None = None, weights: list[float] | None = None,
           anti_descriptors: list[str] | None = None, anti_weights: list[float] | None = None,
           anti_strength: float = 1.0, book_id: str | None = None, subject_branch=None,
           flt: models.Filter | None = None, limit: int = 5, prefetch: int | None = None,
           normalize: str | None = "zscore", method_weights: dict | None = None,
           rrf_k: int = 60, exact: bool = False):
    if flt is None:
        flt = _and_filters(book_filter(book_id), subject_filter(subject_branch))   # AND book + subject pre-filters
    prefetch = prefetch or max(limit * 5, 50)

    rankings: dict = {}
    if summary or moments:
        rankings["scenes"] = search_scenes(                    # what-happens: max(summary, svos)
            client, summary=summary, moments=moments, limit=prefetch, flt=flt,
            normalize=normalize, exact=exact)
    if descriptors:
        rankings["flavor"] = search_weighted_descriptors(       # flavor: weighted descriptor centroid
            client, descriptors, weights, anti_descriptors=anti_descriptors,
            anti_weights=anti_weights, anti_strength=anti_strength, limit=prefetch,
            flt=flt, exact=exact)
    if not rankings:
        raise ValueError("search needs at least one of: summary, moments, descriptors")
    if len(rankings) == 1:
        return next(iter(rankings.values()))[:limit]           # one channel -> its ranking, untouched
    mw = method_weights or DEFAULT_METHOD_WEIGHTS
    return _rrf(rankings, mw, rrf_k, limit)                     # >1 channel -> rank-fuse them
