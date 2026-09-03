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
# + the svos moment-clause multivector). Edit scene_schema.json to add/drop a vector, then re-index.
VECTOR_NAMES = schema.VECTOR_NAMES
# multivector:true fields (svos) store a LIST of per-item vectors and score by MAX_SIM: for a
# query clause the field's score is the MAX cosine over the scene's stored clauses (max-pool).
# A specific query locks onto a specific beat; a general one onto a general beat — neither
# averaged away. These fields MUST be queried with a matrix (list of vectors), even a 1-row
# one — a single flat vector is rejected by the multivector index.
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


# --- field weights (registry) — LEGACY, not a search knob --- #
# Per-vector `weight` from scene_schema.json. NOTHING in the read path consumes these anymore:
# search_scenes fuses summary vs svos by MAX (greatest single match — no weights), and search()
# blends what-happens vs flavor by weighted RRF via its `method_weights` arg (NOT these). Kept
# only so the schema<->search drift check stays meaningful. TO TUNE retrieval, use search()'s
# knobs, NOT these:
#   * method_weights={"scenes":.., "flavor":..}  — balance what-happens vs descriptor flavor (RRF)
#   * normalize="zscore"|"minmax"|None            — how each channel is scaled before the MAX
# summary-vs-svos has NO weight by design (it is a max). Editing the per-field weights below
# does nothing to search; it only keeps this constant in sync with the registry.
DEFAULT_FIELD_WEIGHTS = dict(schema.DEFAULT_WEIGHTS)   # per-vector `weight` from the registry (legacy)


def _as_terms(v) -> list[str]:
    """Normalize a multivector field value to a clean list of items (e.g. svos clauses).

    Accepts either a bare string (single value) or a list of strings, and returns the
    non-empty trimmed items. An empty/None field yields [] (the field simply abstains).
    """
    if v is None:
        return []
    items = [v] if isinstance(v, str) else list(v)
    return [t.strip() for t in items if isinstance(t, str) and t.strip()]


def _normalize_pool(raw: dict, ids: list, method: str | None) -> dict:
    """Rescale ONE channel's cosines across the current candidate pool so a cross-channel MAX
    compares RELATIVE strength, not raw cosine band.

    bge cosines sit in narrow, channel-specific bands (the summary channel may spread
    0.60-0.88, a short svos clause tighter). search_scenes ranks by max(z(summary), z(svos));
    on RAW cosines the higher/wider-band channel would win regardless of relevance. Normalizing
    each channel over the pool first makes "greatest" mean greatest relative to that channel's
    own spread — the fair max the greatest-single-match design needs. (This is a scaling step,
    NOT a weighting one: there is no per-channel weight in the max.)

    Normalized over `ids` ONLY — ranking is relative to these candidates. A candidate
    missing from `raw` is imputed to the channel mean (neutral), NOT 0.0, so a missing vector
    doesn't crater a scene the way a raw-0.0 cosine outlier would after scaling. A channel
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


# --- unified search: orchestrate the specific retrievers + merge --- #

# Default RRF balance of the what-happens (scenes) vs flavor (descriptors) channels when
# search() fuses both. RRF is rank-based, so only the scenes:flavor RATIO matters. search()
# falls back to this when `method_weights` is not passed, and evals `--tune` uses it as the
# sweep baseline — so changing the default retrieval balance is editing this one line.
DEFAULT_METHOD_WEIGHTS = {"scenes": 0.7, "flavor": 0.3}


def _and_filters(*filters: models.Filter | None) -> models.Filter | None:
    """AND several optional filters into one (merge their `must` conditions). None if empty."""
    musts: list = []
    for f in filters:
        if f is not None and f.must:
            musts.extend(f.must)
    return models.Filter(must=musts) if musts else None


def _moment_sentences(moments) -> list[str]:
    """Manual query moments -> clean clause SENTENCES. Accepts a single string, a list of
    strings, or a list of {"sentence": ...} dicts (the enrichment shape). Empties dropped."""
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


def search_scenes(client: QdrantClient, *, summary: str | None = None, moments=None,
                  limit: int = 5, flt: models.Filter | None = None,
                  prefetch: int | None = None, normalize: str | None = "zscore",
                  exact: bool = False):
    """What-happens search: the general `summary` and the `svos` moment-clauses fused by the
    GREATEST SINGLE MATCH. Two channels — summary (one holistic vector) and svos (the scene's
    moment SENTENCES as a MAX_SIM multivector). The candidate pool is the UNION of each
    channel's prefetch (either can drive recall), and a scene scores
        max( z(summary_cos), z(svos_maxsim) )
    over that pool — each channel z-normalized so "greatest" means greatest RELATIVE strength,
    not whichever channel sits in the higher cosine band. So a vague/holistic query resolves on
    summary and a sharp single-action query on a beat, neither diluting the other.

    `moments` is the query's clause sentence(s): a string, a list of strings, or a list of
    {"sentence": ...} dicts (manual input). At least one of summary/moments is required.
    Returns Qdrant ScoredPoints (payload == record), best-first; `.score` is pool-relative.
    """
    summ = (summary or "").strip()
    sents = _moment_sentences(moments)
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
    ids = list(cand)
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

    normed = {ch: _normalize_pool(sc, ids, normalize) for ch, sc in scores.items()}
    fused = sorted(((i, max(n[i] for n in normed.values())) for i in ids),
                   key=lambda t: t[1], reverse=True)
    out = []
    for i, sc in fused[:limit]:
        p = cand[i]
        p.score = sc
        out.append(p)
    return out


def search_frame(client: QdrantClient, field: str, terms, *, limit: int = 5,
                 flt: models.Filter | None = None, exact: bool = False):
    """Per-facet 'S V O S individual' search: query ONE frame multivector on its own.

    `field` is subject / verb / object / setting (or svos) — a multivector field; `terms` is a
    query string or list of terms, queried as a MAX_SIM matrix against that field alone, so a
    scene's score is its best-matching stored facet term. Returns Qdrant ScoredPoints, best-first.
    """
    if field not in MULTIVECTOR_NAMES:
        raise ValueError(f"{field!r} is not a multivector field (have {sorted(MULTIVECTOR_NAMES)})")
    items = _as_terms(terms)
    if not items:
        raise ValueError("search_frame needs at least one term")
    qmat = embed([QUERY_PREFIX + t for t in items])
    return client.query_points(COLLECTION, query=qmat, using=field, limit=limit,
                               query_filter=flt, search_params=_search_params(exact),
                               with_payload=True).points


def _rrf(rankings: dict, weights: dict, k: int, limit: int) -> list:
    """Weighted Reciprocal Rank Fusion of several ranked ScoredPoint lists -> one ranking.

    Rank-based, so heterogeneous scores (a z-scored what-happens score, a raw descriptor
    cosine) reconcile without a shared scale: each method contributes w_m / (k + rank) to a
    scene's total, best rank first. Returns the top `limit` ScoredPoints, `.score` = RRF total.
    """
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


def search(client: QdrantClient, *, summary: str | None = None, moments=None,
           descriptors: list[str] | None = None, weights: list[float] | None = None,
           anti_descriptors: list[str] | None = None, anti_weights: list[float] | None = None,
           anti_strength: float = 1.0, book_id: str | None = None, subject_branch=None,
           flt: models.Filter | None = None, limit: int = 5, prefetch: int | None = None,
           normalize: str | None = "zscore", method_weights: dict | None = None,
           rrf_k: int = 60, exact: bool = False):
    """Unified scene search: ONE entry, all configurations. Orchestrates the specific
    retrievers and MERGES their rankings.

    Retrievers, activated by what you pass:
      * what-happens -> search_scenes(summary, moments): max(summary, svos) greatest-single-match.
      * flavor       -> search_weighted_descriptors(descriptors, weights, anti_*): weighted
                        descriptor centroid, optional anti-descriptors.
    All active retrievers run over the SAME filter — `flt` if given, else book_id AND
    subject_branch ANDed together. Exactly one active -> that ranking, untouched. More than
    one -> their ranked outputs are fused by weighted Reciprocal Rank Fusion (rank-based, so
    a z-scored what-happens score and a raw descriptor cosine merge without a shared scale);
    `method_weights` overrides the per-method RRF weight (default DEFAULT_METHOD_WEIGHTS,
    {"scenes":0.7,"flavor":0.3}).
    Each retriever collects `prefetch` candidates before the merge trims to `limit`.
    Returns Qdrant ScoredPoints (payload == record), best-first.

    TUNING — the search has exactly TWO knobs, both here (NOT the schema per-field weights):
      * method_weights — the RRF balance of what-happens vs flavor. Pass it per call (the
        webtest query forwards a per-query `method_weights`); raise "scenes" to favor the
        beats/summary, "flavor" to lean on descriptors.
      * normalize — how each channel is scaled before the what-happens MAX ("zscore" default;
        evals `--mode norm` A/Bs it). summary-vs-svos itself is a MAX and has NO weight.
    The per-field `weight` in scene_schema.json / DEFAULT_FIELD_WEIGHTS is LEGACY — it does not
    affect this function at all.
    """
    if flt is None:
        flt = _and_filters(book_filter(book_id), subject_filter(subject_branch))
    prefetch = prefetch or max(limit * 5, 50)

    rankings: dict = {}
    if summary or moments:
        rankings["scenes"] = search_scenes(
            client, summary=summary, moments=moments, limit=prefetch, flt=flt,
            normalize=normalize, exact=exact)
    if descriptors:
        rankings["flavor"] = search_weighted_descriptors(
            client, descriptors, weights, anti_descriptors=anti_descriptors,
            anti_weights=anti_weights, anti_strength=anti_strength, limit=prefetch,
            flt=flt, exact=exact)
    if not rankings:
        raise ValueError("search needs at least one of: summary, moments, descriptors")
    if len(rankings) == 1:
        return next(iter(rankings.values()))[:limit]
    mw = method_weights or DEFAULT_METHOD_WEIGHTS
    return _rrf(rankings, mw, rrf_k, limit)
