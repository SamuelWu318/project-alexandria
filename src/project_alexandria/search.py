import numpy as np
from qdrant_client import QdrantClient, models
from utils import SrcPaths
from utils.read_write import read_json   # tuned field_weights override (legacy weight stack, retired in Phase 8)
from utils.vectorstore import (COLLECTION, EMBED_MODEL, VECTOR_NAMES, MULTIVECTOR_NAMES,   # the Qdrant contract:
                               SUBJECT_PATHS_FIELD, QUERY_PREFIX, NAMESPACE, point_id, embed,  # defined ONCE in
                               open_client, book_filter, subject_filter, facet_filter,         # utils/vectorstore.py,
                               _search_params, _as_terms)                                      # re-exposed here for now

# ---- Read path: query the scene vector DB (import THIS from the app / API) ----
# Pulls in only qdrant + fastembed — NO LLM, NO segmentation — so the query path stays light. The
# vector-store CONTRACT (COLLECTION, vector set, embedder, point id, filters) now lives in
# utils/vectorstore.py — the ONE home both this reader and the writer (index.py) import; extracting it
# removed the old embed->search coupling (principle #4). This file is read LOGIC only. Invariants
# (CLAUDE.md): EMBED_MODEL must match the index; point_id is a stable uuid5; bge is asymmetric (queries prefixed).

# ---- vector-store contract (imported from utils/vectorstore.py, the ONE home) ----
# COLLECTION, EMBED_MODEL, VECTOR_NAMES, MULTIVECTOR_NAMES, SUBJECT_PATHS_FIELD, QUERY_PREFIX, NAMESPACE,
# point_id, embed, open_client, book_filter, subject_filter, facet_filter, _search_params and _as_terms
# are imported at the top of this file. They are re-exposed as `search.*` for the not-yet-migrated
# read-path consumers (evals / tests / webtest); utils/vectorstore.py is where they are DEFINED.
# tone/intensity/arc below are read-side hard filters (retired in the Phase 8 rewrite), so they stay here.


# ** MAIN ** — webtest/evals hard-filter a search by flavor facet (single value or any-of a list)
# Restrict a search to scene(s) whose dominant tone matches (payload col `dominant_tone`).
def tone_filter(tone) -> models.Filter | None:
    return facet_filter("dominant_tone", tone)


# ** MAIN ** — webtest/evals hard-filter a search by flavor facet (single value or any-of a list)
# Restrict a search to scene(s) whose intensity matches (payload col `intensity`).
def intensity_filter(intensity) -> models.Filter | None:
    return facet_filter("intensity", intensity)


# ** MAIN ** — webtest/evals hard-filter a search by flavor facet (single value or any-of a list)
# Restrict a search to scene(s) whose narrative arc matches (payload col `arc`).
def arc_filter(arc) -> models.Filter | None:
    return facet_filter("arc", arc)


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


# ---- field weights (LEGACY — retired in Phase 8) ----
# The per-field `weight` is RETIRED (PLAN D3): the read path will tune with method_weights + the
# soft-rank knobs, not per-vector weights. The whole weight stack here — DEFAULT_FIELD_WEIGHTS /
# SCENES_DEFAULT_WEIGHTS / active_field_weights / _resolve_field_weights and their use in
# score_channels / blend_channels — is deleted in the Phase 8 read-path rewrite. Frozen below (the
# pre-D3 scene_schema.json values) ONLY so the read path keeps its exact current blend through the wave.
DEFAULT_FIELD_WEIGHTS = {"summary": 0.25, "descriptors": 0.25, "svos": 0.5,   # frozen legacy weights
                        "subject": 0.2, "verb": 0.1, "object": 0.15, "setting": 0.05}

# the vector channels fused INSIDE search_scenes (descriptors is the separate `flavor` method + RRF)
SCENES_VECTORS = ("summary", "svos", "subject", "verb", "object", "setting")
SCENES_DEFAULT_WEIGHTS = {k: DEFAULT_FIELD_WEIGHTS.get(k, 0.0) for k in SCENES_VECTORS}


# ** MAIN ** — the live default field_weights; every None-weight caller (webtest, evals base runs) resolves through here
# The DEFAULT field_weights for search: the tuned override that `evals --tune` writes to SrcPaths.TUNED_WEIGHTS_PATH
# if it exists and is valid, else the schema defaults. Read fresh each call, so a re-tune takes effect with no
# restart. A caller passing explicit field_weights still overrides this. Bad/empty file -> schema defaults.
def active_field_weights() -> dict:
    data = read_json(SrcPaths.TUNED_WEIGHTS_PATH, default=None)   # {"field_weights": {chan: w}, ...} or None
    if isinstance(data, dict):
        fw = data.get("field_weights")
        if isinstance(fw, dict) and any(fw.values()):
            return {k: float(v) for k, v in fw.items()}
    return SCENES_DEFAULT_WEIGHTS


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


# Frame query terms per facet: an explicit `frame` dict wins, else the parts are read off moment dicts.
def _frame_query_terms(moments, frame=None) -> dict:
    frame = frame or {}
    out = {f: _as_terms(frame.get(f)) for f in ("subject", "verb", "object", "setting")}
    if moments and not isinstance(moments, str):
        for m in moments:
            if not isinstance(m, dict):
                continue
            for f in ("subject", "verb", "object", "setting"):
                if _as_terms(frame.get(f)):            # explicit terms override derivation
                    continue
                v = (m.get(f) or "").strip()
                if v and v not in out[f]:
                    out[f].append(v)
    return {f: t for f, t in out.items() if t}


# Build the query for each active vector channel: summary -> one vector, svos + frame facets -> matrices.
# `channel_vectors` (optional) supplies PRE-EMBEDDED query vectors per channel — bypassing bge for those
# channels (used by the learned query adapter, which emits vectors, not text). A supplied vector OVERRIDES
# the text-derived one for that channel; other channels still come from text. Value shape must match the
# channel: a single vector (list[float]) for summary/descriptors, a matrix (list[list[float]]) for a
# multivector field. Pass text, vectors, or a mix.
def _channel_queries(summary, moments, frame, channel_vectors: dict | None = None) -> dict:
    channels: dict = {}
    summ = (summary or "").strip()
    if summ:
        channels["summary"] = embed([QUERY_PREFIX + summ])[0]          # single holistic vector
    sents = _moment_sentences(moments)
    if sents:
        channels["svos"] = embed([QUERY_PREFIX + s for s in sents])    # MAX_SIM matrix of clause sentences
    for f, terms in _frame_query_terms(moments, frame).items():
        channels[f] = embed([QUERY_PREFIX + t for t in terms])         # per-facet MAX_SIM matrix
    if channel_vectors:                                                 # pre-embedded vectors win over text
        for name, v in channel_vectors.items():
            channels[name] = v.tolist() if isinstance(v, np.ndarray) else v
    return channels


# ** LOCKED **
# Resolve per-channel weights over the ACTIVE channels: non-negative, renormalized to sum to 1 (all-zero -> equal).
def _resolve_field_weights(field_weights, names: list) -> dict:
    base = field_weights if field_weights is not None else active_field_weights()   # tuned override (if any) else schema defaults
    w = {n: max(0.0, float(base.get(n, 0.0))) for n in names}
    total = sum(w.values())
    if total <= 0:
        return {n: 1.0 / len(names) for n in names}
    return {n: w[n] / total for n in names}


# ** MAIN ** — search_scenes blends these; evals caches them once to re-blend offline while tuning
# Score every active vector channel over ONE union candidate pool and z-normalize each; returns {"channels": {name: {id: z}}, "cand": {id: point}, "ids": [...]}.
def score_channels(client: QdrantClient, *, summary: str | None = None, moments=None, frame=None,
                   channel_vectors: dict | None = None,
                   normalize: str | None = "zscore", flt: models.Filter | None = None,
                   prefetch: int = 50, exact: bool = False) -> dict:
    channels = _channel_queries(summary, moments, frame, channel_vectors)
    if not channels:
        raise ValueError("score_channels needs a summary, moment sentence, frame term, or channel_vectors")
    cand: dict = {}                                            # union of every channel's prefetch
    for name, q in channels.items():
        for p in client.query_points(COLLECTION, query=q, using=name, limit=prefetch,
                                     query_filter=flt, search_params=_search_params(exact),
                                     with_payload=True).points:
            cand.setdefault(p.id, p)
    ids = list(cand)
    if not ids:
        return {"channels": {}, "cand": {}, "ids": []}
    idflt = models.Filter(must=[models.HasIdCondition(has_id=ids)])
    normed: dict = {}                                          # score every channel over the whole union
    for name, q in channels.items():
        hits = client.query_points(COLLECTION, query=q, using=name, limit=len(ids),
                                   query_filter=idflt, with_payload=False).points
        normed[name] = _normalize_pool({h.id: h.score for h in hits}, ids, normalize)
    return {"channels": normed, "cand": cand, "ids": ids}


# ** MAIN ** — evals blends cached channels under any weights while tuning
# Fuse pre-scored channels into one ranking: per-channel z-score * weight, combined by weighted SUM (blend) or MAX (greatest single match). Returns [(id, score)] best-first.
def blend_channels(scored: dict, field_weights: dict | None = None, combine: str = "sum") -> list:
    normed, ids = scored["channels"], scored["ids"]
    if not ids or not normed:
        return []
    weights = _resolve_field_weights(field_weights, list(normed))
    fused = []
    for i in ids:
        contribs = [weights[name] * normed[name][i] for name in normed]
        fused.append((i, max(contribs) if combine == "max" else sum(contribs)))
    fused.sort(key=lambda t: t[1], reverse=True)
    return fused


# ** MAIN ** — search() runs the what-happens/frame channel here; evals A/Bs it
# What-happens + frame search: the summary, svos, and subject/verb/object/setting channels fused by a per-channel z-normalized weighted blend (no channel overpowers by scale; field_weights tilt it). Returns ScoredPoints, pool-relative score.
def search_scenes(client: QdrantClient, *, summary: str | None = None, moments=None, frame=None,
                  channel_vectors: dict | None = None,
                  field_weights: dict | None = None, combine: str = "sum",
                  limit: int = 5, flt: models.Filter | None = None,
                  prefetch: int | None = None, normalize: str | None = "zscore",
                  exact: bool = False):
    prefetch = prefetch or max(limit * 5, 50)
    scored = score_channels(client, summary=summary, moments=moments, frame=frame,   # prefetch + z-norm each channel
                            channel_vectors=channel_vectors,                         # pre-embedded (adapter) vectors, if any
                            normalize=normalize, flt=flt, prefetch=prefetch, exact=exact)
    fused = blend_channels(scored, field_weights, combine)                           # weighted blend over the pool
    out = []
    for i, sc in fused[:limit]:
        p = scored["cand"][i]
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
# Unified scene search: orchestrate the active retrievers over one filter and MERGE by weighted RRF. Inputs — text (summary/moments/frame) and/or channel_vectors (pre-embedded per-channel query vectors from the learned adapter; override the text-derived ones) drive the scenes method; descriptors drive the flavor method. Knobs — field_weights (the 6 vector channels), method_weights (scenes vs flavor RRF), combine, normalize. Hard pre-filters (ANDed): book_id, subject_branch, tone/intensity/arc (each a single value or any-of a list). Returns ScoredPoints best-first.
def search(client: QdrantClient, *, summary: str | None = None, moments=None, frame=None,
           channel_vectors: dict | None = None,
           descriptors: list[str] | None = None, weights: list[float] | None = None,
           anti_descriptors: list[str] | None = None, anti_weights: list[float] | None = None,
           anti_strength: float = 1.0, book_id: str | None = None, subject_branch=None,
           tone=None, intensity=None, arc=None,
           flt: models.Filter | None = None, limit: int = 5, prefetch: int | None = None,
           normalize: str | None = "zscore", combine: str = "sum",
           field_weights: dict | None = None, method_weights: dict | None = None,
           rrf_k: int = 60, exact: bool = False):
    if flt is None:
        flt = _and_filters(book_filter(book_id), subject_filter(subject_branch),   # AND book + subject +
                           tone_filter(tone), intensity_filter(intensity),         # flavor-facet hard filters
                           arc_filter(arc))
    prefetch = prefetch or max(limit * 5, 50)

    rankings: dict = {}
    if summary or moments or frame or channel_vectors:
        rankings["scenes"] = search_scenes(                    # what-happens + frame: z-normed weighted blend
            client, summary=summary, moments=moments, frame=frame, channel_vectors=channel_vectors,
            field_weights=field_weights,
            combine=combine, limit=prefetch, flt=flt, normalize=normalize, exact=exact)
    if descriptors:
        rankings["flavor"] = search_weighted_descriptors(       # flavor: weighted descriptor centroid
            client, descriptors, weights, anti_descriptors=anti_descriptors,
            anti_weights=anti_weights, anti_strength=anti_strength, limit=prefetch,
            flt=flt, exact=exact)
    if not rankings:
        raise ValueError("search needs at least one of: summary, moments, frame, channel_vectors, descriptors")
    if len(rankings) == 1:
        return next(iter(rankings.values()))[:limit]           # one method -> its ranking, untouched
    mw = method_weights or DEFAULT_METHOD_WEIGHTS
    return _rrf(rankings, mw, rrf_k, limit)                     # >1 method -> weighted rank-fuse
