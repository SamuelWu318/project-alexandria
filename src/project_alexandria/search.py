import numpy as np
from qdrant_client import QdrantClient, models
from utils.vectorstore import (COLLECTION, MULTIVECTOR_NAMES, QUERY_PREFIX, embed,   # the Qdrant contract,
                               open_client, book_filter, subject_filter, facet_filter,   # defined ONCE in
                               _search_params, _as_terms)                                 # utils/vectorstore.py

# ---- Read path: query the scene vector DB (import THIS from the app / API) ----
# Pulls in only qdrant + fastembed — NO LLM, NO segmentation — so the query path stays light. The
# vector-store CONTRACT (COLLECTION, vector set, embedder, point id, filters) lives in
# utils/vectorstore.py — the ONE home both this reader and the writer (index.py) import; extracting it
# removed the old embed->search coupling (principle #4). This file is read LOGIC only. Invariants
# (CLAUDE.md): EMBED_MODEL must match the index; point_id is a stable uuid5; bge is asymmetric (queries prefixed).
#
# Four stages (search(), the one door): (1) HARD pre-filter — book / subject / pov / tense, each an
# exclude, never softened; (2) SEMANTIC pool — the what-happens (summary + svos + S/V/O/S) and flavor
# (descriptors) methods, RRF-merged; (3) SOFT re-rank — tilt the pool by the prose / dialogue sliders and
# the tone CURVE (never excludes); (4) slice to limit. pov/tense replaced the retired tone/intensity/arc
# hard filters; the per-field `weight` stack is retired (PLAN D3) — the semantic blend is weight-free.


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


# The flavor channel: descriptor search with per-descriptor weights and optional anti-descriptors (weighted centroid of INDIVIDUAL embeddings, anti = subtraction). Returns ScoredPoints best-first.
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


# ** LOCKED **  ** MAIN ** — webtest ANDs its hard filters through here
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


# Fuse the z-scored channels into one ranking, WEIGHT-FREE (PLAN D3): combine="sum" = additive over the equal channels (DEFAULT — holds the 0b gold: book@1 .99), "max" = greatest single channel match (kept for A/B; lost the gold, book@1 .86). Returns [(id, score)] best-first.
def blend_channels(scored: dict, combine: str = "sum") -> list:
    normed, ids = scored["channels"], scored["ids"]
    if not ids or not normed:
        return []
    fused = []
    for i in ids:
        contribs = [normed[name][i] for name in normed]        # each channel already z-scored over the pool
        fused.append((i, max(contribs) if combine == "max" else sum(contribs)))
    fused.sort(key=lambda t: t[1], reverse=True)
    return fused


# What-happens + frame search: the summary, svos, and subject/verb/object/setting channels z-normalized per channel then fused weight-free (no channel overpowers by scale; combine="sum" additive is the default that holds the 0b gold, "max" = greatest single match). Returns ScoredPoints, pool-relative score.
def search_scenes(client: QdrantClient, *, summary: str | None = None, moments=None, frame=None,
                  channel_vectors: dict | None = None, combine: str = "sum",
                  limit: int = 5, flt: models.Filter | None = None,
                  prefetch: int | None = None, normalize: str | None = "zscore",
                  exact: bool = False):
    prefetch = prefetch or max(limit * 5, 50)
    scored = score_channels(client, summary=summary, moments=moments, frame=frame,   # prefetch + z-norm each channel
                            channel_vectors=channel_vectors,                         # pre-embedded (adapter) vectors, if any
                            normalize=normalize, flt=flt, prefetch=prefetch, exact=exact)
    fused = blend_channels(scored, combine)                                          # weight-free blend over the pool
    out = []
    for i, sc in fused[:limit]:
        p = scored["cand"][i]
        p.score = sc
        out.append(p)
    return out


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


# ---- stage 3: SOFT re-rank (tilt the pool by prose / dialogue sliders + the tone CURVE; never excludes) ----
# Soft inputs are NUMERIC at the search boundary — `tones` = an ordered list of 1-5 [v,d,i] points,
# `prose`/`dialogue` = floats in [0,1] or None. The word->coord mapping (tone/intensity words ->
# tags.moment_vdi, prose word -> tags.prose_coord) is the NORMALIZER's job (query.py), NOT search's, so
# this stage stays a pure numeric tilt with no `tags` import. Re-rank runs over the semantic pool's
# PAYLOAD (no extra vector work), so search() prefetches deep (SOFT_PREFETCH) when a slider is set.

# soft-axis mix (which soft signal counts how much) — D5 defaults, fixed in code.
W_TONE, W_PROSE, W_DIA = 0.5, 0.3, 0.2
# affect-axis mix inside a curve distance ("rising" is mostly an intensity claim) — D5 defaults.
CURVE_WV, CURVE_WD, CURVE_WI = 0.3, 0.2, 0.5
# global soft strength: a worst-case soft miss (penalty 1.0) costs LAMBDA z of semantic score (~±2),
# so soft reorders near-ties but never overrides a clear semantic winner. Fixed in code.
LAMBDA = 0.5
# generous prefetch when any slider is set, so the re-rank has a deep pool to reorder.
SOFT_PREFETCH = 200


# ** LOCKED **
# Resample the query tone curve onto m candidate moments: linear-interp each [v,d,i] axis at u = j/(m-1); a 1-point query is flat, a 1-moment candidate samples the curve's midpoint.
def resample(tones, m: int) -> list:
    k = len(tones)
    if k == 1:
        return [[float(a) for a in tones[0]] for _ in range(m)]

    def at(u: float) -> list:                               # sample the k-point curve at u in [0,1]
        x = u * (k - 1)
        lo = int(np.floor(x)); hi = min(lo + 1, k - 1); f = x - lo
        return [float(tones[lo][a]) * (1 - f) + float(tones[hi][a]) * f for a in range(3)]

    if m == 1:
        return [at(0.5)]                                    # one moment -> the curve's single midpoint summary
    return [at(j / (m - 1)) for j in range(m)]


# ** LOCKED **
# Mean over positions of the per-axis-weighted Euclidean distance between two m×3 curves (each in [0,1]); result in [0,1].
def curve_dist(U: list, S: list) -> float:
    if not U:
        return 0.0
    wsum = CURVE_WV + CURVE_WD + CURVE_WI
    acc = 0.0
    for u, s in zip(U, S):
        d2 = CURVE_WV * (u[0] - s[0]) ** 2 + CURVE_WD * (u[1] - s[1]) ** 2 + CURVE_WI * (u[2] - s[2]) ** 2
        acc += (d2 / wsum) ** 0.5
    return acc / len(U)


# Soft penalty for one candidate: sum ONLY the soft axes present on BOTH the query and this candidate, renormalized — a candidate missing an axis is neither rewarded nor punished on it. Returns [0,1].
def soft_penalty(payload: dict, prose, dialogue, u_by_len) -> float:
    pen = wsum = 0.0
    pr = payload.get("prose_register")
    if prose is not None and pr is not None:
        pen += W_PROSE * abs(float(pr) - prose); wsum += W_PROSE
    dr = payload.get("dialogue_ratio")
    if dialogue is not None and dr is not None:
        pen += W_DIA * abs(float(dr) - dialogue); wsum += W_DIA
    curve = payload.get("vdi_curve")
    if u_by_len is not None and curve:
        S = [[float(a) for a in pt] for pt in curve]
        pen += W_TONE * curve_dist(u_by_len(len(S)), S); wsum += W_TONE
    return pen / wsum if wsum else 0.0


# Stage-3 soft re-rank: z-normalize the pool's semantic scores, subtract LAMBDA*soft_penalty, resort. All soft inputs unset -> the pool is returned UNTOUCHED (identical ranking to no-soft).
def _soft_rerank(pool: list, prose, dialogue, tones) -> list:
    if not pool or (prose is None and dialogue is None and not tones):
        return pool
    u_by_len = None
    if tones:
        memo: dict = {}                                     # moment cap 6 -> <=6 distinct curve lengths
        def u_by_len(m: int) -> list:
            if m not in memo:
                memo[m] = resample(tones, m)
            return memo[m]
    scores = np.asarray([p.score for p in pool], dtype=np.float32)
    std = float(scores.std())
    z = (scores - float(scores.mean())) / std if std > 1e-9 else np.zeros_like(scores)   # no spread -> soft alone orders
    for p, zc in zip(pool, z):
        p.score = float(zc) - LAMBDA * soft_penalty(p.payload, prose, dialogue, u_by_len)
    pool.sort(key=lambda p: p.score, reverse=True)
    return pool


# ** MAIN ** — the ONE search entry: imported by tests, evals, webtest
# Unified scene search, four stages: (1) HARD pre-filter (ANDed excludes: book_id, subject_branch, pov, tense — each a single value or any-of a list); (2) SEMANTIC pool — text (summary/moments/frame) and/or channel_vectors (pre-embedded per-channel query vectors from the learned adapter; override the text-derived ones) drive the what-happens method, descriptors drive the flavor method, merged by weighted RRF; NO semantic input but a hard filter or a soft input -> BROWSE the filtered set; (3) SOFT re-rank — the numeric prose/dialogue sliders + the tone CURVE (`tones` = ordered [v,d,i] points) tilt the pool (skipped when all three are unset); (4) slice to limit. Knobs — method_weights (scenes vs flavor RRF), combine (sum/max vector blend; sum default holds the 0b gold), normalize. Returns ScoredPoints best-first.
def search(client: QdrantClient, *, summary: str | None = None, moments=None, frame=None,
           channel_vectors: dict | None = None,
           descriptors: list[str] | None = None, weights: list[float] | None = None,
           anti_descriptors: list[str] | None = None, anti_weights: list[float] | None = None,
           anti_strength: float = 1.0, book_id: str | None = None, subject_branch=None,
           pov=None, tense=None,
           prose: float | None = None, dialogue: float | None = None, tones=None,
           flt: models.Filter | None = None, limit: int = 5, prefetch: int | None = None,
           normalize: str | None = "zscore", combine: str = "sum",
           method_weights: dict | None = None, rrf_k: int = 60, exact: bool = False):
    if flt is None:
        flt = _and_filters(book_filter(book_id), subject_filter(subject_branch),   # AND book + subject +
                           facet_filter("pov", pov), facet_filter("tense", tense))  # pov/tense hard facets
    soft_on = prose is not None or dialogue is not None or bool(tones)
    prefetch = prefetch or (SOFT_PREFETCH if soft_on else max(limit * 5, 50))       # deep pool when a slider is set

    rankings: dict = {}
    if summary or moments or frame or channel_vectors:
        rankings["scenes"] = search_scenes(                    # what-happens + frame: z-normed weight-free blend
            client, summary=summary, moments=moments, frame=frame, channel_vectors=channel_vectors,
            combine=combine, limit=prefetch, flt=flt, normalize=normalize, exact=exact)
    if descriptors:
        rankings["flavor"] = search_weighted_descriptors(       # flavor: weighted descriptor centroid
            client, descriptors, weights, anti_descriptors=anti_descriptors,
            anti_weights=anti_weights, anti_strength=anti_strength, limit=prefetch,
            flt=flt, exact=exact)

    if rankings:
        if len(rankings) == 1:
            pool = list(next(iter(rankings.values())))          # one method -> its ranking
        else:
            mw = method_weights or DEFAULT_METHOD_WEIGHTS
            pool = _rrf(rankings, mw, rrf_k, prefetch)          # >1 method -> weighted rank-fuse (to prefetch depth)
    elif soft_on or flt is not None:
        pool = [models.ScoredPoint(id=r.id, version=0, score=0.0, payload=r.payload)   # pure BROWSE: no semantic
                for r in client.scroll(COLLECTION, scroll_filter=flt,                   # input, so scroll the
                                       limit=prefetch, with_payload=True)[0]]           # hard-filtered set + soft-rank it
    else:
        raise ValueError("search needs a semantic input (summary/moments/frame/channel_vectors/descriptors), "
                         "a hard filter (book_id/subject_branch/pov/tense), or a soft input (prose/dialogue/tones)")

    pool = _soft_rerank(pool, prose, dialogue, tones)           # stage 3: tilt by sliders + tone curve (or untouched)
    return pool[:limit]
