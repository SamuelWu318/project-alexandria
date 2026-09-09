from __future__ import annotations
import uuid
import numpy as np
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from utils import SrcPaths, schema   # scene-record registry — drives the named-vector set

# ---- Qdrant contract: the ONE home of the vector-store primitives ----
# The read path (search.py) and the write path (index.py) BOTH import this module. Extracting the
# contract here removes the old wrong-direction coupling where the writer imported 7 symbols out of the
# reader (principle #4 in CLAUDE.md). This file owns the shared vocabulary — collection name, named-vector
# set, embedder, stable point id, payload-filter builders — and NOTHING here queries or indexes; the
# read/write LOGIC lives in search.py / index.py. Pulls in only qdrant + fastembed (no LLM), so both
# sides stay light. Invariants (CLAUDE.md): EMBED_MODEL MUST match the model the index was built with;
# point_id is a stable uuid5 so a re-index overwrites; bge is asymmetric (queries get QUERY_PREFIX,
# indexed passages + descriptor queries stay raw); a multivector field MUST be queried with a matrix.

# ---- config ----

# Qdrant collection name — the read/write join key (index.py writes here, search.py queries here).
COLLECTION = "scenes"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"     # MUST match the model the index was built with
# named vectors == the registry's vector:true fields (summary + descriptors single; svos + the four
# subject/verb/object/setting facets multivector — the committed 7-vector set, PLAN Phase 0b).
VECTOR_NAMES = schema.VECTOR_NAMES
# multivector:true fields: a LIST of per-item vectors scored by MAX_SIM (max-pool). MUST be queried
# with a matrix (list of vectors), even a 1-row one — a flat vector is rejected by the index.
MULTIVECTOR_NAMES = frozenset(schema.MULTIVECTOR_NAMES)

# order-aware svos (PLAN §3.2): the svos beats carry SVOS_POS_DIMS extra positional dims, so the svos
# named vector is WIDER than the bare embed dim; every other named vector is the bare dim. index.py sizes
# its Qdrant params through vector_size; a re-index rebuilds the collection if this width drifts.
SVOS_POS_DIMS = 2

# ** LOCKED **  ** MAIN ** — index.py sizes each named vector's Qdrant params through here
# Stored size of a named vector at a given base embed dim: svos gets SVOS_POS_DIMS positional dims appended.
def vector_size(name: str, base_dim: int) -> int:
    return base_dim + SVOS_POS_DIMS if name == "svos" else base_dim

# Qdrant payload label for subject-branch filtering: subject_filter reads it; index.py stamps + indexes it.
SUBJECT_PATHS_FIELD = "subject_paths"

# bge query-side instruction prefix (summary/svos/frame queries only). Set to "" to A/B without re-indexing.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# stable per-scene Qdrant id namespace: uuid5(NAMESPACE, scene_id) -> the same point every re-index.
NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "projectalexandria.scenes")


# ---- point id ----

# ** LOCKED **  ** MAIN ** — index.py + schema.sync_qdrant address points by this id
# Stable Qdrant point id for a scene (uuid5, so a re-index of the same scene_id overwrites its point).
def point_id(scene_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, scene_id))


# ---- embedder (same model for index + query) ----

_EMBEDDER = None


# ** LOCKED **
# Lazily construct and cache the shared TextEmbedding model (first call downloads it).
def _embedder() -> TextEmbedding:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = TextEmbedding(model_name=EMBED_MODEL)
    return _EMBEDDER


# ** LOCKED **  ** MAIN ** — index.py builds every stored vector, search.py embeds every query, through here
# Embed a batch of texts into plain float lists.
def embed(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in _embedder().embed(texts)]


# ---- order-aware svos beat encoding (PLAN §3.2) — index + query MUST share this or svos scores are junk ----
# Event ORDER can't come from MAX_SIM (it is permutation-invariant), so it is baked into each svos beat
# vector: a normalized-position component is appended. For beat k of a K-beat sequence, u = k/(K-1) (u=0
# when K=1), and the beat vector is
#     v = [ sqrt(1-β)·unit(sem) ; sqrt(β)·p(u) ] ,   p(u) = [cos(freq·u), sin(freq·u)] ,   ||v|| = 1
# so cos(v_q, v_s) = (1-β)·sem_cos + β·pos_cos, with pos_cos = cos(freq·|u_q-u_s|) in [0,1] — position
# never flips a sign, it only WITHHOLDS reward. MAX_SIM over these prefers beats that agree BOTH in meaning
# AND in relative position (in-order), but SOFTLY: a strong out-of-order match still wins if its sem_cos
# clears the β gap. β is the order-strength ⟷ recall-robustness knob (β=0 == today's order-free svos). u is
# taken over each side's OWN beat count, so a query's beats and a scene's compare by relative position —
# never by raw index or a forced common length. svos ONLY — subject/verb/object/setting stay order-free sets.

# order strength in [0,1); tune on the gold (PLAN §3.2), β=0 is the order-free fallback.
SVOS_POS_BETA = 0.2
# p(u) sweeps a quarter turn over u in [0,1] so pos_cos = cos(freq·Δu) stays >= 0 across the whole range.
_SVOS_POS_FREQ = np.pi / 2


# ** LOCKED **
# L2-normalize a vector to unit length; a zero vector is returned unchanged (guards divide-by-zero).
def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n else v


# ** LOCKED **
# The positional component sqrt(β)·p(u) for a beat at fractional position u in [0,1] (SVOS_POS_DIMS long).
def _svos_position(u: float) -> list[float]:
    s = SVOS_POS_BETA ** 0.5
    return [s * float(np.cos(_SVOS_POS_FREQ * u)), s * float(np.sin(_SVOS_POS_FREQ * u))]


# ** LOCKED **  ** MAIN ** — index.py builds every stored svos matrix, search.py every svos query matrix, through here
# Order-aware svos beat matrix for ONE sequence of beat sentences (already bge-prefixed on the query side,
# raw on the index side — the bge asymmetry stays at the call site). Embeds the beats, then appends the
# shared positional component per beat in sequence order (§3.2). Empty in -> []. No other named vector
# passes through here (the facets stay order-free plain embeds).
def svos_beat_vectors(sentences: list[str]) -> list[list[float]]:
    if not sentences:
        return []
    sem = embed(sentences)                                     # one batched bge embed over this sequence's beats
    keep = (1.0 - SVOS_POS_BETA) ** 0.5
    n = len(sem)
    out = []
    for k, e in enumerate(sem):
        u = k / (n - 1) if n > 1 else 0.0                      # normalized position over THIS sequence
        out.append((_unit(e) * keep).tolist() + _svos_position(u))   # [sqrt(1-β)·unit(sem) ; sqrt(β)·p(u)]
    return out


# ---- client ----

# ** MAIN ** — index, search, evals, schema.sync_qdrant, webtest open the on-disk store here
# Open the on-disk Qdrant client (single-process; first call also downloads the embed model).
def open_client() -> QdrantClient:
    return QdrantClient(path=str(SrcPaths.QDRANT_DIR))


# ---- payload filters (hard pre-filters shared by the read path) ----

# ** LOCKED **
# Pick the retrieval STRATEGY: exact brute-force over the filtered set (few books) vs the filtered HNSW walk (broad).
def _search_params(exact: bool):
    return models.SearchParams(exact=True) if exact else None


# ** LOCKED **
# Hard-filter on one categorical payload facet: a single value (MatchValue) or any-of a list (MatchAny); None -> no restriction.
def facet_filter(key: str, value) -> models.Filter | None:
    if not value:
        return None
    if isinstance(value, str):
        match = models.MatchValue(value=value)
    else:
        vals = [v for v in value if v]
        if not vals:
            return None
        match = models.MatchAny(any=vals)
    return models.Filter(must=[models.FieldCondition(key=key, match=match)])


# ** MAIN ** — tests + webtest + search restrict a search to one book here
# Filter that restricts a search to one book, or None for all books.
def book_filter(book_id: str | None) -> models.Filter | None:
    if not book_id:
        return None
    return models.Filter(must=[models.FieldCondition(
        key="book_id", match=models.MatchValue(value=book_id))])


# ** MAIN ** — webtest + search restrict a search to one subject branch here
# Restrict a search to one subject branch via the indexed `subject_paths` payload label (one exact keyword term).
def subject_filter(branch) -> models.Filter | None:
    if not branch:
        return None
    suffix = branch if isinstance(branch, str) else " -- ".join(reversed(list(branch)))
    return models.Filter(must=[models.FieldCondition(
        key=SUBJECT_PATHS_FIELD, match=models.MatchValue(value=suffix))])


# ** LOCKED **  ** MAIN ** — index.py normalizes each multivector field, search.py cleans query terms, through here
# Normalize a multivector field value to a clean list of items (bare string or list -> non-empty trimmed items; None -> []).
def _as_terms(v) -> list[str]:
    if v is None:
        return []
    items = [v] if isinstance(v, str) else list(v)
    return [t.strip() for t in items if isinstance(t, str) and t.strip()]
