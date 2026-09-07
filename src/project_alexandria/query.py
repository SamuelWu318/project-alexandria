# ---- Query Normalizer (read-path front door): raw summary -> search.search input ----
# The seam between "what a human typed" and search()'s structured input. SINGLE BEAT ONLY: a summary is
# treated as ONE scene target and searched as-is — no automatic decomposition. Output contract == kwargs
# for search.search(): {summary, moments, ...}.
#
# Multi-beat ("a duel AND a pardon") is a SEPARATE, EXPLICIT feature (not this file's job yet): the caller
# builds the beats itself and searches each. No sentence-splitting / clustering here. search is imported
# lazily (like evals) so importing this module stays cheap (no qdrant/fastembed load).

NORMALIZER_STAGE = "single-beat"

QueryObject = dict     # kwargs for search.search: {summary, moments, ...}


# ** MAIN ** — wrap a raw summary into one search.search kwargs dict (single beat)
# Turn ANY given text into ONE QueryObject with no splitting: the whole text is the `summary` vector, and
# one `moment` carries the same text so the svos channel fires too. A sparse sentence ("a duel where the
# challenger beats a cocky opponent") and a complex multi-clause summary go through this SAME path — the
# only difference is the length of the one beat. Empty input -> {}.
# DELIBERATE: the moment's svos FACETS (subject/verb/object/setting) are left empty, so the per-facet
# query channels do not fire. Filling them from the text was considered and DECLINED — it would add no
# information the summary + svos channels don't already carry, and query-time extraction (heuristic or LLM)
# is not worth its cost/dep here. Do NOT re-add facet extraction without a measured retrieval win.
def to_query_object(summary: str) -> QueryObject:
    s = (summary or "").strip()
    if not s:
        return {}
    return {"summary": s, "moments": [{"sentence": s}]}


# ** MAIN ** — the ONE public entry: app/webtest/evals normalize raw text here
# Normalize a raw summary into a single QueryObject ready for search. No decomposition — one beat.
def normalize(summary: str) -> QueryObject:
    return to_query_object(summary)


# ** MAIN ** — convenience: raw summary -> single-beat search results, one call
# Normalize a summary and run it through the unified read path. Passthrough filters/knobs via **kw.
# Returns ScoredPoints best-first (empty summary -> []).
def run(client, summary: str, *, limit: int = 5, **search_kwargs):
    import search                                           # unified read path (lazy)
    obj = to_query_object(summary)
    if not obj:
        return []
    return search.search(client, limit=limit, **{**obj, **search_kwargs})
