from utils import tags   # word -> coordinate tables (the ONLY heavy-free foundation dep; search is lazy)

# ---- Query Normalizer (read-path front door): a writer's request -> search.search input ----
# Two jobs: (1) SINGLE BEAT — the summary is one scene target, searched as-is (no beat-split, no svos-facet
# extraction; both were tried + DELETED — don't resurrect; multi-beat is a future caller-built feature).
# (2) WORD -> COORD — soft controls arrive as WORDS/sliders, search wants NUMBERS, and that mapping lives
# HERE (search has no `tags` import). Output == search.search kwargs; search imported lazily to stay cheap.

NORMALIZER_STAGE = "single-beat"

QueryObject = dict     # kwargs for search.search: {summary, moments, ...}


# ---- soft word -> coord (symmetric with how derive.py builds the stored payload) ----

# One tone-curve beat -> its (tone_word, intensity_word) pair, from a (tone, intensity) 2-seq or a {"tone","intensity"} dict.
def _beat_words(beat) -> tuple:
    if isinstance(beat, dict):
        return beat.get("tone"), beat.get("intensity")
    return beat[0], beat[1]                                    # a 2-seq: (tone_word, intensity_word)


# (tone, intensity) WORD pairs -> numeric `tones` [[v,d,i],...] via tags.moment_vdi (same scale as derive's vdi_curve); out-of-vocab raises.
def tone_curve(beats) -> list | None:
    if not beats:
        return None
    out = []
    for beat in beats:
        tone, intensity = _beat_words(beat)
        try:
            v, d, i = tags.moment_vdi(tone, intensity)         # (tone->v,d) + (intensity->i), one curve point
        except KeyError as e:
            raise ValueError(f"tone curve: {e.args[0]!r} is not an in-vocab tone/intensity word") from None
        out.append([v, d, i])
    return out


# A prose-register word -> its `prose` float (already-numeric input passes through; None -> None).
# Raises ValueError naming an out-of-vocab word.
def prose_level(word) -> float | None:
    if word is None:
        return None
    if isinstance(word, (int, float)):
        return float(word)                                     # already a coord (a slider sent the number)
    try:
        return tags.prose_coord(word)
    except KeyError:
        raise ValueError(f"prose: {word!r} is not an in-vocab prose-register word") from None


# ** MAIN ** — wrap a writer's request into one search.search kwargs dict (single beat)
# ONE QueryObject, no splitting: `summary` is both the holistic vector and the single `moments` beat (fires
# svos); soft WORDS become numbers here; hard facets + descriptors pass through; only set keys emitted (a
# summary-less request = a valid pure-BROWSE object, empty request -> {}).
# DELIBERATE: svos facets left empty + descriptor auto-routing OFF (both declined, no measured win) — don't re-add without one.
def to_query_object(summary: str = "", *, pov=None, tense=None, book_id: str | None = None,
                    prose=None, dialogue: float | None = None, tones=None,
                    descriptors: list[str] | None = None) -> QueryObject:
    obj: QueryObject = {}
    s = (summary or "").strip()
    if s:
        obj["summary"] = s
        obj["moments"] = [{"sentence": s}]                     # single beat -> the svos channel
    if descriptors:
        obj["descriptors"] = descriptors
    if pov is not None:
        obj["pov"] = pov                                       # hard facets pass straight to search
    if tense is not None:
        obj["tense"] = tense
    if book_id is not None:
        obj["book_id"] = book_id
    p = prose_level(prose)                                     # prose WORD -> float
    if p is not None:
        obj["prose"] = p
    if dialogue is not None:
        obj["dialogue"] = float(dialogue)                      # already-numeric slider
    t = tone_curve(tones)                                      # tone/intensity WORD-curve -> [[v,d,i]...]
    if t is not None:
        obj["tones"] = t
    return obj


# ** MAIN ** — the ONE normalize entry: app/webtest/evals turn a raw request into search kwargs here
# Normalize a writer's request into a single QueryObject ready for search. No decomposition — one beat.
def normalize(summary: str = "", **request) -> QueryObject:
    return to_query_object(summary, **request)


# ** MAIN ** — convenience: a writer's request -> single-beat search results, one call
# Normalize a request and run it through the unified read path. The soft words are mapped to numbers by
# to_query_object; extra search knobs (limit, method_weights, combine, normalize, weights, anti_*) ride
# **search_kwargs. Returns ScoredPoints best-first (a wholly empty request -> []).
def run(client, summary: str = "", *, pov=None, tense=None, book_id: str | None = None,
        prose=None, dialogue: float | None = None, tones=None,
        descriptors: list[str] | None = None, limit: int = 5, **search_kwargs):
    import search                                              # unified read path (lazy)
    obj = to_query_object(summary, pov=pov, tense=tense, book_id=book_id,
                          prose=prose, dialogue=dialogue, tones=tones, descriptors=descriptors)
    if not obj:
        return []
    return search.search(client, limit=limit, **{**obj, **search_kwargs})   # normalized obj wins; extra knobs ride along
