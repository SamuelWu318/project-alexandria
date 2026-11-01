from utils import tags   # word -> coordinate tables (the ONLY heavy-free foundation dep; search is lazy)

# ---- Query Normalizer (read-path front door): a writer's request -> search.search input ----
# Two jobs: (1) SUMMARY-ONLY semantic — the summary drives ONLY the holistic `summary` channel; the query
# fires NO moments, so svos + subject/verb/object/setting stay empty on the query side (the old whole-summary
# svos "blob" beat was net-negative and was removed; per-beat/facet queries return only via Bridge 2's
# synthesized channel_vectors). Beat-split + svos-facet extraction were tried + DELETED; multi-beat is a
# future caller-built feature. (2) WORD -> COORD — soft controls arrive as WORDS/sliders, search wants
# NUMBERS, and that mapping lives HERE (search has no `tags` import). Output == search.search kwargs; search
# imported lazily to stay cheap.

NORMALIZER_STAGE = "summary-only"

QueryObject = dict     # kwargs for search.search: {summary, descriptors, pov, tense, prose, dialogue, tones, ...}


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


# The 6 writer-facing intensity ARCS -> an intensity WORD sequence. The writer picks a SHAPE (not valence/
# dominance); each word maps to the i-axis via tags.intensity_i and v/d are left UNSET so the soft re-rank
# compares intensity only (search.curve_dist skips None axes). Shapes are drawn over LOW/MODERATE/HIGH.
INTENSITY_ARCS: dict[str, tuple] = {
    "rising":  ("low", "moderate", "high"),   # builds to a climax
    "falling": ("high", "moderate", "low"),   # winds down from a peak
    "high":    ("high", "high", "high"),      # sustained high tension
    "low":     ("low", "low", "low"),         # sustained calm
    "hill":    ("low", "high", "low"),        # rise to a mid peak, then ebb
    "valley":  ("high", "low", "high"),       # dip to a mid lull, then climb back
}


# Fixed query tone-curve length: a writer supplies a SHAPE (a tension arc + a few emotions), not a
# per-moment transcript, so both the i sample and the VD sample resolve to this many points.
ARC_CURVE_SIZE = 3


# An intensity-arc NAME (one of the 6) -> its 3 i-values; None -> [None, None, None] (i left unspecified).
def _arc_i(name) -> list:
    if name is None:
        return [None] * ARC_CURVE_SIZE
    key = name.strip().lower() if isinstance(name, str) else name
    if key not in INTENSITY_ARCS:
        raise ValueError(f"intensity arc: {name!r} is not one of {sorted(INTENSITY_ARCS)}")
    return [tags.intensity_i(w) for w in INTENSITY_ARCS[key]]        # word -> i


# An ordered VD tone-WORD list -> 3 (valence, dominance) points. len 1 = held across all; len 2 = ends with
# their average in the middle; len 3 = in order. Empty/None -> 3 (None, None) (v/d left unspecified).
def _arc_vd(vd) -> list:
    if not vd:
        return [(None, None)] * ARC_CURVE_SIZE
    try:
        pts = [tags.tone_vd(w) for w in vd]                         # each emotion word -> (v, d)
    except KeyError as e:
        raise ValueError(f"VD curve: {e.args[0]!r} is not an in-vocab tone word") from None
    n = len(pts)
    if n == 1:
        return [pts[0]] * ARC_CURVE_SIZE                            # one emotion held across the whole curve
    if n == 2:
        mid = ((pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2)
        return [pts[0], mid, pts[1]]                                # beginning, averaged middle, end
    if n == ARC_CURVE_SIZE:
        return pts                                                  # one emotion per point, in order
    raise ValueError(f"VD curve: expected 1, 2, or {ARC_CURVE_SIZE} tone words, got {n}")


# A writer's tension pick -> a fixed 3-point query tone curve: the intensity ARC sets the i-axis, the optional
# ordered VD tone-word list (len 1/2/3) sets valence/dominance. An axis left unspecified stays None so the soft
# re-rank compares only what the writer chose (search.curve_dist skips None axes). Both empty -> None.
def arc_curve(arc=None, vd=None) -> list | None:
    if arc is None and not vd:
        return None
    i_vals = _arc_i(arc)
    vd_pts = _arc_vd(vd)
    return [[vd_pts[k][0], vd_pts[k][1], i_vals[k]] for k in range(ARC_CURVE_SIZE)]


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


# ** MAIN ** — wrap a writer's request into one search.search kwargs dict
# ONE QueryObject, no splitting: `summary` drives ONLY the holistic `summary` channel; soft WORDS become
# numbers here; hard facets + descriptors pass through; only set keys emitted (a summary-less request =
# a valid pure-BROWSE object, empty request -> {}).
# DELIBERATE: the query fires NO `moments`, so the svos + subject/verb/object/setting channels stay empty on
# the query side. Feeding the whole summary as one svos "blob" beat was NET-NEGATIVE (ablation, scene@1: blob
# .36 vs summary-only .42 — one blob matches no single indexed per-beat clause; every writer-gold tier also
# rose when it was dropped), so it was removed (Phase 11). Descriptor auto-routing stays OFF. Re-add per-beat
# svos / facet queries only via Bridge 2 (synthesized channel_vectors), not the blob — and only behind a
# measured win. Multi-beat is still a future caller-built feature.
def to_query_object(summary: str = "", *, pov=None, tense=None, book_id: str | None = None,
                    prose=None, dialogue: float | None = None, tones=None, arc=None, vd=None,
                    descriptors: list[str] | None = None) -> QueryObject:
    obj: QueryObject = {}
    s = (summary or "").strip()
    if s:
        obj["summary"] = s                                     # drives the holistic `summary` channel ONLY
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
    if t is None:
        t = arc_curve(arc, vd)                                 # else the writer's arc (i) + VD emotions (len 1/2/3)
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
        prose=None, dialogue: float | None = None, tones=None, arc=None, vd=None,
        descriptors: list[str] | None = None, limit: int = 5, **search_kwargs):
    import search                                              # unified read path (lazy)
    obj = to_query_object(summary, pov=pov, tense=tense, book_id=book_id,
                          prose=prose, dialogue=dialogue, tones=tones, arc=arc, vd=vd, descriptors=descriptors)
    if not obj:
        return []
    return search.search(client, limit=limit, **{**obj, **search_kwargs})   # normalized obj wins; extra knobs ride along
