import re
from pathlib import Path, PurePath

from utils import read_json, write_json, log
from utils import tags   # word->coord tables: the ONE home for tone/intensity/prose coordinates (never hardcode a coord here)

# ---- Stage 3b: derivation (mechanical, no LLM) ----
# in:  scenes/pg{code}-s.json after enrich.py (LLM fields filled; every `source:"derived"` field still null)
# out: same file with the derived payload filled IN PLACE — svos (the moment sentences), the four S/V/O/S
# facet lists, vdi_curve (each moment's tone+intensity WORDS -> a (valence, dominance, intensity) sample),
# prose_register (register WORD -> float), dialogue_ratio (quote-char ratio), arc (the vdi intensity-axis
# shape). Everything here is a pure word->number / text->number transform, so re-tuning a utils/tags.py
# table re-runs THIS stage only (a payload refresh), never enrichment (PLAN §3.5). No LLM, no DB — records
# only. Two doors: derive_records (in-place, idempotent — index.py's pre-embed safety net) and derive_file
# (read -> derive -> write). The word->coord math lives entirely in utils.tags; derive never hardcodes one.

# ---- field derivations (each a pure recompute -> idempotent) ----

# ** LOCKED **
# Scene prose with markup stripped and whitespace collapsed (input to the dialogue-ratio measure, not stored).
def _plain(text_html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text_html or "")).strip()


_STRAIGHT_QUOTE = re.compile(r'"([^"]*)"')     # a straight "..." pair
_SMART_QUOTE = re.compile(r'“([^”]*)”')   # a smart “...” pair (U+201C .. U+201D)

# Fraction of the (markup-stripped) prose that sits inside matched quote pairs — straight AND smart; [0,1], None if no prose.
def _dialogue_ratio(text_html: str | None) -> float | None:
    prose = _plain(text_html or "")
    if not prose:
        return None
    quoted = sum(len(s) for s in _STRAIGHT_QUOTE.findall(prose))   # chars inside straight pairs
    quoted += sum(len(s) for s in _SMART_QUOTE.findall(prose))     # + chars inside smart pairs
    return max(0.0, min(1.0, quoted / len(prose)))


# One facet's terms across the moments: dedup case-insensitively, order-preserving; [] -> None (the old embed._derive_frame behaviour, kept per the 0b ablation).
def _facet(moments: list[dict], field: str) -> list[str] | None:
    seen, out = set(), []
    for m in moments:
        t = (m.get(field) or "").strip() if isinstance(m, dict) else ""
        key = t.lower()
        if t and key not in seen:
            seen.add(key)
            out.append(t)
    return out or None


# The scene's affective arc as raw samples: each moment's (tone, intensity) WORDS -> a [valence, dominance, intensity] point; empty -> None.
def _vdi_curve(moments: list[dict]) -> list[list[float]] | None:
    curve = [list(tags.moment_vdi(m["tone"], m["intensity"]))   # utils.tags: two WORDS -> one (v,d,i) point
             for m in moments]
    return curve or None


ARC_FLAT_BAND = 0.15   # net |Δ intensity| within this reads as "steady"; a mid-curve reversal beyond it is a "turn"

# Classify the vdi_curve's intensity (i) axis into an Arc word: a mid-curve peak/valley beyond the band -> turn, else net rise/fall past the band -> rising/falling, else steady (<2 samples -> steady).
def _arc(vdi_curve: list[list[float]] | None) -> str:
    i = [pt[2] for pt in vdi_curve] if vdi_curve else []
    if len(i) < 2:
        return tags.Arc.STEADY.value
    peak = max(i) - i[0] > ARC_FLAT_BAND and max(i) - i[-1] > ARC_FLAT_BAND     # rose then fell back
    valley = i[0] - min(i) > ARC_FLAT_BAND and i[-1] - min(i) > ARC_FLAT_BAND   # fell then rose back
    if peak or valley:
        return tags.Arc.TURN.value
    net = i[-1] - i[0]
    if net > ARC_FLAT_BAND:
        return tags.Arc.RISING.value
    if net < -ARC_FLAT_BAND:
        return tags.Arc.FALLING.value
    return tags.Arc.STEADY.value


# Fill ONE record's derived payload in place. Order is load-bearing: arc reads vdi_curve, so it runs last.
def _derive_record(rec: dict) -> dict:
    moments = rec.get("moments") or []
    rec["svos"] = [m["sentence"] for m in moments
                   if isinstance(m, dict) and m.get("sentence")] or None     # the moment sentences = the svos beats
    for facet in ("subject", "verb", "object", "setting"):
        rec[facet] = _facet(moments, facet)                                  # roll moment parts up per facet
    rec["vdi_curve"] = _vdi_curve(moments)
    pw = rec.get("prose_word")
    rec["prose_register"] = tags.prose_coord(pw) if pw else None             # utils.tags: register WORD -> float
    rec["dialogue_ratio"] = _dialogue_ratio(rec.get("text_html"))
    rec["arc"] = _arc(rec["vdi_curve"])
    return rec


# ---- doors ----

# ** MAIN ** — index.py (Phase 7) runs this as its pre-embed safety net; the Stage-3 driver runs it per book
# Fill the derived payload for every record IN PLACE and return them. IDEMPOTENT: each field is a pure
# recompute from moments / prose_word / text_html, so a second run reproduces the first byte-for-byte.
def derive_records(records: list[dict]) -> list[dict]:
    for r in records:
        _derive_record(r)                # svos + facets + vdi_curve + prose_register + dialogue_ratio + arc
    return records


# ** MAIN ** — tests + the standalone derive step run one book through here
# Derive one scenes json in place: read -> derive_records -> write; returns the records ([] if the file is empty/missing).
def derive_file(path: Path) -> list[dict]:
    records = read_json(path, [])                     # read_write: the book's flat scene records
    if not records:
        return records
    derive_records(records)
    write_json(path, records)                         # read_write: rewrite the derived json in place
    log.done(f"derived {len(records)} scenes -> {PurePath(path).name}")
    return records
