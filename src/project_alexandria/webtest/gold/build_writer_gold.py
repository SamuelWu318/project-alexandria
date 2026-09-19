"""Build the writer-realistic gold query set (Phase 11) -> webtest/gold/writer_queries.json.

Per DISTINCT target scene in test_queries.json, emit THREE queries carrying ONLY what a writer supplies —
summary, descriptors, pov, tense, and (similar + ambiguous) an intensity ARC. NO facets
(subject/verb/object/svos): the writer never gives them, so the gold must not either.

  similar    the existing `specific` summary (clear, detailed) + its descriptors
  ambiguous  the existing `ambiguous` summary (general language, same plot) + descriptors
  sparse     a hand-authored terse summary (general language, only 1-2 major beats) — NO arc

The arc is CLASSIFIED from the target scene's real vdi_curve i-axis into one of query.INTENSITY_ARCS
(rising/falling/high/low/hill/valley), so the writer's arc pick == the scene's actual shape. Authored
in-repo (no LLM call): the `specific`/`ambiguous` summaries are reused as-is; SPARSE_QUERIES below is the
only new prose and is the tuning surface. Run:  python webtest/gold/build_writer_gold.py  (PYTHONPATH=src).
"""
import json
from pathlib import Path

from derive import ARC_FLAT_BAND                 # reuse the exact steady/turn deadband derive classified scenes with
from utils.storage import SrcPaths

SOURCE = SrcPaths.ROOT_DIR / "src" / "project_alexandria" / "webtest" / "gold" / "test_queries.json"
OUT = SOURCE.with_name("writer_queries.json")


# ---- the one new prose: a terse, general-language, 1-2 beat query per target scene (the tuning surface) ----
SPARSE_QUERIES = {
    "103-105": "A man burns his own ship for fuel to reach a deadline.",
    "1342-47": "A woman rejects an arrogant suitor's proposal.",
    "1661-82": "Two men keep watch at night and a snake kills a schemer.",
    "2147-125": "A painter's obsession finishes a portrait as his wife dies.",
    "215-16": "A sled dog defeats its rival for dominance of the pack.",
    "2701-85": "A captain rallies his crew to hunt a white whale.",
    "4300-167": "Men in a bar mock an absent acquaintance.",
    "64317-7": "A man watches a stranger reach toward a distant light across the water.",
    "84-23": "A man gives life to a creature and recoils in horror.",
    "98-196": "A condemned prisoner awaits execution and steadies himself.",
    "103-34": "A man rescues a woman from being sacrificed and escapes with her.",
    "1342-29": "A family mocks a pompous cousin's letter.",
    "1342-44": "A woman is embarrassed by her family at a dance.",
    "1661-70": "A thief admits hiding a stolen gem inside a goose.",
    "2147-82": "An escaped ape murders two women.",
    "215-41": "A dog hauls an enormous load to win a wager.",
    "2701-307": "A captain is dragged under and his ship goes down.",
    "4300-0": "A man stages a mocking ritual and taunts his companion.",
    "4300-91": "A flippant young man disrupts scholars debating literature.",
    "64317-32": "A rich man's display of fine shirts overwhelms a woman.",
    "84-98": "A monster murders a bride on her wedding night.",
    "98-57": "An aristocrat tosses a coin after his carriage kills a child.",
    "103-93": "Attackers raid a moving train and a servant saves the cars.",
    "1342-152": "A woman reads a letter revealing a proud man secretly saved her family.",
    "1661-60": "A missing gentleman is found beneath a beggar's disguise.",
    "2147-50": "Treasure hunters dig up a chest of gold.",
    "215-47": "A relentless dog runs down a moose over days until it collapses.",
    "2701-45": "A man argues with his companion over a religious fast.",
    "4300-179": "A man watches a woman on a beach as fireworks burst.",
    "64317-50": "Two rivals force a woman to choose between them.",
    "84-83": "A creator destroys his second creature as the first looks on.",
    "98-21": "Visitors find a broken old prisoner making shoes in a garret.",
    "103-6": "A man wagers he can travel around the world in eighty days.",
    "1342-39": "Men at a ball debate whom to dance with.",
    "1342-41": "A woman trades barbs with a proud man while dancing.",
    "1661-12": "A pawnbroker answers a strange advertisement for red-headed men.",
    "1661-13": "A man is hired for pointless copying work that abruptly stops.",
    "2147-36": "A balloonist falls and crashes among strange moon beings.",
    "215-6": "A newcomer dog is torn apart by a savage pack.",
    "2701-22": "A chaplain climbs to his pulpit and withdraws the ladder.",
    "4300-30": "A man tends his cat and visits the outhouse one morning.",
    "64317-64": "A man is shot dead floating in his pool.",
    "84-62": "A lonely outcast begs a blind man for acceptance.",
    "98-15": "A crowd scrambles to drink wine spilled in the street.",
    "103-107": "A traveler arrives minutes too late and loses his bet.",
    "1342-52": "A woman finally accepts a proud man's proposal.",
    "1661-1": "A detective dazzles a visitor with deductions as a masked client arrives.",
    "2147-120": "A castaway is flung aboard a phantom ship and hides.",
    "215-48": "A dog returns to avenge his murdered master.",
    "2701-224": "A boy is abandoned alone in the open sea.",
    "4300-333": "A woman lies awake before dawn drifting through desire and memory.",
    "64317-70": "A handful of mourners stand at a grave in the rain.",
    "84-110": "A man finds a creature grieving over its dead creator.",
    "84-6": "A captain befriends a grief-stricken rescued man.",
    "98-119": "A mob storms and breaches a prison fortress.",
}


# ---- arc classification (6-way, mirrors derive._arc's 4-way + a mean split of the flat band) ----

# The vdi_curve's intensity (i) axis -> one of query.INTENSITY_ARCS. Mid-curve peak/valley beyond the band
# -> hill/valley; else net rise/fall past the band -> rising/falling; else flat -> high/low by mean level.
def classify_arc(vdi_curve) -> str:
    i = [float(p[2]) for p in (vdi_curve or [])]
    if len(i) < 2:
        return "high" if (i and i[0] >= 0.5) else "low"
    peak = max(i) - i[0] > ARC_FLAT_BAND and max(i) - i[-1] > ARC_FLAT_BAND     # rose then fell back
    valley = i[0] - min(i) > ARC_FLAT_BAND and i[-1] - min(i) > ARC_FLAT_BAND   # fell then rose back
    if peak and not valley:
        return "hill"
    if valley and not peak:
        return "valley"
    if peak and valley:                                                         # both -> the larger excursion wins
        return "hill" if (max(i) - max(i[0], i[-1])) >= (min(i[0], i[-1]) - min(i)) else "valley"
    net = i[-1] - i[0]
    if net > ARC_FLAT_BAND:
        return "rising"
    if net < -ARC_FLAT_BAND:
        return "falling"
    return "high" if sum(i) / len(i) >= 0.5 else "low"


# ---- assembly ----

# Group the source gold by target scene -> {scene_id: {variant: entry}} (keeps `specific` + `ambiguous`).
def _scenes(source: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for e in source:
        out.setdefault(e["target_scene_id"], {})[e["variant"]] = e
    return out


# One writer-realistic entry: writer inputs only (summary, descriptors, pov, tense, [arc]) + the target + refs.
def _entry(variant: str, summary: str, descriptors: list, ref: dict, arc: str | None) -> dict:
    e = {
        "id": f"w-{ref['target_scene_id']}-{variant}",
        "book_id": ref["book_id"],
        "book_title": ref["book_title"],
        "sharpness": ref["sharpness"],
        "variant": variant,
        "summary": summary,
        "descriptors": descriptors,
        "pov": ref["target_pov"],
        "tense": ref["target_tense"],
        "target_scene_id": ref["target_scene_id"],
        "target_scene_summary": ref["target_scene_summary"],
    }
    if arc is not None:
        e["arc"] = arc                                                          # sparse omits the arc
    return e


# ** ENTRY ** — build the 3-variant writer gold over every source scene and write writer_queries.json.
def main():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))["test_queries"]
    scenes = _scenes(source)

    missing = [sid for sid in scenes if sid not in SPARSE_QUERIES]
    extra = [sid for sid in SPARSE_QUERIES if sid not in scenes]
    if missing or extra:
        raise SystemExit(f"SPARSE_QUERIES mismatch — missing {missing}, extra {extra}")

    entries = []
    for sid, by_var in scenes.items():
        spec = by_var.get("specific") or next(iter(by_var.values()))
        ambig = by_var.get("ambiguous") or spec
        arc = classify_arc(spec["target_vdi_curve"])
        descriptors = spec.get("descriptors") or []
        entries.append(_entry("similar", spec["summary"], descriptors, spec, arc))
        entries.append(_entry("ambiguous", ambig["summary"], ambig.get("descriptors") or descriptors, spec, arc))
        entries.append(_entry("sparse", SPARSE_QUERIES[sid], descriptors, spec, None))

    entries.sort(key=lambda e: (e["sharpness"], e["target_scene_id"],
                                {"similar": 0, "ambiguous": 1, "sparse": 2}[e["variant"]]))
    doc = {
        "_about": ("Writer-realistic gold: what a writer actually inputs — summary + descriptors + pov/tense, "
                   "plus (similar/ambiguous) an intensity ARC classified from the target scene's real vdi_curve. "
                   "NO facets (subject/verb/object/svos). 3 variants per scene: similar (detailed), ambiguous "
                   "(general language, same plot), sparse (general language, 1-2 major beats, no arc). "
                   f"{len(scenes)} scenes x 3 = {len(entries)} queries. Target = target_scene_id."),
        "test_queries": entries,
    }
    OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}  ({len(entries)} queries over {len(scenes)} scenes)")


if __name__ == "__main__":
    main()
