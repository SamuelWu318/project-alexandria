from __future__ import annotations
import json
from pathlib import Path

# ---- retrieval eval + A/B comparator + weight tuner for the read path ----
# Answers ONE question: does approach A or B retrieve better on the gold set? The EMPHASIS is rank-1 —
# is the 1st result the correct BOOK and the correct SCENE? Ground truth is two-level: book_id (always)
# and target_scene_id (auto-labeled by `--label-scenes`: the rank-1 in-book scene under default weights).
# Headline metrics are top_scene / top_book (rank-1 accuracy) + the top1 composite (1.0 exact scene,
# 0.5 right book, 0.0 miss); MRR / Hit@k / scene-MRR are secondary. The scorer grades OUTPUTS (a Run)
# without re-running search; run_search drives search() to produce a Run. Gold queries carry a summary +
# subject/verb/object/setting frame + a combined `svos` clause (synthesized from that frame) +
# descriptors, so they exercise the summary/svos/frame vector channels + the descriptor flavor channel.
# `--tune` assumes the real query is a sentence summary + an S/V/O/S frame + its combined svos (what a
# user actually types), so it coordinate-ascends the field_weights over summary + svos + subject/verb/
# object/setting — descriptors/flavor stay out of the tune — then normalizes the winner to sum 1. Caches
# each channel once, re-blends for free. Run: `python -m evals` (see --mode/--tune).

# Run  = dict[query_id, list[{"scene_id": str, "book_id": str, "score": float}]]  best-first
# Gold = dict[query_id, {"book_id": str, "scene_id": str | None, "sharpness": int | None}]  scene_id = target scene

GOLD_PATH = Path(__file__).resolve().parent / "webtest" / "gold" / "test_queries.json"
DEFAULT_KS = (1, 3, 5, 10)


# ---- gold loading ----

# Load the gold query set -> (raw query entries, gold judgments {qid: {book_id, sharpness}}).
def load_gold(path: Path | str | None = None) -> tuple[list[dict], dict]:
    data = json.loads(Path(path or GOLD_PATH).read_text(encoding="utf-8"))
    queries = data["test_queries"]
    gold = {e["id"]: {"book_id": e["book_id"], "scene_id": e.get("target_scene_id"),
                      "sharpness": e.get("sharpness")}
            for e in queries}
    return queries, gold


# ---- scoring (operates on OUTPUTS, no search needed) ----

# The target book_id for a gold entry (accepts a bare id or a {book_id,...} dict).
def _target(g) -> str:
    return g["book_id"] if isinstance(g, dict) else g


# The target scene_id for a gold entry (the auto-labeled correct scene); None if unlabeled.
def _target_scene(g) -> str | None:
    return g.get("scene_id") if isinstance(g, dict) else None


# ** MAIN ** — compare_runs + coordinate_ascent grade every Run through here
# Grade one Run against the gold with HEAVY rank-1 emphasis: the 1st result must be the correct BOOK and the correct SCENE. Tracks book-match + scene-match at every rank (MRR / Hit@k) but headlines top_book (rank-1 right book), top_scene (rank-1 exact scene), and top1 (composite: 1.0 exact scene, 0.5 right book, 0.0 wrong). Returns aggregate + per-query breakdown.
def score_run(run: dict, gold: dict, ks: tuple = DEFAULT_KS) -> dict:
    ks = tuple(sorted(ks))
    per: dict[str, dict] = {}
    for qid, g in gold.items():
        tb = _target(g)
        ts = _target_scene(g)
        results = run.get(qid, [])
        brel = [1 if r.get("book_id") == tb else 0 for r in results]                 # book-match at each rank
        srel = [1 if (ts and r.get("scene_id") == ts) else 0 for r in results]       # exact scene-match at each rank
        bfirst = next((i + 1 for i, x in enumerate(brel) if x), 0)   # 1-based rank of first correct book; 0 = miss
        sfirst = next((i + 1 for i, x in enumerate(srel) if x), 0)   # 1-based rank of the exact scene; 0 = miss
        top_book = float(brel[0]) if brel else 0.0                  # THE headline: is result #1 the right book?
        top_scene = float(srel[0]) if srel else 0.0                 # THE headline: is result #1 the exact scene?
        per[qid] = {
            "rr": (1.0 / bfirst) if bfirst else 0.0,
            "first_rank": bfirst,
            "hit": {k: (1.0 if any(brel[:k]) else 0.0) for k in ks},
            "prec": {k: (sum(brel[:k]) / k if results else 0.0) for k in ks},
            "scene_rr": (1.0 / sfirst) if sfirst else 0.0,
            "scene_first_rank": sfirst,
            "scene_hit": {k: (1.0 if any(srel[:k]) else 0.0) for k in ks},
            "top_book": top_book,
            "top_scene": top_scene,
            "top1": 1.0 if top_scene else (0.5 if top_book else 0.0),   # rank-1 composite (scene >> book >> miss)
            "has_scene": 1.0 if ts else 0.0,
            "sharpness": g.get("sharpness") if isinstance(g, dict) else None,
        }
    n = len(per) or 1
    ns = sum(p["has_scene"] for p in per.values()) or 1.0           # scene metrics average only over scene-labeled queries
    agg = {
        "n": len(per),
        "n_scene": int(sum(p["has_scene"] for p in per.values())),
        "mrr": sum(p["rr"] for p in per.values()) / n,
        "hit": {k: sum(p["hit"][k] for p in per.values()) / n for k in ks},
        "prec": {k: sum(p["prec"][k] for p in per.values()) / n for k in ks},
        "scene_mrr": sum(p["scene_rr"] for p in per.values()) / ns,
        "scene_hit": {k: sum(p["scene_hit"][k] for p in per.values()) / ns for k in ks},
        "top_book": sum(p["top_book"] for p in per.values()) / n,   # rank-1 correct-book accuracy
        "top_scene": sum(p["top_scene"] for p in per.values()) / ns,  # rank-1 exact-scene accuracy
        "top1": sum(p["top1"] for p in per.values()) / n,           # rank-1 composite objective
    }
    return {"aggregate": agg, "per_query": per, "ks": ks}


# Rank-1 accuracy (top1 composite, top_scene, top_book) + MRR grouped by query sharpness (1 sharp .. 5 generic).
def by_sharpness(scored: dict, k: int = 5) -> dict:
    groups: dict[int, list] = {}
    for p in scored["per_query"].values():
        groups.setdefault(p["sharpness"], []).append(p)
    out = {}
    for s in sorted(groups, key=lambda x: (x is None, x)):
        ps = groups[s]
        nsc = sum(p["has_scene"] for p in ps) or 1.0
        out[s] = {
            "n": len(ps),
            "mrr": sum(p["rr"] for p in ps) / len(ps),
            "hit": sum(p["hit"][k] for p in ps) / len(ps),
            "top1": sum(p["top1"] for p in ps) / len(ps),
            "top_scene": sum(p["top_scene"] for p in ps) / nsc,
            "top_book": sum(p["top_book"] for p in ps) / len(ps),
        }
    return out


# ---- comparison (A vs B) ----

# Score two Runs and diff them: aggregate deltas, per-query head-to-head (by RANK-1 composite top1, reciprocal-rank tiebreak), and top-k id overlap (Jaccard).
def compare_runs(run_a: dict, run_b: dict, gold: dict, *,
                 label_a: str = "A", label_b: str = "B",
                 ks: tuple = DEFAULT_KS, k_overlap: int = 5) -> dict:
    a = score_run(run_a, gold, ks)                             # grade A
    b = score_run(run_b, gold, ks)                             # grade B
    wins_a = wins_b = ties = 0
    rows = []
    for qid in gold:
        ra, rb = a["per_query"][qid], b["per_query"][qid]
        sa = {r.get("scene_id") for r in run_a.get(qid, [])[:k_overlap]}
        sb = {r.get("scene_id") for r in run_b.get(qid, [])[:k_overlap]}
        union = len(sa | sb) or 1
        overlap = len(sa & sb) / union
        ka = (ra["top1"], ra["rr"])                            # rank-1 first, book RR as tiebreak
        kb = (rb["top1"], rb["rr"])
        if ka > kb:
            wins_a += 1; winner = label_a
        elif kb > ka:
            wins_b += 1; winner = label_b
        else:
            ties += 1; winner = "tie"
        rows.append({"qid": qid, "sharpness": ra["sharpness"],
                     "a_rank": ra["first_rank"], "b_rank": rb["first_rank"],
                     "a_srank": ra["scene_first_rank"], "b_srank": rb["scene_first_rank"],
                     "d_top1": rb["top1"] - ra["top1"], "d_rr": rb["rr"] - ra["rr"],
                     "winner": winner, "overlap": overlap})
    return {
        "labels": (label_a, label_b),
        "a": a, "b": b,
        "head_to_head": {label_a: wins_a, label_b: wins_b, "tie": ties},
        "per_query": rows, "ks": tuple(sorted(ks)), "k_overlap": k_overlap,
    }


# ---- text report ----

# Render compare_runs() output as a plain-text report (no deps).
def format_comparison(cmp: dict) -> str:
    la, lb = cmp["labels"]
    aa, ba = cmp["a"]["aggregate"], cmp["b"]["aggregate"]
    ks = cmp["ks"]
    w = max(len(la), len(lb), 7)
    L = []
    L.append("=" * 64)
    L.append(f"RETRIEVAL A/B  —  RANK-1 correctness (book + scene) on {aa['n']} gold queries"
             f"  ({aa['n_scene']} scene-labeled)")
    L.append(f"  A = {la}")
    L.append(f"  B = {lb}")
    L.append("=" * 64)

    # one metric row: A value, B value, signed delta.
    def row(name, va, vb):
        d = vb - va
        arrow = "  " if abs(d) < 1e-9 else (" +" if d > 0 else " -")
        return f"  {name:<12} {va:>{w}.3f}   {vb:>{w}.3f}   {arrow}{abs(d):.3f}"

    L.append(f"  {'metric':<12} {la:>{w}}   {lb:>{w}}   {'B-A':>7}")
    L.append("  " + "-" * 60)
    L.append("  >> RANK-1 (the 1st result) — THE emphasis <<")
    L.append(row("top1 scene", aa["top_scene"], ba["top_scene"]))    # #1 result IS the exact correct scene
    L.append(row("top1 book", aa["top_book"], ba["top_book"]))       # #1 result IS the correct book
    L.append(row("top1 comp", aa["top1"], ba["top1"]))               # composite: 1.0 scene / 0.5 book / 0 miss
    L.append("  -- ranking quality (secondary) --")
    L.append(row("scene MRR", aa["scene_mrr"], ba["scene_mrr"]))
    L.append(row("book MRR", aa["mrr"], ba["mrr"]))
    for k in ks:
        L.append(row(f"scene Hit@{k}", aa["scene_hit"][k], ba["scene_hit"][k]))
    for k in ks:
        L.append(row(f"book Hit@{k}", aa["hit"][k], ba["hit"][k]))

    h = cmp["head_to_head"]
    L.append("")
    L.append(f"  head-to-head (by rank-1 top1, RR tiebreak): "
             f"{la} {h[la]}  |  {lb} {h[lb]}  |  tie {h['tie']}")

    # sharpness rollup — rank-1 composite (the emphasis) by query sharpness
    sa = by_sharpness(cmp["a"]); sb = by_sharpness(cmp["b"])
    L.append("")
    L.append(f"  top1 (rank-1 composite) by sharpness (1 unique .. 5 generic):")
    L.append(f"    {'sharp':<6} {'n':>3}   {la:>{w}}   {lb:>{w}}")
    for s in sorted(set(sa) | set(sb), key=lambda x: (x is None, x)):
        na = sa.get(s, {}).get("top1", 0.0); nb = sb.get(s, {}).get("top1", 0.0)
        cnt = sa.get(s, sb.get(s, {})).get("n", 0)
        L.append(f"    {str(s):<6} {cnt:>3}   {na:>{w}.3f}   {nb:>{w}.3f}")

    # biggest rank-1 swings: where the winner flipped, largest |Δtop1| first (then |ΔRR|)
    flips = [r for r in cmp["per_query"] if r["winner"] != "tie"]
    flips.sort(key=lambda r: (abs(r["d_top1"]), abs(r["d_rr"])), reverse=True)
    L.append("")
    L.append(f"  biggest rank-1 swings (book_rank / scene_rank of 1st correct; 0 = miss):")
    L.append(f"    {'qid':<10} {'sh':>2}  {'bk_'+la[:4]+'/'+lb[:4]:>11} {'sc_'+la[:4]+'/'+lb[:4]:>11}  winner")
    for r in flips[:12]:
        L.append(f"    {r['qid']:<10} {str(r['sharpness']):>2}  "
                 f"{str(r['a_rank'])+'/'+str(r['b_rank']):>11} "
                 f"{str(r['a_srank'])+'/'+str(r['b_srank']):>11}  {r['winner']}")
    L.append("=" * 64)
    return "\n".join(L)


# ---- driver: produce a Run from the unified search() ----

# ** MAIN ** — every A/B mode + the default/tuned configs run through here
# Drive the unified search() over the gold queries with one FULL config -> a Run. Flags gate which channels run (isolate one for an A/B); field_weights/method_weights/combine/normalize set the blend.
def run_search(client, queries: list[dict], *, use_summary: bool = True, use_moments: bool = True,
               use_frame: bool = True, use_descriptors: bool = False,
               field_weights: dict | None = None, method_weights: dict | None = None,
               combine: str = "sum", normalize: str | None = "zscore", limit: int = 10) -> dict:
    import search
    run: dict[str, list] = {}
    for e in queries:
        summary = e.get("summary") if use_summary else None
        moments = _gold_moments(e) if use_moments else None                # combined svos clause -> svos channel
        frame = _gold_frame(e) if use_frame else None                      # subject/verb/object/setting query terms
        descriptors = (e.get("descriptors") or None) if use_descriptors else None
        try:
            pts = search.search(client, summary=summary, moments=moments, frame=frame,   # unified search
                                descriptors=descriptors, field_weights=field_weights,
                                method_weights=method_weights, combine=combine,
                                normalize=normalize, limit=limit)
        except Exception as ex:                 # an empty/invalid query shouldn't sink the run
            print(f"[evals] {e['id']}: {type(ex).__name__}: {ex}")
            run[e["id"]] = []
            continue
        run[e["id"]] = [{"scene_id": p.payload.get("scene_id"),
                         "book_id": p.payload.get("book_id"),
                         "score": p.score} for p in pts]
    return run


# The frame query dict for a gold entry: its subject/verb/object/setting fields (empty facets dropped); None if all empty.
def _gold_frame(e: dict) -> dict | None:
    frame = {f: e.get(f) for f in ("subject", "verb", "object", "setting") if e.get(f)}
    return frame or None


# The moment/svos query for a gold entry: its combined `svos` clause (a bare sentence -> the svos multivector channel); None if absent.
def _gold_moments(e: dict):
    return e.get("moments") or e.get("svos") or None


# ---- scene-target auto-labeling (fills gold's target_scene_id so rank-1 can be graded at the SCENE level) ----

# Auto-label each query's correct scene: the rank-1 scene WITHIN its own book under a FIXED default-weight what-happens query (summary + svos + frame). Held at default weights on purpose, so the label is independent of the field_weights the tune later sweeps (not self-fulfilling). Mutates queries in place, writing target_scene_id/title/summary (None if unresolved); returns (n_labeled, n_missing). Review the written titles before trusting scene metrics.
def autolabel_scenes(client, queries: list[dict], *, normalize: str | None = "zscore") -> tuple:
    import search
    labeled = missing = 0
    for e in queries:
        e["target_scene_id"] = None
        e["target_scene_title"] = None
        e["target_scene_summary"] = None
        try:
            pts = search.search(client, summary=e.get("summary"), moments=_gold_moments(e),   # default field_weights
                                frame=_gold_frame(e), book_id=e["book_id"],
                                normalize=normalize, limit=1)
        except Exception as ex:
            print(f"[evals] {e['id']} label: {type(ex).__name__}: {ex}")
            missing += 1
            continue
        if pts:
            pl = pts[0].payload
            e["target_scene_id"] = pl.get("scene_id")
            e["target_scene_title"] = pl.get("scene_title")
            e["target_scene_summary"] = pl.get("summary")
            labeled += 1
        else:
            missing += 1
    return labeled, missing


# ---- fine-tuning: cache the summary+svos+frame channels once, then blend offline under any weights (coordinate ascent) ----
# The tune assumes a real query is a sentence summary + a combined svos clause + an S/V/O/S frame, so it
# tunes ONE weight set: field_weights (the per-channel vector blend inside search_scenes over summary +
# svos + subject/verb/object/setting). Descriptors/flavor are excluded. The expensive part — the vector
# queries — runs ONCE per gold query (collect_vector_channels caches each channel's z-normed scores);
# after that every candidate weight vector is a FREE re-blend (blend_run), so coordinate ascent can sweep
# the whole grid. `normalize`/`combine` are held fixed during a sweep (they shape the cache).

# Cache, per gold query: each summary+svos+frame vector channel's normalized scores + the id->(scene_id,book_id) map. Descriptors are not cached — only what a user types (summary + combined svos + S/V/O/S).
def collect_vector_channels(client, queries: list[dict], *, normalize: str | None = "zscore",
                            prefetch: int = 50) -> dict:
    import search
    out: dict[str, dict] = {}
    for e in queries:
        qid = e["id"]
        entry = {"channels": {}, "meta": {}}
        if e.get("summary") or _gold_moments(e) or _gold_frame(e):
            try:
                scored = search.score_channels(client, summary=e.get("summary"),     # summary + combined svos + S/V/O/S frame
                                               moments=_gold_moments(e), frame=_gold_frame(e),
                                               normalize=normalize, prefetch=prefetch)
                entry["channels"] = scored["channels"]                     # {channel: {id: z}}
                entry["meta"] = {i: (p.payload.get("scene_id"), p.payload.get("book_id"))
                                 for i, p in scored["cand"].items()}       # id -> (scene_id, book_id)
            except Exception as ex:
                print(f"[evals] {qid} scenes: {type(ex).__name__}: {ex}")
        out[qid] = entry
    return out


# Re-blend the cached summary+svos+frame channels under (field_weights, combine) -> a Run. Pure math, no search.
def blend_run(cached: dict, field_weights: dict | None, *,
              combine: str = "sum", limit: int = 10) -> dict:
    import search
    run: dict[str, list] = {}
    for qid, e in cached.items():
        if not e["channels"]:
            run[qid] = []
            continue
        scored = {"channels": e["channels"], "ids": list(e["meta"]), "cand": {}}
        fused = search.blend_channels(scored, field_weights, combine)      # [(id, score)] best-first under these weights
        meta = e["meta"]
        run[qid] = [{"scene_id": meta[i][0], "book_id": meta[i][1], "score": sc}
                    for i, sc in fused[:limit]]
    return run


# Pull one scalar objective out of a score_run aggregate. Rank-1 objectives (the emphasis): top1 (composite), scene@1, book@1. Also mrr, scene_mrr, hit@K, scene_hit@K, p@K.
def _metric_value(agg: dict, metric: str) -> float:
    flat = {
        "top1": agg["top1"],                # rank-1 composite (1.0 exact scene / 0.5 right book / 0)
        "scene@1": agg["top_scene"],        # rank-1 exact-scene accuracy
        "book@1": agg["top_book"],          # rank-1 correct-book accuracy
        "mrr": agg["mrr"],
        "scene_mrr": agg["scene_mrr"],
    }
    if metric in flat:
        return flat[metric]
    if metric.startswith("scene_hit@"):
        return agg["scene_hit"][int(metric.split("@")[1])]
    if metric.startswith("hit@"):
        return agg["hit"][int(metric[4:])]
    if metric.startswith("p@"):
        return agg["prec"][int(metric[2:])]
    raise ValueError(f"unknown metric {metric!r} (use top1, scene@1, book@1, mrr, scene_mrr, "
                     f"hit@K, scene_hit@K, or p@K)")


# ** LOCKED **
# Clamp weights non-negative and renormalize them to sum to 1 (all-zero -> uniform). Mirrors search._resolve_field_weights so the tuned vector is on the same simplex search uses.
def _normalize_weights(w: dict) -> dict:
    w = {k: max(0.0, float(v)) for k, v in w.items()}
    total = sum(w.values())
    if total <= 0:
        n = len(w) or 1
        return {k: 1.0 / n for k in w}
    return {k: v / total for k, v in w.items()}


# Coordinate-ascent tune of the summary+svos+frame field_weights to maximize `metric` on gold (each blend is free). Returns (best_field, best_score, history) with best_field NORMALIZED to sum 1.
def coordinate_ascent(cached: dict, gold: dict, *, metric: str = "mrr", combine: str = "sum",
                      limit: int = 10, rounds: int = 4,
                      grid: tuple = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1)) -> tuple:
    import search
    seen: list = []                                            # active vector channels across the cache
    for e in cached.values():
        for c in e["channels"]:
            if c not in seen:
                seen.append(c)
    chans = [c for c in search.SCENES_VECTORS if c in seen]    # stable canonical order (summary + svos + S/V/O/S)
    field = {c: float(search.SCENES_DEFAULT_WEIGHTS.get(c, 0.0)) for c in chans}

    def obj(fw):
        run = blend_run(cached, fw, combine=combine, limit=limit)
        return _metric_value(score_run(run, gold)["aggregate"], metric)

    best = obj(field)
    history = [("init", dict(field), best)]
    for r in range(rounds):
        improved = False
        for ch in chans:                                       # optimize each channel weight in turn
            best_v = field[ch]
            for v in grid:
                field[ch] = v
                s = obj(field)
                if s > best + 1e-9:
                    best, best_v, improved = s, v, True
            field[ch] = best_v
        history.append((f"round{r + 1}", dict(field), best))
        if not improved:                                       # converged
            break
    return _normalize_weights(field), best, history            # normalize the winning blend to sum 1 (scale-invariant, so score unchanged)


# Render the coordinate-ascent result: the winning (normalized) field_weights + per-round score.
def format_tuning(field: dict, best_score: float, history: list, metric: str) -> str:
    L = ["", f"COORDINATE-ASCENT TUNE  (objective {metric.upper()})",
         f"  best {metric} = {best_score:.4f}",
         "  field_weights (summary + svos + S/V/O/S vector blend, normalized to sum 1):"]
    for c, v in field.items():
        L.append(f"    {c:<10} {v:>6.3f}")
    L.append(f"    {'sum':<10} {sum(field.values()):>6.3f}")
    L.append("  per-round best:")
    for tag, _, sc in history:
        L.append(f"    {tag:<8} {sc:>8.4f}")
    return "\n".join(L)


# ---- CLI ----

# ** LOCKED **
# CLI string -> normalize value ('none'/'raw'/'null'/'' -> None).
def _norm_arg(s: str) -> str | None:
    return None if s.lower() in ("none", "raw", "null", "") else s.lower()


# CLI helper for --label-scenes: auto-label target scenes, rewrite the gold json in place (backs up to .bak), print a review table.
def _label_scenes_cli(gold_path: str | None, normalize: str | None) -> None:
    import search
    import shutil
    path = Path(gold_path or GOLD_PATH)
    data = json.loads(path.read_text(encoding="utf-8"))
    queries = data["test_queries"]
    client = search.open_client()
    try:
        labeled, missing = autolabel_scenes(client, queries, normalize=normalize)   # writes target_scene_* per query
    finally:
        client.close()
    order = ["id", "book_id", "book_title", "sharpness", "summary", "subject", "verb", "object",
             "setting", "svos", "descriptors", "target_scene_id", "target_scene_title",
             "target_scene_summary"]                                                 # review fields sit by the frame
    data["test_queries"] = [{k: e[k] for k in order if k in e} |
                            {k: v for k, v in e.items() if k not in order} for e in queries]
    shutil.copy(path, str(path) + ".bak")                                            # backup before overwrite
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"labeled {labeled} / {len(queries)} queries ({missing} unresolved). backup: {path.name}.bak\n")
    print(f"  {'qid':<10} {'book':>6}  {'scene_id':>10}  title  [REVIEW THESE]")
    for e in queries:
        print(f"  {e['id']:<10} {e['book_id']:>6}  {str(e.get('target_scene_id')):>10}  "
              f"{e.get('target_scene_title')}")


# CLI entry: parse args, run the chosen A/B mode or the coordinate-ascent tune, print the report.
def main():
    import argparse
    import search
    ap = argparse.ArgumentParser(
        description="A/B two read-path configs on the gold set, or coordinate-ascent tune the field_weights.")
    ap.add_argument("--mode", default="norm", choices=("norm", "lift", "flavor", "combine"),
                    help="norm: A/B normalize (--a vs --b). lift: summary-only vs summary+frame. "
                         "flavor: what-happens+frame vs + descriptors. combine: sum-blend vs max.")
    ap.add_argument("--a", default="none", help="normalize for A (mode=norm): none|zscore|minmax")
    ap.add_argument("--b", default="zscore", help="normalize for B (mode=norm): none|zscore|minmax")
    ap.add_argument("--normalize", default="zscore",
                    help="normalize held fixed for modes lift/flavor/combine + tune: none|zscore|minmax")
    ap.add_argument("--combine", default="sum", choices=("sum", "max"),
                    help="vector-blend combine held fixed for lift/flavor/tune (sum=weighted blend, max=greatest single)")
    ap.add_argument("--limit", type=int, default=10, help="results retrieved per query")
    ap.add_argument("--gold", default=None, help="path to a gold query json (default: webtest gold)")
    ap.add_argument("--tune", action="store_true",
                    help="coordinate-ascent tune the summary + svos + S/V/O/S field_weights for best --metric")
    ap.add_argument("--metric", default="top1",
                    help="tune objective (rank-1 emphasis): top1 | scene@1 | book@1 | mrr | scene_mrr | hit@K | scene_hit@K | p@K")
    ap.add_argument("--label-scenes", action="store_true",
                    help="auto-label each gold query's target_scene_id (rank-1 in-book, default weights) and write it back")
    args = ap.parse_args()

    queries, gold = load_gold(args.gold)                       # gold entries + judgments

    if args.label_scenes:
        _label_scenes_cli(args.gold, _norm_arg(args.normalize))
        return

    if args.tune:
        nrm = _norm_arg(args.normalize)
        client = search.open_client()
        try:
            cached = collect_vector_channels(client, queries, normalize=nrm)   # one expensive pass
        finally:
            client.close()
        field, best_score, history = coordinate_ascent(                       # free re-blends
            cached, gold, metric=args.metric, combine=args.combine, limit=args.limit)
        base_run = blend_run(cached, None, combine=args.combine, limit=args.limit)     # default weights
        tuned_run = blend_run(cached, field, combine=args.combine, limit=args.limit)   # tuned weights
        cmp = compare_runs(base_run, tuned_run, gold, label_a="default", label_b="tuned")
        print(format_comparison(cmp))
        print(format_tuning(field, best_score, history, args.metric))
        return

    client = search.open_client()
    try:
        nrm = _norm_arg(args.normalize)
        if args.mode == "lift":
            run_a = run_search(client, queries, use_frame=False, normalize=nrm, combine=args.combine, limit=args.limit)  # summary only
            run_b = run_search(client, queries, use_frame=True, normalize=nrm, combine=args.combine, limit=args.limit)   # + frame
            la, lb = "summary_only", "summary+frame"
        elif args.mode == "flavor":
            run_a = run_search(client, queries, use_descriptors=False, normalize=nrm, combine=args.combine, limit=args.limit)  # what-happens
            run_b = run_search(client, queries, use_descriptors=True, normalize=nrm, combine=args.combine, limit=args.limit)   # + descriptors
            la, lb = "what_happens", "+descriptors"
        elif args.mode == "combine":
            run_a = run_search(client, queries, normalize=nrm, combine="sum", limit=args.limit)   # weighted blend
            run_b = run_search(client, queries, normalize=nrm, combine="max", limit=args.limit)   # greatest single
            la, lb = "combine=sum", "combine=max"
        else:  # norm
            run_a = run_search(client, queries, normalize=_norm_arg(args.a), combine=args.combine, limit=args.limit)
            run_b = run_search(client, queries, normalize=_norm_arg(args.b), combine=args.combine, limit=args.limit)
            la, lb = f"norm={args.a}", f"norm={args.b}"
    finally:
        client.close()

    cmp = compare_runs(run_a, run_b, gold, label_a=la, label_b=lb)   # score + diff A vs B
    print(format_comparison(cmp))


if __name__ == "__main__":
    main()
