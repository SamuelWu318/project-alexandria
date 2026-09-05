from __future__ import annotations
import json
from pathlib import Path

# ---- retrieval eval + A/B comparator + weight tuner for the read path ----
# Answers ONE question: does approach A or B retrieve better on the gold set? Ground truth is
# BOOK-LEVEL (a result is a HIT when its book_id == the query's target), so metrics are book-match
# MRR / Hit@k / P@k, broken down by query sharpness. The scorer grades OUTPUTS (a Run) without
# re-running search; run_search drives search() to produce a Run. Gold queries carry a summary +
# subject/verb/object/setting frame + descriptors (no moment sentences), so they exercise the
# summary/frame vector channels + the descriptor flavor channel. `--tune` coordinate-ascends BOTH
# weight sets — field_weights (the per-channel vector blend) + method_weights (scenes:flavor RRF) —
# by caching each channel once, then re-blending for free. Run: `python -m evals` (see --mode/--tune).

# Run  = dict[query_id, list[{"scene_id": str, "book_id": str, "score": float}]]  best-first
# Gold = dict[query_id, {"book_id": str, "sharpness": int | None}]

GOLD_PATH = Path(__file__).resolve().parent / "webtest" / "gold" / "test_queries.json"
DEFAULT_KS = (1, 3, 5, 10)


# ---- gold loading ----

# Load the gold query set -> (raw query entries, gold judgments {qid: {book_id, sharpness}}).
def load_gold(path: Path | str | None = None) -> tuple[list[dict], dict]:
    data = json.loads(Path(path or GOLD_PATH).read_text(encoding="utf-8"))
    queries = data["test_queries"]
    gold = {e["id"]: {"book_id": e["book_id"], "sharpness": e.get("sharpness")}
            for e in queries}
    return queries, gold


# ---- scoring (operates on OUTPUTS, no search needed) ----

# The target book_id for a gold entry (accepts a bare id or a {book_id,...} dict).
def _target(g) -> str:
    return g["book_id"] if isinstance(g, dict) else g


# ** MAIN ** — compare_runs + coordinate_ascent grade every Run through here
# Grade one Run against the gold on book-match relevance (result relevant iff book_id == target). Returns aggregate MRR / Hit@k / P@k + per-query breakdown.
def score_run(run: dict, gold: dict, ks: tuple = DEFAULT_KS) -> dict:
    ks = tuple(sorted(ks))
    per: dict[str, dict] = {}
    for qid, g in gold.items():
        tb = _target(g)
        results = run.get(qid, [])
        rel = [1 if r.get("book_id") == tb else 0 for r in results]
        first = next((i + 1 for i, x in enumerate(rel) if x), 0)   # 1-based rank; 0 = miss
        per[qid] = {
            "rr": (1.0 / first) if first else 0.0,
            "first_rank": first,
            "hit": {k: (1.0 if any(rel[:k]) else 0.0) for k in ks},
            "prec": {k: (sum(rel[:k]) / k if results else 0.0) for k in ks},
            "sharpness": g.get("sharpness") if isinstance(g, dict) else None,
        }
    n = len(per) or 1
    agg = {
        "n": len(per),
        "mrr": sum(p["rr"] for p in per.values()) / n,
        "hit": {k: sum(p["hit"][k] for p in per.values()) / n for k in ks},
        "prec": {k: sum(p["prec"][k] for p in per.values()) / n for k in ks},
    }
    return {"aggregate": agg, "per_query": per, "ks": ks}


# Mean MRR + Hit@k grouped by query sharpness (1 sharp .. 5 generic).
def by_sharpness(scored: dict, k: int = 5) -> dict:
    groups: dict[int, list] = {}
    for p in scored["per_query"].values():
        groups.setdefault(p["sharpness"], []).append(p)
    out = {}
    for s in sorted(groups, key=lambda x: (x is None, x)):
        ps = groups[s]
        out[s] = {
            "n": len(ps),
            "mrr": sum(p["rr"] for p in ps) / len(ps),
            "hit": sum(p["hit"][k] for p in ps) / len(ps),
        }
    return out


# ---- comparison (A vs B) ----

# Score two Runs and diff them: aggregate deltas, per-query head-to-head (by reciprocal rank), and top-k id overlap (Jaccard).
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
        if ra["rr"] > rb["rr"]:
            wins_a += 1; winner = label_a
        elif rb["rr"] > ra["rr"]:
            wins_b += 1; winner = label_b
        else:
            ties += 1; winner = "tie"
        rows.append({"qid": qid, "sharpness": ra["sharpness"],
                     "a_rank": ra["first_rank"], "b_rank": rb["first_rank"],
                     "d_rr": rb["rr"] - ra["rr"], "winner": winner, "overlap": overlap})
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
    L.append(f"RETRIEVAL A/B  —  book-match accuracy on {aa['n']} gold queries")
    L.append(f"  A = {la}")
    L.append(f"  B = {lb}")
    L.append("=" * 64)

    # one metric row: A value, B value, signed delta.
    def row(name, va, vb):
        d = vb - va
        arrow = "  " if abs(d) < 1e-9 else (" +" if d > 0 else " -")
        return f"  {name:<10} {va:>{w}.3f}   {vb:>{w}.3f}   {arrow}{abs(d):.3f}"

    L.append(f"  {'metric':<10} {la:>{w}}   {lb:>{w}}   {'B-A':>7}")
    L.append("  " + "-" * 60)
    L.append(row("MRR", aa["mrr"], ba["mrr"]))
    for k in ks:
        L.append(row(f"Hit@{k}", aa["hit"][k], ba["hit"][k]))
    for k in ks:
        L.append(row(f"P@{k}", aa["prec"][k], ba["prec"][k]))

    h = cmp["head_to_head"]
    L.append("")
    L.append(f"  head-to-head (by first-hit rank): "
             f"{la} {h[la]}  |  {lb} {h[lb]}  |  tie {h['tie']}")

    # sharpness rollup
    sa = by_sharpness(cmp["a"]); sb = by_sharpness(cmp["b"])
    L.append("")
    L.append(f"  MRR by sharpness (1 unique .. 5 generic):")
    L.append(f"    {'sharp':<6} {'n':>3}   {la:>{w}}   {lb:>{w}}")
    for s in sorted(set(sa) | set(sb), key=lambda x: (x is None, x)):
        na = sa.get(s, {}).get("mrr", 0.0); nb = sb.get(s, {}).get("mrr", 0.0)
        cnt = sa.get(s, sb.get(s, {})).get("n", 0)
        L.append(f"    {str(s):<6} {cnt:>3}   {na:>{w}.3f}   {nb:>{w}.3f}")

    # biggest divergences: where the winner flipped, largest |ΔRR| first
    flips = [r for r in cmp["per_query"] if r["winner"] != "tie"]
    flips.sort(key=lambda r: abs(r["d_rr"]), reverse=True)
    L.append("")
    L.append(f"  biggest per-query swings (rank of first correct-book scene; 0 = miss):")
    L.append(f"    {'qid':<10} {'sh':>2}  {'rank_'+la[:5]:>10} {'rank_'+lb[:5]:>10}  "
             f"{'overlap':>7}  winner")
    for r in flips[:12]:
        L.append(f"    {r['qid']:<10} {str(r['sharpness']):>2}  "
                 f"{r['a_rank']:>10} {r['b_rank']:>10}  {r['overlap']:>7.2f}  {r['winner']}")
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
        moments = e.get("moments") if use_moments else None
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


# ---- fine-tuning: cache every channel once, then blend offline under any weights (coordinate ascent) ----
# The read path has TWO weight sets: field_weights (the per-channel vector blend inside search_scenes)
# and method_weights (the scenes:flavor RRF split). Both are tuned here. The expensive part — the vector
# queries — runs ONCE per gold query (collect_vector_channels caches each channel's z-normed scores + the
# flavor ranking); after that every candidate weight vector is a FREE re-blend (blend_run), so coordinate
# ascent can sweep all of it. `normalize`/`combine` are held fixed during a sweep (they shape the cache).

# Cache, per gold query: each active vector channel's normalized scores, the id->(scene_id,book_id) map, and the flavor ranking.
def collect_vector_channels(client, queries: list[dict], *, normalize: str | None = "zscore",
                            prefetch: int = 50) -> dict:
    import search
    out: dict[str, dict] = {}
    for e in queries:
        qid = e["id"]
        entry = {"channels": {}, "meta": {}, "flavor": []}
        if e.get("summary") or e.get("moments") or _gold_frame(e):
            try:
                scored = search.score_channels(client, summary=e.get("summary"), moments=e.get("moments"),
                                               frame=_gold_frame(e), normalize=normalize, prefetch=prefetch)
                entry["channels"] = scored["channels"]                     # {channel: {id: z}}
                entry["meta"] = {i: (p.payload.get("scene_id"), p.payload.get("book_id"))
                                 for i, p in scored["cand"].items()}       # id -> (scene_id, book_id)
            except Exception as ex:
                print(f"[evals] {qid} scenes: {type(ex).__name__}: {ex}")
        if e.get("descriptors"):
            try:
                pts = search.search_weighted_descriptors(client, e["descriptors"], limit=prefetch)   # flavor channel
                entry["flavor"] = [(p.payload.get("scene_id"), p.payload.get("book_id")) for p in pts]
            except Exception as ex:
                print(f"[evals] {qid} flavor: {type(ex).__name__}: {ex}")
        out[qid] = entry
    return out


# Weighted RRF over cached (scene_id, book_id) rankings -> ranked result dicts (mirrors search._rrf; all-zero weights -> equal).
def _rrf_pairs(rankings: dict, method_weights: dict, k: int, limit: int) -> list:
    active = [m for m in rankings if method_weights.get(m, 0.0) > 0] or list(rankings)
    total: dict = {}
    book: dict = {}
    for m in active:
        w = method_weights.get(m, 1.0)
        for rank, (sid, bid) in enumerate(rankings[m]):
            if not sid:
                continue
            total[sid] = total.get(sid, 0.0) + w / (k + rank + 1)
            book[sid] = bid
    order = sorted(total, key=lambda s: total[s], reverse=True)[:limit]
    return [{"scene_id": sid, "book_id": book[sid], "score": total[sid]} for sid in order]


# Re-blend the cached channels under (field_weights, combine) + RRF with flavor under method_weights -> a Run. Pure math, no search.
def blend_run(cached: dict, field_weights: dict | None, method_weights: dict, *,
              combine: str = "sum", k: int = 60, limit: int = 10) -> dict:
    import search
    run: dict[str, list] = {}
    for qid, e in cached.items():
        rankings: dict = {}
        if e["channels"]:
            scored = {"channels": e["channels"], "ids": list(e["meta"]), "cand": {}}
            fused = search.blend_channels(scored, field_weights, combine)   # [(id, score)] under these weights
            meta = e["meta"]
            rankings["scenes"] = [meta[i] for i, _ in fused]
        if e["flavor"]:
            rankings["flavor"] = e["flavor"]
        run[qid] = _rrf_pairs(rankings, method_weights, k, limit)
    return run


# Pull one scalar objective out of a score_run aggregate (mrr | hit@K | p@K).
def _metric_value(agg: dict, metric: str) -> float:
    if metric == "mrr":
        return agg["mrr"]
    if metric.startswith("hit@"):
        return agg["hit"][int(metric[4:])]
    if metric.startswith("p@"):
        return agg["prec"][int(metric[2:])]
    raise ValueError(f"unknown metric {metric!r} (use mrr, hit@K, or p@K)")


# Coordinate-ascent tune of the vector field_weights + the scenes:flavor split to maximize `metric` on gold (each blend is free). Returns (best_field, best_method, best_score, history).
def coordinate_ascent(cached: dict, gold: dict, *, metric: str = "mrr", combine: str = "sum",
                      k: int = 60, limit: int = 10, rounds: int = 4,
                      grid: tuple = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0)) -> tuple:
    import search
    seen: list = []                                            # active vector channels across the cache
    for e in cached.values():
        for c in e["channels"]:
            if c not in seen:
                seen.append(c)
    chans = [c for c in search.SCENES_VECTORS if c in seen]    # stable canonical order
    has_flavor = any(e["flavor"] for e in cached.values())
    field = {c: float(search.SCENES_DEFAULT_WEIGHTS.get(c, 0.0)) for c in chans}
    method = dict(search.DEFAULT_METHOD_WEIGHTS)

    def obj(fw, mw):
        run = blend_run(cached, fw, mw, combine=combine, k=k, limit=limit)
        return _metric_value(score_run(run, gold)["aggregate"], metric)

    best = obj(field, method)
    history = [("init", dict(field), dict(method), best)]
    for r in range(rounds):
        improved = False
        for ch in chans:                                       # optimize each channel weight in turn
            best_v = field[ch]
            for v in grid:
                field[ch] = v
                s = obj(field, method)
                if s > best + 1e-9:
                    best, best_v, improved = s, v, True
            field[ch] = best_v
        if has_flavor:                                         # then the scenes:flavor split
            best_s = method["scenes"]
            for sf in grid:
                cand = {"scenes": round(sf, 3), "flavor": round(1.0 - sf, 3)}
                s = obj(field, cand)
                if s > best + 1e-9:
                    best, best_s, improved = s, sf, True
            method = {"scenes": round(best_s, 3), "flavor": round(1.0 - best_s, 3)}
        history.append((f"round{r + 1}", dict(field), dict(method), best))
        if not improved:                                       # converged
            break
    return field, method, best, history


# Render the coordinate-ascent result: the winning field_weights + method split + per-round score.
def format_tuning(field: dict, method: dict, best_score: float, history: list, metric: str) -> str:
    L = ["", f"COORDINATE-ASCENT TUNE  (objective {metric.upper()})",
         f"  best {metric} = {best_score:.4f}",
         "  field_weights (vector blend, relative):"]
    for c, v in field.items():
        L.append(f"    {c:<10} {v:>6.3f}")
    L.append(f"  method_weights = {{'scenes': {method['scenes']}, 'flavor': {method['flavor']}}}")
    L.append("  per-round best:")
    for tag, _, _, sc in history:
        L.append(f"    {tag:<8} {sc:>8.4f}")
    return "\n".join(L)


# ---- CLI ----

# ** LOCKED **
# CLI string -> normalize value ('none'/'raw'/'null'/'' -> None).
def _norm_arg(s: str) -> str | None:
    return None if s.lower() in ("none", "raw", "null", "") else s.lower()


# CLI entry: parse args, run the chosen A/B mode or the coordinate-ascent tune, print the report.
def main():
    import argparse
    import search
    ap = argparse.ArgumentParser(
        description="A/B two read-path configs on the gold set, or coordinate-ascent tune every weight.")
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
                    help="coordinate-ascent tune the field_weights + scenes:flavor split for best --metric")
    ap.add_argument("--metric", default="mrr", help="tune objective: mrr | hit@K | p@K")
    args = ap.parse_args()

    queries, gold = load_gold(args.gold)                       # gold entries + judgments

    if args.tune:
        nrm = _norm_arg(args.normalize)
        client = search.open_client()
        try:
            cached = collect_vector_channels(client, queries, normalize=nrm)   # one expensive pass
        finally:
            client.close()
        field, method, best_score, history = coordinate_ascent(               # free re-blends
            cached, gold, metric=args.metric, combine=args.combine, limit=args.limit)
        base_run = blend_run(cached, None, search.DEFAULT_METHOD_WEIGHTS,      # default weights
                             combine=args.combine, limit=args.limit)
        tuned_run = blend_run(cached, field, method, combine=args.combine, limit=args.limit)   # tuned weights
        cmp = compare_runs(base_run, tuned_run, gold, label_a="default", label_b="tuned")
        print(format_comparison(cmp))
        print(format_tuning(field, method, best_score, history, args.metric))
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
