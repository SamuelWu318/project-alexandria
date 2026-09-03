from __future__ import annotations
import json
from pathlib import Path

# ---- retrieval eval + A/B comparator for the read path ----
# Answers ONE question: does approach A or B retrieve better on the gold set? Ground truth is
# BOOK-LEVEL (a result is a HIT when its book_id == the query's target), so metrics are book-match
# MRR / Hit@k / P@k, broken down by query sharpness. The scorer grades OUTPUTS (a Run) without
# re-running search; run_search is the thin driver that produces a Run from the unified search().
# The only search weight is search()'s method_weights (RRF); `--tune` sweeps it. Run: `python -m evals`.

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


# ** MAIN ** — compare_runs + tune_method_weights grade every Run through here
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

# Drive the unified search() over the gold queries with one channel/normalize config -> a Run (flags gate which channels run, so an A/B isolates one).
def run_search(client, queries: list[dict], *, use_summary: bool = True,
               use_moments: bool = True, use_descriptors: bool = False,
               normalize: str | None = "zscore", limit: int = 10) -> dict:
    import search
    run: dict[str, list] = {}
    for e in queries:
        summary = e.get("summary") if use_summary else None
        moments = e.get("moments") if use_moments else None
        descriptors = (e.get("descriptors") or None) if use_descriptors else None
        try:
            pts = search.search(client, summary=summary, moments=moments,          # unified search
                                descriptors=descriptors, normalize=normalize, limit=limit)
        except Exception as ex:                 # an empty/invalid query shouldn't sink the run
            print(f"[evals] {e['id']}: {type(ex).__name__}: {ex}")
            run[e["id"]] = []
            continue
        run[e["id"]] = [{"scene_id": p.payload.get("scene_id"),
                         "book_id": p.payload.get("book_id"),
                         "score": p.score} for p in pts]
    return run


# ---- weight tuning: sweep the scenes:flavor RRF ratio (the only search weight) ----
# Collect each gold query's TWO channel rankings ONCE (the expensive part), then re-RRF under any
# ratio for free and keep the best. `normalize` is held fixed (it lives inside the scenes channel's MAX).

# Run each gold query's what-happens + flavor channels SEPARATELY, once, and cache their ranked (scene_id, book_id) lists.
def collect_channels(client, queries: list[dict], *, normalize: str | None = "zscore",
                     limit: int = 10) -> dict:
    import search
    out: dict[str, dict] = {}
    for e in queries:
        qid = e["id"]
        scenes, flavor = [], []
        if e.get("summary") or e.get("moments"):
            try:
                pts = search.search_scenes(client, summary=e.get("summary"),           # what-happens channel
                                           moments=e.get("moments"), normalize=normalize, limit=limit)
                scenes = [(p.payload.get("scene_id"), p.payload.get("book_id")) for p in pts]
            except Exception as ex:
                print(f"[evals] {qid} scenes: {type(ex).__name__}: {ex}")
        if e.get("descriptors"):
            try:
                pts = search.search_weighted_descriptors(client, e["descriptors"], limit=limit)  # flavor channel
                flavor = [(p.payload.get("scene_id"), p.payload.get("book_id")) for p in pts]
            except Exception as ex:
                print(f"[evals] {qid} flavor: {type(ex).__name__}: {ex}")
        out[qid] = {"scenes": scenes, "flavor": flavor}
    return out


# Re-RRF the cached channel rankings under `method_weights` -> a Run (pure math, mirrors search._rrf).
def rrf_from_channels(channels: dict, method_weights: dict, *, k: int = 60, limit: int = 10) -> dict:
    run: dict[str, list] = {}
    for qid, ch in channels.items():
        total: dict = {}
        book: dict = {}
        for m, ranked in ch.items():
            w = method_weights.get(m, 0.0)
            if w <= 0:
                continue
            for rank, (sid, bid) in enumerate(ranked):
                if not sid:
                    continue
                total[sid] = total.get(sid, 0.0) + w / (k + rank + 1)
                book[sid] = bid
        order = sorted(total, key=lambda s: total[s], reverse=True)[:limit]
        run[qid] = [{"scene_id": sid, "book_id": book[sid], "score": total[sid]} for sid in order]
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


# Sweep the scenes:flavor RRF ratio over `grid` to maximize `metric` (ties keep the LOWER scenes weight). Returns (best_weights, best_score, history).
def tune_method_weights(channels: dict, gold: dict, *, metric: str = "mrr", k: int = 60,
                        limit: int = 10,
                        grid: tuple = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)) -> tuple:
    history, best_s, best_score = [], grid[0], -1.0
    for s in grid:
        mw = {"scenes": round(s, 3), "flavor": round(1.0 - s, 3)}
        run = rrf_from_channels(channels, mw, k=k, limit=limit)        # re-RRF at this ratio (free)
        sc = _metric_value(score_run(run, gold)["aggregate"], metric)  # grade it
        history.append((s, sc))
        if sc > best_score + 1e-9:
            best_score, best_s = sc, s
    best = {"scenes": round(best_s, 3), "flavor": round(1.0 - best_s, 3)}
    return best, best_score, history


# Render the scenes:flavor sweep table + the winning method_weights.
def format_tuning(history: list, best: dict, best_score: float, metric: str, n_flavor: int) -> str:
    L = ["", f"METHOD_WEIGHTS SWEEP  (scenes:flavor RRF ratio, objective {metric.upper()})",
         f"  {n_flavor} gold queries have a flavor channel — only those move under the ratio",
         f"    {'scenes':>7} {'flavor':>7}   {metric:>8}"]
    for s, sc in history:
        mark = "  <- best" if abs(s - best["scenes"]) < 1e-9 else ""
        L.append(f"    {s:>7.2f} {1 - s:>7.2f}   {sc:>8.4f}{mark}")
    L.append(f"  best method_weights = {{'scenes': {best['scenes']}, 'flavor': {best['flavor']}}}"
             f"   ({metric} {best_score:.4f})")
    return "\n".join(L)


# ---- CLI ----

# ** LOCKED **
# CLI string -> normalize value ('none'/'raw'/'null'/'' -> None).
def _norm_arg(s: str) -> str | None:
    return None if s.lower() in ("none", "raw", "null", "") else s.lower()


# CLI entry: parse --mode / --tune, run the chosen A/B (or the RRF-ratio sweep), print the report.
def main():
    import argparse
    import search
    ap = argparse.ArgumentParser(
        description="A/B two read-path configs on the gold set (book-match accuracy).")
    ap.add_argument("--mode", default="norm", choices=("norm", "lift", "flavor"),
                    help="norm: A/B the z-normalize setting (--a vs --b). "
                         "lift: summary-only (A) vs summary+svos moments (B). "
                         "flavor: what-happens only (A) vs + descriptors RRF merge (B).")
    ap.add_argument("--a", default="none", help="normalize for A (mode=norm): none|zscore|minmax")
    ap.add_argument("--b", default="zscore", help="normalize for B (mode=norm): none|zscore|minmax")
    ap.add_argument("--normalize", default="zscore",
                    help="z-norm held fixed for mode=lift/flavor: none|zscore|minmax")
    ap.add_argument("--limit", type=int, default=10, help="results retrieved per query")
    ap.add_argument("--gold", default=None, help="path to a gold query json (default: webtest gold)")
    ap.add_argument("--tune", action="store_true",
                    help="sweep the scenes:flavor RRF ratio (method_weights) for best --metric")
    ap.add_argument("--metric", default="mrr", help="tune objective: mrr | hit@K | p@K")
    args = ap.parse_args()

    queries, gold = load_gold(args.gold)                       # gold entries + judgments

    if args.tune:
        nrm = _norm_arg(args.normalize)
        client = search.open_client()
        try:
            channels = collect_channels(client, queries, normalize=nrm, limit=args.limit)  # one expensive pass
        finally:
            client.close()
        n_flavor = sum(1 for ch in channels.values() if ch["flavor"])
        best, best_score, hist = tune_method_weights(channels, gold, metric=args.metric, limit=args.limit)  # sweep
        base = search.DEFAULT_METHOD_WEIGHTS
        base_run = rrf_from_channels(channels, base, limit=args.limit)   # default ratio
        best_run = rrf_from_channels(channels, best, limit=args.limit)   # tuned ratio
        cmp = compare_runs(base_run, best_run, gold,
                           label_a=f"default {base['scenes']}/{base['flavor']}",
                           label_b=f"tuned {best['scenes']}/{best['flavor']}")
        print(format_comparison(cmp))
        print(format_tuning(hist, best, best_score, args.metric, n_flavor))
        return

    client = search.open_client()
    try:
        if args.mode == "lift":
            nrm = _norm_arg(args.normalize)
            run_a = run_search(client, queries, use_moments=False, normalize=nrm, limit=args.limit)  # summary only
            run_b = run_search(client, queries, use_moments=True, normalize=nrm, limit=args.limit)   # + svos
            la, lb = "summary_only", "summary+svos"
        elif args.mode == "flavor":
            nrm = _norm_arg(args.normalize)
            run_a = run_search(client, queries, use_descriptors=False, normalize=nrm, limit=args.limit)  # what-happens
            run_b = run_search(client, queries, use_descriptors=True, normalize=nrm, limit=args.limit)   # + descriptors
            la, lb = "what_happens", "+descriptors"
        else:  # norm
            run_a = run_search(client, queries, normalize=_norm_arg(args.a), limit=args.limit)
            run_b = run_search(client, queries, normalize=_norm_arg(args.b), limit=args.limit)
            la, lb = f"norm={args.a}", f"norm={args.b}"
    finally:
        client.close()

    cmp = compare_runs(run_a, run_b, gold, label_a=la, label_b=lb)   # score + diff A vs B
    print(format_comparison(cmp))


if __name__ == "__main__":
    main()
