# FOR CLAUDE — Retrieval eval + A/B comparator for the read path.
# -----------------------------------------------------------------------------
# Answers ONE question: does approach A or approach B retrieve better on the gold set?
#
# Ground truth is BOOK-LEVEL. The gold queries (webtest/gold/test_queries.json) each
# name the book_id they should surface + a sharpness 1..5 (1 = unique to the book,
# 5 = generic archetype). There is no per-scene label, so a result counts as a HIT when
# its book_id == the query's target book_id. Metrics are therefore book-match metrics:
#   * MRR      — 1/rank of the first hit (how high the right book lands)
#   * Hit@k    — did ANY hit land in the top k (recall proxy)
#   * P@k      — fraction of the top k from the target book (precision proxy)
# broken down by sharpness, since generic queries are expected to fall off.
#
# The scorer (`score_run` / `compare_runs`) takes OUTPUTS — a Run is just
#   {query_id: [ {"scene_id", "book_id", "score"}, ... ]}  ranked best-first
# so it can grade ANY two result sets (raw-vs-normalized fuse, two weight profiles, a
# future new vector) without re-running search. `run_fused` is a thin driver that
# produces a Run from search.search_fused for a given config.
#
# Run:  python -m evals                     (from src/project_alexandria/, venv active)
#       python -m evals --a none --b zscore  # compare raw-cosine fuse vs z-scored fuse
# Stop the webtest server first — both open the same on-disk Qdrant (single-process).
# -----------------------------------------------------------------------------
from __future__ import annotations
import json
from pathlib import Path

# --- types (documentation only) ---
# Run  = dict[query_id, list[{"scene_id": str, "book_id": str, "score": float}]]  best-first
# Gold = dict[query_id, {"book_id": str, "sharpness": int | None}]

GOLD_PATH = Path(__file__).resolve().parent / "webtest" / "gold" / "test_queries.json"
DEFAULT_KS = (1, 3, 5, 10)


# --- gold loading --- #

def load_gold(path: Path | str | None = None) -> tuple[list[dict], dict]:
    """Load the gold query set. Returns (raw query entries, gold judgments).

    Each raw entry is the distilled frame (summary/subject/verb/object/setting/descriptors)
    plus id/book_id/sharpness; `gold` maps query id -> {book_id, sharpness} (the judgment).
    """
    data = json.loads(Path(path or GOLD_PATH).read_text(encoding="utf-8"))
    queries = data["test_queries"]
    gold = {e["id"]: {"book_id": e["book_id"], "sharpness": e.get("sharpness")}
            for e in queries}
    return queries, gold


# --- scoring (operates on OUTPUTS, no search needed) --- #

def _target(g) -> str:
    """The target book_id for a gold entry (accepts a bare id or a {book_id,...} dict)."""
    return g["book_id"] if isinstance(g, dict) else g


def score_run(run: dict, gold: dict, ks: tuple = DEFAULT_KS) -> dict:
    """Grade one Run against the gold on book-match relevance.

    A result is relevant iff its book_id == the query's target book_id. Returns aggregate
    MRR / Hit@k / P@k (mean over queries) plus the per-query breakdown used by compare_runs
    and the sharpness rollup.
    """
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


def by_sharpness(scored: dict, k: int = 5) -> dict:
    """Mean MRR + Hit@k grouped by query sharpness (1 sharp .. 5 generic)."""
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


# --- comparison (A vs B) --- #

def compare_runs(run_a: dict, run_b: dict, gold: dict, *,
                 label_a: str = "A", label_b: str = "B",
                 ks: tuple = DEFAULT_KS, k_overlap: int = 5) -> dict:
    """Score two Runs and diff them: aggregate deltas, per-query head-to-head, divergence.

    `winner` per query is decided on reciprocal rank (how high the first correct-book scene
    lands). `overlap` is the Jaccard of the two approaches' top-`k_overlap` scene ids — how
    differently they retrieved, independent of who was right — so low-overlap + flipped
    winner marks the queries where the change actually did something.
    """
    a = score_run(run_a, gold, ks)
    b = score_run(run_b, gold, ks)
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


# --- text report --- #

def format_comparison(cmp: dict) -> str:
    """Render compare_runs() output as a plain-text report (no deps)."""
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


# --- driver: produce a Run from search.search_fused --- #

def run_fused(client, queries: list[dict], *, normalize: str | None = "zscore",
              weights: dict | None = None, limit: int = 10) -> dict:
    """Drive search_fused over the gold queries with one config -> a Run.

    `normalize` / `weights` are the knobs under test; everything else is held fixed so a
    comparison isolates them. Returns {query_id: ranked [{scene_id, book_id, score}]}.
    """
    import search
    run: dict[str, list] = {}
    for e in queries:
        frame = {k: e.get(k) for k in ("summary", "subject", "verb", "object", "setting")}
        frame["descriptors"] = e.get("descriptors") or []
        try:
            pts = search.search_fused(client, frame, weights=weights,
                                      limit=limit, normalize=normalize)
        except Exception as ex:                 # a bad frame shouldn't sink the whole run
            print(f"[evals] {e['id']}: {type(ex).__name__}: {ex}")
            run[e["id"]] = []
            continue
        run[e["id"]] = [{"scene_id": p.payload.get("scene_id"),
                         "book_id": p.payload.get("book_id"),
                         "score": p.score} for p in pts]
    return run


# --- weight tuning (collect pools once, sweep weights in pure Python) --- #

def collect_pools(client, queries: list[dict], *, limit: int = 10,
                  weights: dict | None = None) -> dict:
    """Retrieve every gold query's pool + per-field raw cosines ONCE (the expensive part).

    Uses DEFAULT_FIELD_WEIGHTS by default so ALL present frame fields are scored — the
    tuner can then raise or zero any of them for free. Returns
        {qid: {"ids": [...], "scores": {field: {id: cos}}, "meta": {id: (scene_id, book_id)}}}
    """
    import search
    weights = weights or search.DEFAULT_FIELD_WEIGHTS
    pools: dict[str, dict] = {}
    for e in queries:
        frame = {k: e.get(k) for k in ("summary", "subject", "verb", "object", "setting")}
        frame["descriptors"] = e.get("descriptors") or []
        try:
            cands, ids, scores = search._fused_pool(client, frame, weights, limit=limit, flt=None)
        except Exception as ex:
            print(f"[evals] {e['id']}: {type(ex).__name__}: {ex}")
            pools[e["id"]] = {"ids": [], "scores": {}, "meta": {}}
            continue
        meta = {c.id: (c.payload.get("scene_id"), c.payload.get("book_id")) for c in cands}
        pools[e["id"]] = {"ids": ids, "scores": scores, "meta": meta}
    return pools


def run_from_pools(pools: dict, weights: dict, normalize: str | None = "zscore",
                   limit: int = 10) -> dict:
    """Re-fuse cached pools under `weights` -> a Run. Pure math: no embeds, no Qdrant."""
    import search
    run: dict[str, list] = {}
    for qid, cell in pools.items():
        if not cell["ids"]:
            run[qid] = []
            continue
        ranked = search._fuse(cell["ids"], cell["scores"], weights, normalize=normalize, limit=limit)
        meta = cell["meta"]
        run[qid] = [{"scene_id": meta[i][0], "book_id": meta[i][1], "score": s}
                    for i, s in ranked]
    return run


def _metric_value(agg: dict, metric: str) -> float:
    """Pull one scalar objective out of a score_run aggregate."""
    if metric == "mrr":
        return agg["mrr"]
    if metric.startswith("hit@"):
        return agg["hit"][int(metric[4:])]
    if metric.startswith("p@"):
        return agg["prec"][int(metric[2:])]
    raise ValueError(f"unknown metric {metric!r} (use mrr, hit@K, or p@K)")


def tune(pools: dict, gold: dict, base: dict, *, normalize: str | None = "zscore",
         metric: str = "mrr", limit: int = 10,
         grid: tuple = (0.0, 0.03, 0.06, 0.10, 0.15, 0.20, 0.28, 0.40),
         passes: int = 4) -> tuple:
    """Coordinate-ascent the field weights to maximize `metric` on the gold set.

    Only ratios matter (_fuse renormalizes), so each field is swept over `grid` (0.0 lets
    the tuner drop a field) with the others held, keeping the best value, repeated for up
    to `passes` sweeps or until a full sweep yields no gain. Cheap because every trial
    re-fuses the cached pools. Returns (best_weights, best_score, history).
    """
    w = dict(base)

    def obj(weights: dict) -> float:
        run = run_from_pools(pools, weights, normalize, limit)
        return _metric_value(score_run(run, gold)["aggregate"], metric)

    best = obj(w)
    history = [("base", dict(w), best)]
    for _ in range(passes):
        improved = False
        for f in list(base.keys()):
            cur, best_v, best_s = w[f], w[f], best
            for v in grid:
                if v == cur:
                    continue
                w[f] = v
                s = obj(w)
                if s > best_s + 1e-9:
                    best_s, best_v = s, v
            w[f] = best_v
            if best_s > best + 1e-9:
                best, improved = best_s, True
                history.append((f, dict(w), best))
        if not improved:
            break
    return w, best, history


def _renorm(w: dict) -> dict:
    """Renormalize positive weights to sum 1.00 (for display); non-positive -> 0."""
    tot = sum(v for v in w.values() if v > 0) or 1.0
    return {f: (v / tot if v > 0 else 0.0) for f, v in w.items()}


def format_tuning(base: dict, tuned: dict, score: float, history: list, metric: str) -> str:
    """Render the default -> tuned weight table + the improving moves."""
    b, t = _renorm(base), _renorm(tuned)
    L = ["", f"TUNED FIELD WEIGHTS  (coordinate ascent on {metric.upper()})",
         f"  {'field':<12} {'default':>8} {'tuned':>8}"]
    for f in base:
        L.append(f"  {f:<12} {b.get(f,0):>8.3f} {t.get(f,0):>8.3f}")
    L.append(f"  best {metric} = {score:.4f}   ({len(history)-1} improving moves)")
    if len(history) > 1:
        L.append("  moves: " + " -> ".join(h[0] for h in history[1:]))
    return "\n".join(L)


def _norm_arg(s: str) -> str | None:
    """CLI string -> normalize value ('none'/'raw'/'null' -> None)."""
    return None if s.lower() in ("none", "raw", "null", "") else s.lower()


def main():
    import argparse
    import search
    ap = argparse.ArgumentParser(
        description="Compare two fused-search configs on the gold set (book-match accuracy).")
    ap.add_argument("--a", default="none", help="normalize for approach A: none|zscore|minmax")
    ap.add_argument("--b", default="zscore", help="normalize for approach B: none|zscore|minmax")
    ap.add_argument("--limit", type=int, default=10, help="results retrieved per query")
    ap.add_argument("--gold", default=None, help="path to a gold query json (default: webtest gold)")
    ap.add_argument("--tune", action="store_true",
                    help="coordinate-ascent field weights (uses --b as the normalizer)")
    ap.add_argument("--metric", default="mrr", help="tuning objective: mrr|hit@K|p@K")
    args = ap.parse_args()

    queries, gold = load_gold(args.gold)

    if args.tune:
        normalize = _norm_arg(args.b)
        client = search.open_client()
        try:
            pools = collect_pools(client, queries, limit=args.limit)   # one expensive pass
        finally:
            client.close()
        base = dict(search.DEFAULT_FIELD_WEIGHTS)
        best_w, best_s, hist = tune(pools, gold, base, normalize=normalize,
                                    metric=args.metric, limit=args.limit)
        base_run = run_from_pools(pools, base, normalize, args.limit)
        tuned_run = run_from_pools(pools, best_w, normalize, args.limit)
        cmp = compare_runs(base_run, tuned_run, gold,
                           label_a="default_w", label_b="tuned_w")
        print(format_comparison(cmp))
        print(format_tuning(base, best_w, best_s, hist, args.metric))
        return

    client = search.open_client()
    try:
        run_a = run_fused(client, queries, normalize=_norm_arg(args.a), limit=args.limit)
        run_b = run_fused(client, queries, normalize=_norm_arg(args.b), limit=args.limit)
    finally:
        client.close()

    cmp = compare_runs(run_a, run_b, gold,
                       label_a=f"norm={args.a}", label_b=f"norm={args.b}")
    print(format_comparison(cmp))


if __name__ == "__main__":
    main()
