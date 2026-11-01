# HyDE 10.4 — apply the adapter at query time + measure it. adapt() = the runtime door: embed a query, map
# through W, return it as a search channel_vectors override for the summary channel. main() = the gate: a
# baseline-vs-adapter gold A/B by sharpness. W loads lazily on first adapt().

import numpy as np

from utils import SrcPaths, log
from utils.vectorstore import embed, QUERY_PREFIX
from train.fit_adapter import W_FILE

_W = None


# Lazily load + cache the fitted adapter matrix W (D+1, D); a clear error if it has not been fit yet.
def _weights() -> np.ndarray:
    global _W
    if _W is None:
        path = SrcPaths.ADAPTER_DIR / W_FILE
        if not path.exists():
            raise FileNotFoundError(f"no fitted adapter at {path} — run `python -m train.build_dataset` "
                                    f"then `python -m train.fit_adapter` first")
        _W = np.load(path).astype(np.float32)
    return _W


# ** MAIN ** — query.run / evals feed this into search.search as channel_vectors
# Map a query -> {"summary": adapted vector} (embed QUERY-side, apply W, unit-normalize); empty query -> {}.
def adapt(summary: str) -> dict:
    summary = (summary or "").strip()
    if not summary:
        return {}
    x = np.asarray(embed([QUERY_PREFIX + summary])[0], dtype=np.float32)   # bge query-side embedding
    y = np.append(x, 1.0) @ _weights()                                     # ridge map with the bias column
    y = y / (np.linalg.norm(y) + 1e-9)                                     # unit length -> cosine-ready
    return {"summary": y.tolist()}


# ** ENTRY ** — the gate: baseline (summary text) vs +adapter (summary channel overridden), per sharpness.
def main():
    import search
    import evals
    queries, gold = evals.load_gold()                                      # the 100-query gold + judgments
    client = search.open_client()
    try:
        base = evals.run_search(client, queries)                           # normal text read path
        adapted = evals.run_search(client, queries,
                                   adapter=lambda e: adapt(e.get("summary")))   # summary channel from the adapter
    finally:
        client.close()
    cmp = evals.compare_runs(base, adapted, gold, label_a="baseline", label_b="+adapter")
    print(evals.format_comparison(cmp))                                    # includes the by-sharpness top1 rollup


if __name__ == "__main__":
    main()
