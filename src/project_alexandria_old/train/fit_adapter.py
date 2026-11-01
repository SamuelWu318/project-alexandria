# HyDE 10.4 — fit the v0 query adapter: closed-form RIDGE map W from a QUERY embedding to its target SUMMARY
# vector (pure numpy). Shrinks toward the IDENTITY prior, not zero — W = (XaᵀXa + λI)⁻¹(XaᵀY + λP), Xa=[X|1],
# P=[I;0] — so high λ ≈ baseline, low λ bends hard toward the corpus. Its gold lift gates whether v1 is built.

from pathlib import Path
import numpy as np

from utils import SrcPaths, log
from train.build_dataset import X_FILE, Y_FILE   # dataset artefact names (build_dataset is their one owner)

W_FILE = "summary_adapter.npy"
RIDGE_LAMBDA = 100.0   # L2 strength: higher -> closer to a gentle nudge (safe), lower -> fits the corpus harder


# Mean paired-row cosine of two row-aligned matrices (L2-normalized first) — the train-set alignment metric.
def _mean_cosine(A: np.ndarray, B: np.ndarray) -> float:
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return float(np.mean(np.sum(A * B, axis=1)))


# ** ENTRY ** — one door: solve the ridge map, save W, report train alignment gained over raw bge.
def fit_adapter(data_dir=SrcPaths.ADAPTER_DIR, lam: float = RIDGE_LAMBDA) -> dict:
    d = Path(data_dir)
    X = np.load(d / X_FILE).astype(np.float64)          # (N, D) query embeddings
    Y = np.load(d / Y_FILE).astype(np.float64)          # (N, D) target summary vectors
    n, dim = X.shape

    Xa = np.hstack([X, np.ones((n, 1))])                # augment with a bias column -> (N, D+1)
    prior = np.zeros((dim + 1, dim))
    prior[:dim, :] = np.eye(dim)                        # identity prior: x->x (raw bge query), zero bias
    W = np.linalg.solve(Xa.T @ Xa + lam * np.eye(dim + 1),
                        Xa.T @ Y + lam * prior)         # (D+1, D) ridge shrinking toward the prior, not zero

    base = _mean_cosine(X, Y)                           # raw bge query<->target alignment (the baseline)
    fit = _mean_cosine(Xa @ W, Y)                       # adapter query<->target alignment
    np.save(d / W_FILE, W.astype(np.float32))
    log.done(f"HyDE adapter fit (λ={lam}): train cos bge={base:.4f} -> adapter={fit:.4f} "
             f"(+{fit - base:.4f}); W {W.shape} -> {d / W_FILE}")
    return {"dim": dim, "n": n, "cos_bge": base, "cos_adapter": fit, "lambda": lam}


if __name__ == "__main__":
    fit_adapter()
