# HyDE 10.4 — build the self-labelled adapter training set: each scene -> query variants (summary=COMPLEX,
# moment sentences=MEDIUM, subject+verb+object=SPARSE), all targeting that scene's stored summary vector.
# Queries embedded QUERY-side, targets INDEX-side (search's asymmetry); book_id rides each row for v1 negatives.

from pathlib import Path
import glob
import numpy as np

from utils import read_json, write_json, SrcPaths, log
from utils.vectorstore import embed, QUERY_PREFIX   # one contract: embed + the query-side prefix

# dataset artefacts under ADAPTER_DIR (fit_adapter reads these names back)
X_FILE, Y_FILE, META_FILE = "queries.npy", "targets.npy", "dataset_meta.json"


# The SPARSE query variant for a scene: its subject/verb/object action terms as one terse keyword string.
def _sparse_query(rec: dict) -> str:
    terms = (rec.get("subject") or []) + (rec.get("verb") or []) + (rec.get("object") or [])
    return " ".join(t for t in terms if t).strip()


# Walk scenes jsons -> (query_text, kind, scene_id, book_id) rows + {scene_id: summary}; enriched scenes only.
def _collect_rows(scenes_dir) -> tuple[list, dict]:
    rows, target_text = [], {}
    for path in sorted(glob.glob(f"{scenes_dir}/pg*-s.json")):
        for rec in read_json(path, []):
            summ = (rec.get("summary") or "").strip()
            sid = rec.get("scene_id")
            if not summ or not sid:
                continue
            bid = rec.get("book_id")
            target_text[sid] = summ
            rows.append((summ, "complex", sid, bid))                       # COMPLEX: the whole summary
            for m in rec.get("moments") or []:
                sent = (m.get("sentence") or "").strip()
                if sent:
                    rows.append((sent, "medium", sid, bid))                # MEDIUM: one beat sentence
            sparse = _sparse_query(rec)
            if sparse:
                rows.append((sparse, "sparse", sid, bid))                  # SPARSE: s+v+o keyword bag
    return rows, target_text


# ** ENTRY ** — one door: build the variants, embed queries (prefixed) + targets (raw), save X/Y/meta.
def build_dataset(scenes_dir=SrcPaths.SCENES_DIR, out_dir=SrcPaths.ADAPTER_DIR) -> dict:
    rows, target_text = _collect_rows(scenes_dir)                          # self-labelled query->scene variants
    if not rows:
        raise RuntimeError("no enriched scenes with a summary — nothing to build")
    log.step(f"HyDE dataset: {len(rows)} query rows over {len(target_text)} scenes")

    sids = list(target_text)
    tvecs = embed([target_text[s] for s in sids])                          # INDEX-side (raw) target summary vectors
    target_of = dict(zip(sids, tvecs))
    qvecs = embed([QUERY_PREFIX + text for text, *_ in rows])              # QUERY-side (prefixed) query embeddings

    X = np.asarray(qvecs, dtype=np.float32)                                # (N, D) query embeddings
    Y = np.asarray([target_of[sid] for _, _, sid, _ in rows], dtype=np.float32)   # (N, D) matched targets
    meta = [{"scene_id": sid, "book_id": bid, "kind": kind} for _, kind, sid, bid in rows]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / X_FILE, X)
    np.save(out / Y_FILE, Y)
    write_json(str(out / META_FILE), meta)                                 # scene_id/book_id/kind per row (v1 negatives)
    log.done(f"HyDE dataset -> {out}  (X {X.shape}  Y {Y.shape})")
    return {"rows": len(rows), "scenes": len(sids), "dim": int(X.shape[1])}


if __name__ == "__main__":
    build_dataset()
