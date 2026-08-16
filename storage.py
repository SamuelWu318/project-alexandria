# FOR CLAUDE — Centralized filesystem layer.
# -----------------------------------------------------------------------------
# Single source of truth for (1) every on-disk path the pipeline uses and (2) the
# only functions that read/write JSON or text on disk. Everything that touches the
# filesystem — data.py, process.py, embed.py, tests.py — imports from here instead
# of hand-rolling `json.dumps(...) + write_text(...)`. Edit a path or the on-disk
# layout in ONE place.
#
# Why this exists / invariants to preserve:
#   * write_json / write_text are ATOMIC: content goes to a sibling <file>.tmp then
#     os.replace()s into place, so a crash mid-write can never leave a half-written
#     or corrupt file. Every writer in the pipeline relies on this (recall cache,
#     scene records, resumable checkpoints, status file).
#   * read_json returns `default` on a missing OR corrupt file. Callers lean on this
#     for their "recompute if unreadable" policy (caches and checkpoints).
#   * os.replace is atomic only within one filesystem; the .tmp always lives in the
#     destination's own directory, so this holds.
#
# NOT here (owned elsewhere on purpose): the Qdrant collection name / vector config
# (search.py owns the vector store) and all prompts / model settings (process.py,
# embed.py own those).
# -----------------------------------------------------------------------------
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any


# --- paths (single source of truth; the whole corpus + outputs live under master/) --- #

CATALOG_PATH    = "master/pg_catalog.csv"           # Gutenberg metadata catalog (CSV)
DATA_PATH       = "master/data"                      # source book archives, pg{code}-h.zip
RECALL_PATH     = "master/recall"                    # parse cache: metadata.json + books.json
TEST_PATH       = "master/test"                      # ad-hoc payload dumps for inspection
SCENES_PATH     = "master/scenes"                    # per-book scene records, pg{code}-s.json
CHECKPOINT_DIR  = "master/checkpoints"               # segmentation checkpoints (resumable)
ENRICH_CKPT_DIR = "master/checkpoints/enrich"        # enrichment checkpoints (resumable)
STATUS_PATH     = "master/checkpoints/status.json"   # {book_id: "completed"} -> skip on rerun


def read_json(path: str | Path, default: Any = None) -> Any:
    """Load JSON from `path`; return `default` if the file is missing or corrupt."""
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def write_json(path: str | Path, obj: Any, indent: int | None = 2) -> None:
    """Atomically write `obj` as UTF-8 JSON (non-ASCII preserved, parents created)."""
    write_text(path, json.dumps(obj, ensure_ascii=False, indent=indent))


def read_text(path: str | Path, default: str | None = None) -> str | None:
    """Read a UTF-8 text file; return `default` if it does not exist."""
    p = Path(path)
    if not p.exists():
        return default
    return p.read_text(encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    """Atomically write `text`: parents created, temp file written then os.replace()d."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)
