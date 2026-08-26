# FOR CLAUDE — Centralized filesystem layer.
# -----------------------------------------------------------------------------
# The shared foundation every other module imports: (1) every on-disk path the
# pipeline uses, (2) the only functions that read/write JSON or text on disk, (3) the
# single load_dotenv() call — importing storage populates os.environ for everyone —
# (4) the shared scene-tag vocabulary (Tone / Intensity / Arc), and (5) the shared LLM
# client + model id + error policy + SCHEMA_VERSION. Everything that touches the
# filesystem — data.py, process.py, embed.py, tests.py — imports from here instead of
# hand-rolling `json.dumps(...) + write_text(...)`. Edit a path or the on-disk layout
# in ONE place.
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
# (search.py owns the vector store) and the prompts (process.py / embed.py own those).
# The tag-vocab enums + the LLM client / model / SCHEMA_VERSION DO live here, shared.
# -----------------------------------------------------------------------------
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any
from dataclasses import dataclass

# --- paths (single source of truth; the whole corpus + outputs live under master/) --- #

@dataclass(frozen=True)
class SrcPaths:
    ROOT_DIR: Path        = Path(__file__).resolve().parent.parent.parent.parent
    MASTER_DIR: Path      = ROOT_DIR / "logs" / "test"
    CATALOG_PATH: Path    = MASTER_DIR / "pg_catalog.csv"            # Gutenberg metadata catalog (CSV)
    DATA_DIR: Path        = MASTER_DIR / "data"                      # source book archives, pg{code}-h.zip
    RECALL_DIR: Path      = MASTER_DIR / "recall"                    # parse cache: metadata.json + books.json
    SCENES_DIR: Path      = MASTER_DIR / "scenes"                    # per-book scene records, pg{code}-s.json
    CHECKPOINT_DIR: Path  = MASTER_DIR / "checkpoints"               # segmentation checkpoints (resumable)
    ENRICH_CKPT_DIR: Path = MASTER_DIR / "checkpoints" / "enrich"        # enrichment checkpoints (resumable)
    STATUS_PATH: Path     = MASTER_DIR / "checkpoints" / "status.json"   # {book_id: "completed"} -> skip on rerun
    DB_PATH: Path         = MASTER_DIR / "databases" / "scenes.db"       # local on-disk SQLite mirror (no server needed)
    QDRANT_DIR: Path      = MASTER_DIR / "databases" / "qdrant_db"       # local on-disk Qdrant (no server needed)
    SEGMENTS_DIR: Path    = MASTER_DIR / "segments"                 # pre-segmentation staging (Book.to_json)