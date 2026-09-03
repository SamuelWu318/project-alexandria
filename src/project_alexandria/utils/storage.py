from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass

# ---- paths: single source of truth for the on-disk layout (LOCKED · no methods) ----
# Every path the pipeline touches lives here; the corpus + all outputs live under MASTER_DIR.
# Edit a path or the layout in ONE place. (The Qdrant collection/vector config lives in
# search.py and the prompts in process.py/embed.py — deliberately NOT here.)

# ** LOCKED **  ** MAIN ** — imported as SrcPaths by nearly every module
@dataclass(frozen=True)
class SrcPaths:
    ROOT_DIR: Path        = Path(__file__).resolve().parent.parent.parent.parent
    MASTER_DIR: Path      = ROOT_DIR / "logs" / "test"
    SRC_DIR: Path         = ROOT_DIR / "src"
    UTILS_DIR: Path       = SRC_DIR / "project_alexandria" / "utils"
    CATALOG_PATH: Path    = MASTER_DIR / "pg_catalog.csv"            # Gutenberg metadata catalog (CSV)
    DATA_DIR: Path        = MASTER_DIR / "data"                      # source book archives, pg{code}-h.zip
    RECALL_DIR: Path      = MASTER_DIR / "recall"                    # parse cache: metadata.json + books/ shards
    SCENES_DIR: Path      = MASTER_DIR / "scenes"                    # per-book scene records, pg{code}-s.json
    CHECKPOINT_DIR: Path  = MASTER_DIR / "checkpoints"               # segmentation checkpoints (resumable)
    ENRICH_CKPT_DIR: Path = MASTER_DIR / "checkpoints" / "enrich"        # enrichment checkpoints (resumable)
    STATUS_PATH: Path     = MASTER_DIR / "checkpoints" / "status.json"   # {book_id: "completed"} -> skip on rerun
    DB_PATH: Path         = MASTER_DIR / "databases" / "scenes.db"       # local on-disk SQLite mirror (no server needed)
    QDRANT_DIR: Path      = MASTER_DIR / "databases" / "qdrant_db"       # local on-disk Qdrant (no server needed)
    SEGMENTS_DIR: Path    = MASTER_DIR / "segments"                 # pre-segmentation staging (Book.to_json)
    # scene-record registry: a CODE asset shipped with the package (utils/schema/), NOT master data
    SCHEMA_PATH: Path     = UTILS_DIR / "schema" / "scene_schema.json"
