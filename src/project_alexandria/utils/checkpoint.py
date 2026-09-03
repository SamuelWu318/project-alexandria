from __future__ import annotations
import shutil
from pathlib import Path
from typing import Any, Callable
from utils.read_write import read_json, write_json

# ---- resumable per-item work cache (LOCKED · MAIN — used by segmentation + enrichment) ----
# One dir per book, one atomic json per finished item; missing/corrupt reads -> None (recompute);
# the dir is dropped once the book completes. storage owns the atomic bytes, this owns the resume
# lifecycle. An optional load/dump codec keeps a typed payload working without dragging that type
# into the IO layer; a codec error on load counts as corrupt (-> recompute).

# ** LOCKED **  ** MAIN ** — instantiated by process.segment_book and embed.enrich_file
class Checkpoint:

    # ** LOCKED **
    # Bind this cache to <base>/<name>/ and remember the optional load/dump codec.
    def __init__(self, base: str | Path, name: str,
                 load: Callable[[Any], Any] | None = None,
                 dump: Callable[[Any], Any] | None = None):
        self.dir = Path(base) / name
        self._load = load   # raw json value -> object (may raise -> treated as corrupt)
        self._dump = dump   # object -> json-able value (default: store the value as-is)

    # ** LOCKED **
    # The on-disk path for one cache key.
    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    # ** LOCKED **
    # Load one key: None if missing, corrupt, or rejected by the codec (all mean "recompute").
    def load(self, key: str) -> Any | None:
        raw = read_json(self._path(key))        # None if missing or corrupt json
        if raw is None or self._load is None:
            return raw
        try:
            return self._load(raw)
        except Exception:
            return None                         # codec-invalid -> recompute

    # ** LOCKED **
    # Persist one key atomically (encoding via the dump codec if given).
    def save(self, key: str, obj: Any) -> None:
        payload = self._dump(obj) if self._dump else obj
        write_json(self._path(key), payload, indent=None)   # atomic, parents created

    # ** LOCKED **
    # Delete the whole cache dir once its book is done.
    def clear(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)
