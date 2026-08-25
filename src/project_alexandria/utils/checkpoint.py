# FOR CLAUDE — Resumable per-item work cache (built on storage's atomic IO).
# -----------------------------------------------------------------------------
# Both expensive LLM stages resume after a crash the same way: cache each finished
# item to its own json file under a per-book dir, skip anything already cached on
# rerun, and drop the dir once the book is done. That lifecycle —
#     dir  ->  per-item ATOMIC file  ->  None-on-missing/corrupt (recompute)  ->
#     rmtree on completion
# was hand-rolled at BOTH sites (segmentation in process/tests, enrichment in
# embed) with slightly drifted implementations. This collapses it into one class.
#
# Layering: storage.py owns the atomic bytes (.tmp + os.replace); this owns the
# resume LIFECYCLE on top of it. An optional codec keeps a typed payload (e.g. a
# Pydantic model) working WITHOUT dragging that type into the low-level IO layer —
# the caller supplies load/dump, and a codec error on load counts as corrupt
# (-> None -> recompute), exactly as the old inline try/except did.
# -----------------------------------------------------------------------------
from __future__ import annotations
import shutil
from pathlib import Path
from typing import Any, Callable
from project_alexandria.utils.storage import read_json, write_json


class Checkpoint:
    """One book's resume cache: per-key json under <base>/<name>/.

    save() is crash-safe (storage.write_json is atomic + creates parents). load()
    returns None when the file is missing OR unreadable OR the optional codec rejects
    it, so callers keep their "recompute unless cleanly resumable" policy. clear()
    drops the whole dir once the work is done and the checkpoints are dead weight.
    """

    def __init__(self, base: str | Path, name: str,
                 load: Callable[[Any], Any] | None = None,
                 dump: Callable[[Any], Any] | None = None):
        self.dir = Path(base) / name
        self._load = load   # raw json value -> object (may raise -> treated as corrupt)
        self._dump = dump   # object -> json-able value (default: store the value as-is)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def load(self, key: str) -> Any | None:
        raw = read_json(self._path(key))        # None if missing or corrupt json
        if raw is None or self._load is None:
            return raw
        try:
            return self._load(raw)
        except Exception:
            return None                         # codec-invalid -> recompute

    def save(self, key: str, obj: Any) -> None:
        payload = self._dump(obj) if self._dump else obj
        write_json(self._path(key), payload, indent=None)   # atomic, parents created

    def clear(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)
