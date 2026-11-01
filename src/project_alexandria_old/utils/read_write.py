import json, os
from typing import Any
from pathlib import Path

# ---- atomic file IO (LOCKED · imported everywhere via the utils package) ----

# ** LOCKED **  ** MAIN ** — the pipeline's only text reader
# Read a UTF-8 text file, or return `default` when it does not exist.
def read_text(path: str | Path, default: str | None = None) -> str | None:
    p = Path(path)
    if not p.exists():
        return default
    return p.read_text(encoding="utf-8")

# ** LOCKED **  ** MAIN ** — the pipeline's only text writer (crash-safe)
# Write `text` atomically: create parents, write a sibling .tmp, then os.replace it into place.
def write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)

# ** LOCKED **  ** MAIN ** — every cache/checkpoint/record read goes through here
# Load JSON from `path`, returning `default` when the file is missing OR corrupt.
def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
            return default

# ** LOCKED **  ** MAIN ** — every cache/checkpoint/record write goes through here
# Serialize `obj` to UTF-8 JSON and write it atomically (non-ASCII preserved, parents created).
def write_json(path: str | Path, obj: Any, indent: int | None = 2) -> None:
    write_text(path, json.dumps(obj, ensure_ascii=False, indent=indent))   # atomic .tmp + os.replace
