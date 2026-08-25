
import json, os
from typing import Any
from pathlib import Path

def read_text(path: str | Path, default: str | None = None) -> str | None:
    """Read a UTF-8 text file; return `default` if it does not exist."""
    p = Path(path)
    if not p.exists():
        return default
    return p.read_text(encoding="utf-8")

def write_text(path: str | Path, text: str) -> None:
    """Atomically write `text`: parents created, temp file written then os.replaced()."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)

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
