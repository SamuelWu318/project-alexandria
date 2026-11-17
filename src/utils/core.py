from pathlib import Path
from dataclasses import dataclass
import log

@dataclass(frozen=True)
class ProjectPaths:
    ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent

    SRC_DIR: Path = ROOT_DIR / "src"
    TESTING_DIR: Path = ROOT_DIR / "testing"
    LOGS_DIR: Path = ROOT_DIR / "logs"

    DATA_DIR: Path = TESTING_DIR / "data"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    ZIPFILES_DIR: Path = DATA_DIR / "zipfiles"
    HTML_DIR: Path = DATA_DIR / "html"

    DEBUG: Path = LOGS_DIR / "debug.log"
    CATALOG: Path = TESTING_DIR / "pg_catalog.csv"

def log_return(id: int, msg: str, obj): # return wrapper
    if obj:
        str_obj = str(obj)[:100] + "..." if len(str(obj)) > 102 else str(obj)
        log.success("Book ID: %d | %s returned | %s", id, msg, str_obj)
    else: 
        log.warning("Book ID: %d | %s empty", id, msg)
    return obj