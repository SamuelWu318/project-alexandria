import os, subprocess, sys, time, math
from datetime import datetime
import zipfile, json, csv, logging, re
from pathlib import Path, PosixPath, PurePath, PurePosixPath
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from typing import Protocol, Any
from collections import defaultdict
import types, inspect
from utils import log

@dataclass(frozen=True)
class ProjectPaths:
    ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent

    SRC_DIR: Path = ROOT_DIR / "src"
    PUBLIC_DIR: Path = ROOT_DIR / "public"
    LOGS_DIR: Path = PUBLIC_DIR / "logs"

    DATA_DIR: Path = PUBLIC_DIR / "data"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    ZIPFILES_DIR: Path = DATA_DIR / "zipfiles"

    DEBUG: Path = LOGS_DIR / "debug.log"
    CATALOG: Path = PUBLIC_DIR / "pg_catalog.csv"

def log_return(id: int, msg: str, obj): # return wrapper
    if obj: log.success("Book ID: %d | %s returned | %s", id, msg, obj)
    else: log.warning("Book ID: %d | %s empty", id, msg)
    return obj