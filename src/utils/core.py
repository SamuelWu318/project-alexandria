import os, subprocess, sys, time, math
from datetime import datetime
import zipfile, json, logging
from pathlib import Path, PosixPath, PurePath, PurePosixPath
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from typing import Protocol, Any

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