from __future__ import annotations
import logging, os, sys

# ---- console logging (LOCKED · stdlib `logging`, one logger, configured once) ----
# Verbosity: set ALEXANDRIA_LOG_LEVEL in the env, or call set_level() at runtime.
# Emits to stdout so stage headers interleave with the harness's print()ed results;
# point the handler at sys.stderr to split logs from results.

_LOGGER_NAME = "alexandria"

# ** LOCKED **
# Return the shared logger, attaching its stdout handler exactly once (idempotent).
def _build_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:            # already configured on an earlier import
        return logger
    handler = logging.StreamHandler(sys.stdout)   # -> sys.stderr to split logs from results
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(os.getenv("ALEXANDRIA_LOG_LEVEL", "INFO").upper())
    logger.propagate = False       # don't double-emit through the root logger
    return logger


_log = _build_logger()

# ---- semantic helpers (LOCKED · MAIN — called across the whole pipeline via `log`) ----
# warn -> WARNING, fail -> ERROR; step/info/done/skip are INFO with a greppable category tag.

# ** LOCKED **
# Raise or lower verbosity at runtime, e.g. set_level('DEBUG').
def set_level(level: str | int) -> None:
    _log.setLevel(level)

# ** LOCKED **
# Log a section / stage header.
def step(msg: str) -> None:
    _log.info("=== %s ===", msg)

# ** LOCKED **
# Log neutral progress / status.
def info(msg: str) -> None:
    _log.info("%s", msg)

# ** LOCKED **
# Log that a unit of work finished or was written.
def done(msg: str) -> None:
    _log.info("done: %s", msg)

# ** LOCKED **
# Log that something was gated, already done, or excluded.
def skip(msg: str) -> None:
    _log.info("skip: %s", msg)

# ** LOCKED **
# Log a recoverable problem (a retry, a missing input, a degraded fallback).
def warn(msg: str) -> None:
    _log.warning("%s", msg)

# ** LOCKED **
# Log an error, usually just before raising.
def fail(msg: str) -> None:
    _log.error("%s", msg)

# ** LOCKED **
# Log verbose detail; silent unless the level is DEBUG.
def debug(msg: str) -> None:
    _log.debug("%s", msg)
