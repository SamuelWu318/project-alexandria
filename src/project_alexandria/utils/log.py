# FOR CLAUDE — Console logging (stdlib `logging`, configured once).
# -----------------------------------------------------------------------------
# The pipeline calls six semantic helpers — step/info/done/skip/warn/fail — and this
# module backs them with the stdlib `logging` module instead of raw print(): one
# configured logger, timestamps, level names, thread-safe emits, and a single place
# to change format, level, or destination. The caller API is unchanged, so nothing
# in data/process/embed/tests had to move.
#
# Level mapping: warn -> WARNING, fail -> ERROR; step/info/done/skip are INFO with a
# short category tag kept IN the message, so the category stays greppable even though
# they share the INFO level. debug() is silent unless the level is DEBUG.
#
# Verbosity: set ALEXANDRIA_LOG_LEVEL (e.g. DEBUG, WARNING) in the env, or call
# set_level() at runtime.
#
# Destination is stdout on purpose: the harness still writes plain DATA/report lines
# (search hits, query echoes) with print() to stdout, and logging there keeps stage
# headers interleaved in order with those results. Flip the handler to sys.stderr
# (one line below) if you'd rather separate logs from results in a pipeline.
# -----------------------------------------------------------------------------
from __future__ import annotations
import logging, os, sys

_LOGGER_NAME = "alexandria"


def _build_logger() -> logging.Logger:
    """Return the shared logger, configuring its handler exactly once (idempotent)."""
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


def set_level(level: str | int) -> None:
    """Raise/lower verbosity at runtime, e.g. set_level('DEBUG') or set_level('WARNING')."""
    _log.setLevel(level)


def step(msg: str) -> None:
    """Section / stage header."""
    _log.info("=== %s ===", msg)


def info(msg: str) -> None:
    """Progress / neutral status."""
    _log.info("%s", msg)


def done(msg: str) -> None:
    """A unit of work finished or was written."""
    _log.info("done: %s", msg)


def skip(msg: str) -> None:
    """Something was gated, already done, or excluded."""
    _log.info("skip: %s", msg)


def warn(msg: str) -> None:
    """Recoverable problem: a retry, a missing input, a degraded fallback."""
    _log.warning("%s", msg)


def fail(msg: str) -> None:
    """An error, usually logged just before raising."""
    _log.error("%s", msg)


def debug(msg: str) -> None:
    """Verbose detail; silent unless the level is DEBUG."""
    _log.debug("%s", msg)
