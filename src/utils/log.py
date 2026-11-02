from utils.core import logging, RotatingFileHandler
from utils.core import ProjectPaths as proj

_LEVELS = {
  # 10: "DEBUG",
    12: "SKIP",
    15: "STEP",
  # 20: "INFO",
  # 30: "WARNING",
    35: "SUCCESS"
  # 40: "ERROR",
  # 50: "CRITICAL"
}
_FORMAT = logging.Formatter(
    fmt="%(asctime)s - [%(levelname)s] - (%(filename)s:%(lineno)d): %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
_LOGGER = "alexandria"

def _logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER)
    if logger.handlers: return logger

    # build all additional level calls
    for num, name in _LEVELS.items(): 
        logging.addLevelName(num, name)
        setattr(logging, name, num)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(_FORMAT)

    file_handler = RotatingFileHandler(proj.DEBUG, maxBytes=5_000_000, backupCount=10)
    file_handler.setLevel(1)
    file_handler.setFormatter(_FORMAT)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(1) # capture all
    logger.propagate = False

    return logger

_log = _logger()

def _msg(level: int, msg: str, *args):
    if not _log.isEnabledFor(level): return
    _log._log(level, msg, args, stacklevel=3)

def debug(msg: str, *args: object) -> None: _msg(logging.DEBUG, msg, *args)
def skip(msg: str, *args: object) -> None: _msg(logging.SKIP, msg, *args)
def step(msg: str, *args: object) -> None: _msg(logging.STEP, msg, *args)
def info(msg: str, *args: object) -> None: _msg(logging.INFO, msg, *args)
def warning(msg: str, *args: object) -> None: _msg(logging.WARNING, msg, *args)
def success(msg: str, *args: object) -> None: _msg(logging.SUCCESS, msg, *args)
def error(msg: str, *args: object) -> None: _msg(logging.ERROR, msg, *args)
def critical(msg: str, *args: object) -> None: _msg(logging.CRITICAL, msg, *args)
