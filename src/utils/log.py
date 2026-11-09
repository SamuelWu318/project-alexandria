from utils.core import logging, RotatingFileHandler, re
from utils.core import ProjectPaths as proj

"""
guide to logging levels
DEBUG: temporary debugging logs
SKIP: skipping a file/step for any reason
STEP: basic step (start of a method, moving through a method)
INFO: a method that aggregates multiple step methods
WARNING: potential issues in returning/method call
SUCCESS: success in returning/method call
ERROR: error that should be addressed by the program
CRITICAL: error that should stop the program
"""

_LEVELS = {
    10: "DEBUG",  15: "SKIP",     18: "STEP",
    20: "INFO",   30: "WARNING",  35: "SUCCESS",
    40: "ERROR",  50: "CRITICAL"
}

_COLORS = {
    10: "\x1b[37m",  15: "\x1b[33m",  18: "\x1b[34m",
    20: "\x1b[36m",  30: "\x1b[95m",  35: "\x1b[32m",
    40: "\x1b[91m",  50: "\x1b[91m"
}

class ColorFormatter(logging.Formatter):
    """Color Formatting wrapper class for _FORMAT"""
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)

    def format(self, record):
        msg = super().format(record)

        find = f"[{record.levelname}]"
        replace = f"{_COLORS[record.levelno]}{find}\x1b[0m"
        color_msg = re.sub(re.escape(find), replace, msg)

        return color_msg

_LOGGER = "main"

_FMT="%(asctime)s - [%(levelname)s] - (%(filename)s:%(lineno)d): %(message)s"
_DATEFMT="%Y-%m-%d %H:%M:%S"

_CONSOLE_FORMAT = ColorFormatter(_FMT, _DATEFMT)
_FILE_FORMAT = logging.Formatter(_FMT, _DATEFMT)

_LOG_LEVEL = 10
_CONSOLE_LEVEL = 15
_FILE_LEVEL = 10

def _logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER)
    if logger.handlers: return logger

    # build all additional level calls
    for num, name in _LEVELS.items():
        if logging.getLevelName(num) == (name): continue
        logging.addLevelName(num, name)
        setattr(logging, name, num)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(_CONSOLE_LEVEL)
    console_handler.setFormatter(_CONSOLE_FORMAT)

    file_handler = RotatingFileHandler(proj.DEBUG, maxBytes=5_000_000, backupCount=10)
    file_handler.setLevel(_FILE_LEVEL)
    file_handler.setFormatter(_FILE_FORMAT)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(_LOG_LEVEL)
    logger.propagate = False

    return logger

_log = _logger()

def _msg(level: int, msg: str, *args):
    if not _log.isEnabledFor(level): return
    _log.log(level, msg, *args, stacklevel=3)

def debug(msg: str, *args: object) -> None: _msg(logging.DEBUG, msg, *args)
def skip(msg: str, *args: object) -> None: _msg(logging.SKIP, msg, *args)
def step(msg: str, *args: object) -> None: _msg(logging.STEP, msg, *args)
def info(msg: str, *args: object) -> None: _msg(logging.INFO, msg, *args)
def warning(msg: str, *args: object) -> None: _msg(logging.WARNING, msg, *args)
def success(msg: str, *args: object) -> None: _msg(logging.SUCCESS, msg, *args)
def error(msg: str, *args: object) -> None: _msg(logging.ERROR, msg, *args)
def critical(msg: str, *args: object) -> None: _msg(logging.CRITICAL, msg, *args)