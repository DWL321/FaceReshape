import logging
import os
import re
from platform import system

TEDIUM = 5  ## between DEBUG (10) and NOTSET (0)
logging.addLevelName(TEDIUM, "TEDIUM")


def get_logger(
    name="root",
    *,
    logging_level=logging.DEBUG,
    logging_formatter=logging.Formatter("%(asctime)s [%(levelname)-5.5s] [%(name)-8.8s] %(message)s"),
    logfile="",
):
    logger = logging.getLogger(name)
    logger.setLevel(TEDIUM)

    ## add default (console) handler otherwise you cannot change the logging level/formatter
    handler = logging.StreamHandler()
    handler.setLevel(logging_level)
    handler.setFormatter(logging_formatter)
    logger.addHandler(handler)

    ## file handler, more verbose than default
    if len(logfile) > 0:
        dir = os.path.dirname(logfile)
        os.makedirs(dir, exist_ok=True)
        file_handler = logging.FileHandler(logfile, encoding="utf-8")
        file_handler.setLevel(TEDIUM)
        file_handler.setFormatter(logging_formatter)
        logger.addHandler(file_handler)

    return logger


class LogManager:
    logging_level = logging.DEBUG
    logfile = ""
    abbrs = dict()

    logger = None

    @classmethod
    def update_abbrs(cls, **kwargs):
        cls.abbrs.update(kwargs)
        cls.log(f"abbreviations {cls.abbrs}", abbr=False)

    @classmethod
    def log(cls, msg: str, level=logging.DEBUG, abbr=True):
        if cls.logger is None:
            cls.logger = get_logger(
                logging_level=cls.logging_level,
                logging_formatter=logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s"),
                logfile=cls.logfile,
            )

        if abbr:
            for k, v in cls.abbrs.items():
                abbr = f"<{k}>"
                if v.endswith("/"):
                    abbr += "/"
                msg = msg.replace(v, abbr)

        cls.logger.log(level, msg)


log = LogManager.log
info = lambda msg: log(msg, logging.INFO)
error = lambda msg: log(msg, logging.ERROR)

def _add_ansi_color(msg, level):
    # fmt: off
    if level >= 40:    ## ERROR   (red)
        color = "31"
    elif level >= 30:  ## WARNING (yellow)
        color = "33"
    elif level >= 20:  ## INFO    (normal)
        color = "0"
    else:              ## DEBUG   (dim)
        color = "2"
    # fmt: on
    return f"\x1b[{color}m{msg}\x1b[0m"


def stream_handler_emit(self: logging.StreamHandler, record: logging.LogRecord):
    try:
        msg = self.format(record)
        msg = _add_ansi_color(msg, record.levelno)
        self.stream.write(msg + self.terminator)
        self.flush()
    except RecursionError:
        raise
    except Exception:
        self.handleError(record)


def file_handler_emit(self: logging.FileHandler, record: logging.LogRecord):
    if self.stream is None:
        self.stream = self._open()
    try:
        msg = self.format(record)
        msg = re.sub(r"\x1b\[\d+m", "", msg)
        self.stream.write(msg + self.terminator)
        self.flush()
    except RecursionError:
        raise
    except Exception:
        self.handleError(record)


if system() == "Windows":
    # raise NotImplementedError("TODO: test compatibility on Windows")
    pass
else:
    logging.StreamHandler.emit = stream_handler_emit
    logging.FileHandler.emit = file_handler_emit
