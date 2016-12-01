# -------------------------------------------------------------------#
# Released under the MIT license (https://opensource.org/licenses/MIT)
# Contact: mrinal.haloi11@gmail.com
# Enhancement Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#
import sys
import os
import time
import logging as _logging

from logging import DEBUG
from logging import ERROR
from logging import FATAL
from logging import INFO
from logging import WARN


__all_ = ['setFileHandler', 'vlog', 'setVerbosity', 'getVerbosity', 'debug', 'info', 'warn', 'error', 'fatal']

_logger = _logging.getLogger('deeprl')

_handler = _logging.StreamHandler(sys.stderr)
_handler.setFormatter(_logging.Formatter(_logging.BASIC_FORMAT, None))
_logger.addHandler(_handler)


def setFileHandler(filename, mode='a'):
    global _logger
    _f_handler = _logging.FileHandler(filename, mode=mode)
    _f_handler.setFormatter(_logging.Formatter(_logging.BASIC_FORMAT, None))
    _logger.addHandler(_f_handler)


def _get_file_line():
    f = sys._getframe()
    our_file = f.f_code.co_filename
    f = f.f_back
    while f:
        code = f.f_code
        if code.co_filename != our_file:
            return (code.co_filename, f.f_lineno)
        f = f.f_back
    return ('<unknown>', 0)


def _log_prefix(file_and_line=None):
    # current time
    now = time.time()
    now_tuple = time.localtime(now)
    now_millisecond = int(1e5 * (now % 1.0))
    # current filename and line
    filename, line = file_and_line or _get_file_line()
    basename = os.path.basename(filename)

    s = ' %02d%02d:%02d:%02d:%02d.%03d:%s:%d] ' % (
        now_tuple[1],  # month
        now_tuple[2],  # day
        now_tuple[3],  # hour
        now_tuple[4],  # min
        now_tuple[5],  # sec
        now_millisecond, basename, line)

    return s


def vlog(level, msg, *args, **kwargs):
    _logger.log(level, msg, *args, **kwargs)


def setVerbosity(verbosity=0):
    _logger.setLevel(verbosity)


def getVerbosity():
    return _logger.getEffectiveLevel()


def debug(msg, *args, **kwargs):
    _logger.debug(_log_prefix() + msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    _logger.info(_log_prefix() + msg, *args, **kwargs)


def warn(msg, *args, **kwargs):
    _logger.warn(_log_prefix() + msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    _logger.error(_log_prefix() + msg, *args, **kwargs)


def fatal(msg, *args, **kwargs):
    _logger.fatal(_log_prefix() + msg, *args, **kwargs)
