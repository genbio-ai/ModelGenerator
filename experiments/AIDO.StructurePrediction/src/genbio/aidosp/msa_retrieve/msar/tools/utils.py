
"""Common utilities for data pipeline tools."""
import contextlib
import datetime
import logging
import shutil
import tempfile
import time
from typing import Optional


@contextlib.contextmanager
def tmpdir_manager(base_dir: Optional[str] = None):
    """Context manager that deletes a temporary directory on exit."""
    tmpdir = tempfile.mkdtemp(dir=base_dir)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@contextlib.contextmanager
def timing(msg: str):
    logging.info("Started %s", msg)
    tic = time.perf_counter()
    yield
    toc = time.perf_counter()
    logging.info("Finished %s in %.3f seconds", msg, toc - tic)


def to_date(s: str):
    return datetime.datetime(year=int(s[:4]), month=int(s[5:7]), day=int(s[8:10]))
