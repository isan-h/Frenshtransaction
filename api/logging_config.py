"""
logging_config.py
------------------
Sets up structured logging for the whole API. Every other module gets
its logger via `logging.getLogger(__name__)` -- this file just
configures WHERE those messages go and HOW they're formatted, once,
at startup.

Why not just use print()? Three concrete reasons that matter in
production:
  1. Severity levels -- you can filter to "show me only warnings and
     errors" without touching code, impossible with print().
  2. Timestamps + module name are automatic -- essential for debugging
     "which request caused this, and when."
  3. Output destination is configurable in ONE place (console now,
     could redirect to a file or a log-aggregation service later)
     without changing any of the code that actually logs messages.
"""

import logging
import sys

from api.config import LOG_LEVEL


def setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
