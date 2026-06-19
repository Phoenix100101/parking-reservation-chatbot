"""Central logging configuration.

Call :func:`setup_logging` once at process start-up (e.g. in ``main.py``).
Everywhere else, get a module-scoped logger with::

    import logging
    logger = logging.getLogger(__name__)

The ``__name__`` hierarchy (``core.nodes.dynamic_node`` etc.) lets you raise or
lower the level of a single subtree without touching call sites.

Logs go to **stderr** so they never interleave with the chatbot's stdout UI.
"""

import logging
from logging.config import dictConfig

_CONFIGURED = False


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once. Idempotent across repeated calls."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s %(levelname)-7s %(name)s — %(message)s",
                    "datefmt": "%H:%M:%S",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                    "formatter": "default",
                },
            },
            "root": {"level": level.upper(), "handlers": ["console"]},
            "loggers": {
                # Quieten noisy third-party libraries.
                "httpx": {"level": "WARNING"},
                "httpcore": {"level": "WARNING"},
                "openai": {"level": "WARNING"},
                "urllib3": {"level": "WARNING"},
            },
        }
    )
    _CONFIGURED = True
    logging.getLogger(__name__).debug("logging configured at level %s", level.upper())