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