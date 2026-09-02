"""Centralised logging setup for ADAPT-IDS."""

from __future__ import annotations

import logging
import sys

_configured = False


def setup_logging(level: str = "INFO", fmt: str | None = None) -> None:
    global _configured
    if _configured:
        return

    fmt = fmt or "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger("adaptive_ids")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(handler)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"adaptive_ids.{name}")
