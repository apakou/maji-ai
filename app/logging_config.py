from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_setup_done = False


def _parse_level(name: str) -> int:
    level = getattr(logging, name.upper(), None)
    if isinstance(level, int):
        return level
    return logging.DEBUG


def _truthy_env(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")


class _ExactLevelFilter(logging.Filter):
    """Only allows log records that match exactly one level."""

    def __init__(self, level: int) -> None:
        super().__init__()
        self.level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == self.level


def _make_rotating_handler(
    log_path: Path,
    level: int,
    formatter: logging.Formatter,
    exact_level: bool = True,
) -> RotatingFileHandler:
    """
    Build a RotatingFileHandler for a single log file.

    Args:
        log_path: Full path to the log file.
        level: The logging level this handler accepts.
        formatter: Shared formatter instance.
        exact_level: If True, only records at exactly this level are written.
                     If False, all records at this level and above are written.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        str(log_path),
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=5,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)
    if exact_level:
        handler.addFilter(_ExactLevelFilter(level))
    return handler


def setup_logging() -> None:
    """
    Configure the root logger with one file per log level.

    Log directory is controlled by the LOG_DIR env var (default: logs/).
    Minimum level is controlled by LOG_LEVEL (default: DEBUG).
    Set LOG_TO_STDOUT=true to also mirror all logs to the console.

    Files created:
        logs/debug.log    — DEBUG only
        logs/info.log     — INFO only
        logs/warning.log  — WARNING only
        logs/error.log    — ERROR and CRITICAL
    """
    global _setup_done
    if _setup_done:
        return
    _setup_done = True

    log_dir = Path(os.getenv("LOG_DIR", "logs"))
    min_level = _parse_level(os.getenv("LOG_LEVEL", "DEBUG"))

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(min_level)

    # ── DEBUG — exactly DEBUG records ─────────────────────────────────────
    root.addHandler(
        _make_rotating_handler(
            log_dir / "debug.log", logging.DEBUG, formatter, exact_level=True
        )
    )

    # ── INFO — exactly INFO records ────────────────────────────────────────
    root.addHandler(
        _make_rotating_handler(
            log_dir / "info.log", logging.INFO, formatter, exact_level=True
        )
    )

    # ── WARNING — exactly WARNING records ─────────────────────────────────
    root.addHandler(
        _make_rotating_handler(
            log_dir / "warning.log", logging.WARNING, formatter, exact_level=True
        )
    )

    # ── ERROR — ERROR and CRITICAL together (exact_level=False) ───────────
    # These two are grouped because CRITICAL always accompanies a serious error.
    root.addHandler(
        _make_rotating_handler(
            log_dir / "error.log", logging.ERROR, formatter, exact_level=False
        )
    )

    # ── Stdout — always on; set LOG_TO_STDOUT=false to suppress ───────────
    if not _truthy_env(os.getenv("LOG_TO_STDOUT", "true")):
        pass
    else:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(min_level)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    # Confirm setup with an info log
    logging.getLogger(__name__).info(
        "Logging initialised | dir=%s min_level=%s",
        log_dir,
        logging.getLevelName(min_level),
    )
