"""
Shared logging utilities for memory constructor training scripts.

Provides file logging with TeeStream to capture all output (including tqdm)
to .log/ directory under memory_constructor/.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class TeeStream:
    """Tee stdout/stderr to a log file while preserving original output."""

    def __init__(self, original, log_file):
        self.original = original
        self.log_file = log_file

    def write(self, data):
        self.original.write(data)
        if data.strip():
            self.log_file.write(data)
            self.log_file.flush()

    def flush(self):
        self.original.flush()
        self.log_file.flush()

    def fileno(self):
        return self.original.fileno()

    def isatty(self):
        return self.original.isatty()


def setup_file_logging(name: str, project_root: Path) -> Path:
    """
    Setup file logging to .log/ directory under memory_constructor/.

    Creates a timestamped log file and:
    - Adds a FileHandler to the root logger (captures all logger.info/warning/error)
    - Tees sys.stdout and sys.stderr to the log file (captures tqdm progress bars)

    Args:
        name: Identifier for the log filename (e.g. script name or model name)
        project_root: Path to memory_constructor/ directory

    Returns:
        Path to the created log file
    """
    log_dir = project_root / ".log"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{name}_{timestamp}.log"

    # Add FileHandler to root logger
    fh = logging.FileHandler(str(log_path), mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(fh)

    # Tee stdout/stderr to capture tqdm output (loss, lr, etc.)
    log_file = open(log_path, "a")
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)

    logging.getLogger(__name__).info(f"Log file: {log_path}")
    return log_path
