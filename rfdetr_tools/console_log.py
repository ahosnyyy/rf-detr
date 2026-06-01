"""Mirror stdout/stderr to a log file while training."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class _TeeStream:
    def __init__(self, *streams: object) -> None:
        self._streams = streams

    def write(self, data: str) -> None:
        if not data:
            return
        for stream in self._streams:
            stream.write(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        isatty = getattr(self._streams[0], "isatty", None)
        return bool(isatty()) if callable(isatty) else False


def resolve_log_file(value: object, output_dir: Path) -> Path | None:
    """Map config/CLI log_file values to an absolute path under output_dir."""
    if value is None or value is False:
        return None
    if value is True:
        return output_dir / "train.log"
    path = Path(str(value))
    if path.is_absolute():
        return path
    return output_dir / path


@contextmanager
def tee_console_to_file(log_path: Path) -> Iterator[Path]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        stdout = _TeeStream(sys.stdout, log_handle)
        stderr = _TeeStream(sys.stderr, log_handle)
        previous_stdout, previous_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout, stderr  # type: ignore[assignment]
        try:
            yield log_path
        finally:
            sys.stdout, sys.stderr = previous_stdout, previous_stderr
