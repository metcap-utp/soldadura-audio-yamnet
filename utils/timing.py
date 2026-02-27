from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Optional


@dataclass(frozen=True)
class TimingResult:
    seconds: float

    @property
    def minutes(self) -> float:
        return self.seconds / 60

    @property
    def hours(self) -> float:
        return self.seconds / 3600


@contextmanager
def timer(
    label: str,
    *,
    print_fn: Callable[[str], None] = print,
    enabled: bool = True,
) -> Iterator[Callable[[], TimingResult]]:
    """Simple context-manager timer.

    Usage:
        with timer("Cargando datos") as get_elapsed:
            ...
        elapsed = get_elapsed().seconds
    """

    if not enabled:

        def _get_elapsed_disabled() -> TimingResult:
            return TimingResult(0.0)

        yield _get_elapsed_disabled
        return

    start = time.perf_counter()

    def _get_elapsed() -> TimingResult:
        return TimingResult(time.perf_counter() - start)

    print_fn(f"[TIMER] {label}...")
    try:
        yield _get_elapsed
    finally:
        elapsed = _get_elapsed().seconds
        print_fn(f"[TIMER] {label}: {elapsed:.2f}s ({elapsed / 60:.2f}min)")
