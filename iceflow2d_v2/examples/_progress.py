"""Dependency-free terminal progress for the coupled example."""
import math
import sys
import time


def _duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return '--:--'
    hours, remainder = divmod(max(0, round(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}' if hours else f'{minutes:02d}:{seconds:02d}'


class TerminalProgress:
    """Track accepted steps; estimate remaining time from this invocation only."""

    def __init__(self, total, *, dt, initial=0, enabled=True, stream=None):
        self.total = total
        self.dt = dt
        self.initial = initial
        self.completed = initial
        self.enabled = enabled
        self.stream = sys.stderr if stream is None else stream
        self._started_at = None
        self._last_line_length = 0

    def __enter__(self):
        self._started_at = time.monotonic()
        if self.enabled:
            self._render()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.enabled:
            self.stream.write('\n')
            self.stream.flush()

    def update(self, completed):
        self.completed = completed
        if self.enabled:
            self._render()

    def _render(self):
        elapsed = max(0.0, time.monotonic() - self._started_at)
        advanced = self.completed - self.initial
        remaining = max(0, self.total - self.completed)
        eta = remaining * elapsed / advanced if advanced > 0 else None
        if remaining == 0:
            eta = 0.0
        fraction = min(1.0, self.completed / self.total) if self.total else 1.0
        filled = int(24 * fraction)
        bar = '#' * filled + '-' * (24 - filled)
        line = (
            f'Simulating iceflow2d_v2 [{bar}] {fraction:6.1%} | '
            f'{self.completed}/{self.total} LBM steps | '
            f't={self.completed * self.dt:.6g}/{self.total * self.dt:.6g} s | '
            f'elapsed {_duration(elapsed)} | ETA {_duration(eta)}'
        )
        padding = ' ' * max(0, self._last_line_length - len(line))
        self.stream.write(f'\r{line}{padding}')
        self.stream.flush()
        self._last_line_length = len(line)
