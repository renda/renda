# -*- coding: utf-8 -*-
from datetime import datetime
from time import perf_counter


def get_timestamp() -> str:  # pragma: no cover
    """
    Returns the current timestamp in this fashion: 2022y08m01d09h05m30s.
    """
    return datetime.now().strftime("%Yy%mm%dd%Hh%Mm%Ss")


class _TimerValue:  # pragma: no cover
    """
    Nicely printable timer value.
    """

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"_TimerValue({self.value})"

    def __str__(self):
        mins, secs = divmod(self.value, 60)
        hours, mins = divmod(mins, 60)
        hours, mins = int(hours), int(mins)
        return f"{hours:3d}:{mins:02d}:{secs:05.2f}"


class Timer:  # pragma: no cover
    """
    Simple timer.
    """

    def __init__(self):
        self._t0 = 0.0
        self._total = 0.0
        self._running = False

    def reset(self):
        """
        Resets or starts the timer.
        """
        self._t0 = perf_counter()
        self._total = 0.0
        self._running = True

    def pause(self):
        """
        Pauses the timer if it is running.
        """
        if self._running:
            self._total += self._time_elapsed()
            self._running = False

    def resume(self):
        """
        Resumes the timer if it has been paused.

        It also starts the timer if it has not been started yet.
        """
        if not self._running:
            self._t0 = perf_counter()
            self._running = True

    @property
    def value(self) -> _TimerValue:
        """
        Returns the current timer value.
        """
        total = self._total
        if self._running:
            total += self._time_elapsed()

        return _TimerValue(total)

    def _time_elapsed(self):
        return perf_counter() - self._t0
