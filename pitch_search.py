import time


DEFAULT_SEARCH_SEGMENTS = (
    (1, 700),
    (0, 150),
    (-1, 1400),
    (0, 150),
    (1, 700),
    (0, 800),
)


def _now_ms():
    if hasattr(time, "ticks_ms"):
        return time.ticks_ms()
    return int(time.monotonic() * 1000)


def _elapsed_ms(now_ms, start_ms):
    if hasattr(time, "ticks_diff"):
        return time.ticks_diff(now_ms, start_ms)
    return now_ms - start_ms


class PitchSearchController:
    def __init__(self, enabled=True, start_delay_ms=500, search_error=2.0,
                 segments=None):
        self.enabled = bool(enabled)
        self.start_delay_ms = max(0, int(start_delay_ms))
        self.search_error = abs(float(search_error))
        self.segments = segments or DEFAULT_SEARCH_SEGMENTS
        self._missing_since_ms = None
        self._segment_started_ms = None
        self._segment_index = 0
        self._active = False

    def reset(self):
        was_active = self._active
        self._missing_since_ms = None
        self._segment_started_ms = None
        self._segment_index = 0
        self._active = False
        return was_active

    def update(self, motor, allow_drive=True, now_ms=None):
        now_ms = _now_ms() if now_ms is None else now_ms

        if not allow_drive:
            motor.stop()
            self.reset()
            return "CONTROL DISABLED -> HOLD"

        if not self.enabled:
            motor.stop()
            self.reset()
            return "NO RECT -> MOTOR HOLD"

        if self._missing_since_ms is None:
            self._missing_since_ms = now_ms
            motor.stop()
            return "NO RECT -> SEARCH WAIT"

        if not self._active:
            if _elapsed_ms(now_ms, self._missing_since_ms) < self.start_delay_ms:
                motor.stop()
                return "NO RECT -> SEARCH WAIT"
            self._active = True
            self._segment_index = 0
            self._segment_started_ms = now_ms

        direction, duration_ms = self.segments[self._segment_index]
        if _elapsed_ms(now_ms, self._segment_started_ms) >= duration_ms:
            motor.stop()
            self._segment_index = (self._segment_index + 1) % len(self.segments)
            self._segment_started_ms = now_ms
            direction, _ = self.segments[self._segment_index]

        if direction == 0:
            motor.stop()
            return "PITCH SEARCH PAUSE"

        command = self.search_error if direction > 0 else -self.search_error
        motor.drive(0.0, command, allow_drive=True)
        return "PITCH SEARCH +" if direction > 0 else "PITCH SEARCH -"
