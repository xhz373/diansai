import gc
import time


class _NoopStepperController:
    ready = False

    def drive(self, *args, **kwargs):
        return

    def stop(self):
        return

    def deinit(self):
        return


def build_noop_stepper_controller(axis_overrides=None):
    return _NoopStepperController()


def noop_load_calibration(default_red, default_black, default_violet, default_bright=None):
    return (
        False,
        tuple(default_red),
        tuple(default_black),
        tuple(default_violet),
        default_bright,
    )


def load_motion_support():
    try:
        from k230_common import build_stepper_controller, load_calibration

        return build_stepper_controller, load_calibration
    except ImportError:
        return build_noop_stepper_controller, noop_load_calibration


def apply_saved_thresholds(
    load_calibration,
    red_threshold,
    black_threshold,
    violet_threshold,
    bright_threshold=None,
):
    ok, red, black, violet, bright = load_calibration(
        red_threshold,
        black_threshold,
        violet_threshold,
        bright_threshold,
    )
    if ok:
        print("[Calib] thresholds applied")
    else:
        print("[Calib] using built-in thresholds")
    return ok, red, black, violet, bright


class LoopStatsMixin:
    def _init_loop_stats(self, enable_fps=False):
        self.frame_count = 0
        self.gc_counter = 0
        self.fps = 0.0
        self._fps_enabled = bool(enable_fps)
        self._last_fps_time = time.ticks_ms()

    def _mark_frame(self):
        self.frame_count += 1
        self.gc_counter += 1

    def update_fps(self):
        if not getattr(self, "_fps_enabled", False):
            return
        current_time = time.ticks_ms()
        dt = time.ticks_diff(current_time, self._last_fps_time)
        if dt >= 1000:
            self.fps = self.frame_count * 1000 / dt
            self.frame_count = 0
            self._last_fps_time = current_time

    def maybe_collect_gc(self, gc_interval):
        if self.gc_counter >= gc_interval:
            gc.collect()
            self.gc_counter = 0
