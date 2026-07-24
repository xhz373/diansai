import gc
import image
import os
import sys
import time


def _append_import_paths():
    for candidate in (
        ".",
        "/flash",
        "/flash/app",
        "/flash/lib",
        "/sdcard",
        "/sdcard/app",
        "/sdcard/lib",
    ):
        try:
            if candidate not in sys.path:
                sys.path.append(candidate)
        except Exception:
            pass


_append_import_paths()

try:
    from common_hw import (Display, MultiButton, draw_text,
                           camera_init, camera_start, camera_snapshot,
                           camera_deinit,
                           display_init_board)
except ImportError:
    raise ImportError(
        "common_hw not found; copy common_hw.py to /flash, /flash/app, "
        "/flash/lib, /sdcard, /sdcard/app, or /sdcard/lib before running."
    )

try:
    from k230_common import load_calibration, save_calibration
except ImportError:
    def load_calibration(*a):
        return (False,) + tuple(a[1:])

    def save_calibration(*a):
        return False


CAMERA_ID = 2
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 480
SENSOR_HMIRROR = True
SENSOR_VFLIP = True

RED_THRESHOLD = (41, 100, -28, 6, -14, 14)
BLACK_THRESHOLD = (22, 69, -23, -3, -22, 16)
VIOLET_THRESHOLD = (92, 100, -15, 6, -9, 11)

BUILD_TAG = "2026-07-14-key-calib-v2"
SNAPSHOT_RETRY_DELAY_MS = 8
ADJUST_STEP = 2

# Adjust these 4 board GPIO pins to match your wiring.
# Wiring: GPIO pin -> button -> GND, using internal pull-up.
BUTTON_CONFIGS = (
    ("KEY1", 28, 28),
    ("KEY2", 29, 29),
    ("KEY3", 30, 30),
    ("KEY4", 31, 31),
)

LAB_LIMITS = (
    (0, 100),
    (0, 100),
    (-128, 127),
    (-128, 127),
    (-128, 127),
    (-128, 127),
)
LAB_LABELS = ("L Min", "L Max", "A Min", "A Max", "B Min", "B Max")
TARGET_NAMES = ("RECT", "LASER")


class ThresholdCalibrator:
    def __init__(self):
        ok, red, black, violet, _ = load_calibration(
            RED_THRESHOLD, BLACK_THRESHOLD, VIOLET_THRESHOLD
        )
        self.red_threshold = list(red)
        self.black_threshold = list(black)
        self.violet_threshold = list(violet)
        self.initial_black_threshold = list(black)
        self.initial_violet_threshold = list(violet)
        self.active_target_index = 0
        self.active_param_index = 0
        self.status_text = "CALIB READY" if ok else "BUILT-IN THRESHOLDS"
        self.status_expire_ms = 0
        self.keys = MultiButton(BUTTON_CONFIGS)
        self.last_event_text = ""
        self.canvas = self._create_canvas()

    def _create_canvas(self):
        constructors = (
            lambda: image.Image(DISPLAY_WIDTH, DISPLAY_HEIGHT, image.RGB565),
            lambda: image.Image(DISPLAY_WIDTH, DISPLAY_HEIGHT),
            lambda: image.Image(size=(DISPLAY_WIDTH, DISPLAY_HEIGHT)),
        )
        for make in constructors:
            try:
                return make()
            except Exception:
                pass
        raise RuntimeError("failed to create UI canvas")

    def _current_threshold(self):
        if self.active_target_index == 0:
            return self.black_threshold
        return self.violet_threshold

    def _current_initial_threshold(self):
        if self.active_target_index == 0:
            return self.initial_black_threshold
        return self.initial_violet_threshold

    def _switch_target(self):
        self.active_target_index = 1 - self.active_target_index
        self._set_status("TARGET " + TARGET_NAMES[self.active_target_index])

    def _next_parameter(self):
        self.active_param_index = (self.active_param_index + 1) % 6
        self._set_status("PARAM " + LAB_LABELS[self.active_param_index])

    def _normalize_threshold(self, threshold):
        values = list(threshold)
        for index in range(6):
            low, high = LAB_LIMITS[index]
            value = int(values[index])
            if value < low:
                value = low
            if value > high:
                value = high
            values[index] = value

        if values[0] > values[1]:
            values[0] = values[1]
        if values[2] > values[3]:
            values[2] = values[3]
        if values[4] > values[5]:
            values[4] = values[5]
        return values

    def _adjust_threshold(self, index, delta):
        threshold = self._current_threshold()
        threshold[index] += delta
        normalized = self._normalize_threshold(threshold)
        for idx in range(6):
            threshold[idx] = normalized[idx]
        self._set_status(
            "{} {}={}".format(
                TARGET_NAMES[self.active_target_index],
                LAB_LABELS[index],
                threshold[index],
            )
        )

    def _reset_active_threshold(self):
        threshold = self._current_threshold()
        initial = self._current_initial_threshold()
        for index in range(6):
            threshold[index] = int(initial[index])
        self._set_status("RESET " + TARGET_NAMES[self.active_target_index])

    def _save_thresholds(self):
        ok = save_calibration(
            tuple(self.red_threshold),
            tuple(self.black_threshold),
            tuple(self.violet_threshold),
        )
        if ok:
            self._set_status("SAVED")
        else:
            self._set_status("SAVE FAILED")

    def _set_status(self, text, hold_ms=1400):
        self.status_text = text
        self.status_expire_ms = time.ticks_add(time.ticks_ms(), hold_ms)

    def _status_line(self):
        if self.status_expire_ms and time.ticks_diff(self.status_expire_ms, time.ticks_ms()) < 0:
            self.status_expire_ms = 0
            self.status_text = ""
        return self.status_text

    def handle_keys(self):
        for key_name, event_name in self.keys.poll_events():
            self.last_event_text = "{} {}".format(key_name, event_name)
            if key_name == "KEY1":
                if event_name == "short":
                    self._switch_target()
                elif event_name == "long":
                    self._reset_active_threshold()
            elif key_name == "KEY2":
                if event_name == "short":
                    self._next_parameter()
                elif event_name == "long":
                    self._save_thresholds()
            elif key_name == "KEY3":
                if event_name in ("press", "repeat"):
                    self._adjust_threshold(self.active_param_index, -ADJUST_STEP)
            elif key_name == "KEY4":
                if event_name in ("press", "repeat"):
                    self._adjust_threshold(self.active_param_index, ADJUST_STEP)

    def _fill_canvas(self):
        try:
            self.canvas.clear()
            return
        except Exception:
            pass
        self.canvas.draw_rectangle(0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT, color=(18, 18, 18), fill=True)

    def _draw_preview(self, preview_img):
        active_threshold = tuple(self._current_threshold())
        try:
            blob_img = preview_img.copy()
        except Exception:
            blob_img = preview_img

        blobs = blob_img.find_blobs(
            [active_threshold],
            pixels_threshold=8,
            area_threshold=8,
            merge=True,
        ) or []

        try:
            working = preview_img.copy()
            working.binary([active_threshold])
        except Exception:
            working = blob_img

        for blob in blobs:
            working.draw_rectangle(blob.rect(), color=(0, 255, 0), thickness=1)
            working.draw_cross(blob.cx(), blob.cy(), color=(255, 0, 0), size=6, thickness=1)

        try:
            self.canvas.draw_image(working, 12, 72)
        except Exception:
            pass

        self.canvas.draw_rectangle(10, 70, FRAME_WIDTH + 4, FRAME_HEIGHT + 4, color=(255, 255, 255), thickness=2)
        draw_text(self.canvas, 16, 42, "THRESH PREVIEW", color=(255, 255, 255), scale=1)
        draw_text(self.canvas, 16, 348, "white = hit by threshold", color=(190, 190, 190), scale=1)
        draw_text(self.canvas, 16, 326, "blob count: {}".format(len(blobs)), color=(180, 255, 180), scale=1)

    def _draw_threshold_panel(self):
        active_name = TARGET_NAMES[self.active_target_index]
        threshold = self._current_threshold()

        draw_text(self.canvas, 352, 66, "ACTIVE: " + active_name, color=(255, 255, 0), scale=2)

        base_y = 82
        row_h = 56
        for index in range(6):
            y = base_y + index * row_h
            if index == self.active_param_index:
                self.canvas.draw_rectangle(344, y - 2, 430, 46, color=(60, 90, 120), fill=True)
                self.canvas.draw_rectangle(344, y - 2, 430, 46, color=(255, 255, 0), thickness=2)
            draw_text(self.canvas, 352, y + 10, LAB_LABELS[index], color=(255, 255, 255), scale=1)
            draw_text(self.canvas, 505, y + 10, str(threshold[index]), color=(255, 255, 0), scale=1)
        draw_text(self.canvas, 352, 420, "KEY1 short: switch target", color=(190, 190, 190), scale=1)
        draw_text(self.canvas, 352, 440, "KEY1 long : reset target", color=(190, 190, 190), scale=1)
        draw_text(self.canvas, 352, 460, "KEY2 short: next param  KEY2 long: save", color=(190, 190, 190), scale=1)
        draw_text(self.canvas, 352, 400, "KEY3/4: +/-2  hold=fast", color=(190, 190, 190), scale=1)

    def _draw_status(self):
        status = self._status_line()
        self.canvas.draw_rectangle(10, 370, 320, 90, color=(32, 32, 32), fill=True)
        self.canvas.draw_rectangle(10, 370, 320, 90, color=(255, 255, 255), thickness=1)
        draw_text(self.canvas, 18, 382, "BUILD: " + BUILD_TAG, color=(180, 180, 255), scale=1)
        draw_text(self.canvas, 18, 406, "keys ready: {}".format(self.keys.is_ready()), color=(255, 255, 255), scale=1)
        if self.last_event_text:
            draw_text(self.canvas, 18, 430, self.last_event_text[:36], color=(160, 255, 255), scale=1)
        if self.keys._error:
            draw_text(self.canvas, 18, 452, str(self.keys._error)[:36], color=(255, 128, 128), scale=1)
        elif status:
            draw_text(self.canvas, 18, 452, status, color=(255, 255, 0), scale=1)

    def render(self, preview_img):
        self._fill_canvas()
        self._draw_preview(preview_img)
        self._draw_threshold_panel()
        self._draw_status()
        return self.canvas


def main():
    print("=" * 50)
    print("K230 offline key calibration")
    print("build:", BUILD_TAG)
    print("=" * 50)

    display_init_board()
    calibrator = ThresholdCalibrator()
    kw = dict(camera_id=CAMERA_ID, width=FRAME_WIDTH, height=FRAME_HEIGHT,
              hmirror=SENSOR_HMIRROR, vflip=SENSOR_VFLIP)
    sensor = camera_init(CAMERA_ID)
    camera_start(sensor, **kw)

    try:
        while True:
            os.exitpoint()
            calibrator.handle_keys()
            try:
                img = camera_snapshot(sensor)
            except RuntimeError as e:
                print("[Sensor] snapshot failed:", e)
                gc.collect()
                time.sleep_ms(SNAPSHOT_RETRY_DELAY_MS)
                continue
            canvas = calibrator.render(img)
            Display.show_image(canvas)
            time.sleep_ms(20)
    except KeyboardInterrupt:
        print("\n[Calib] interrupted")
    except Exception as e:
        print("[Calib] error:", e)
        sys.print_exception(e)
    finally:
        camera_deinit(sensor)
        print("[Calib] stopped")


if __name__ == "__main__":
    main()
