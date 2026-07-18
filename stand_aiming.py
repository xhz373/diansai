import math
import os
import sys
import time

from common_hw import (DebouncedButton as StartButton, Display, draw_text,
                        camera_init, camera_start, camera_snapshot,
                        camera_restart, camera_deinit,
                        display_init)
from mode_runtime import LoopStatsMixin, load_motion_support
from vision_utils import RectTrackingMixin

build_stepper_controller, _ = load_motion_support()


CAMERA_ID = 2
FRAME_WIDTH = 400
FRAME_HEIGHT = 300
SENSOR_HMIRROR = True
SENSOR_VFLIP = True

START_BUTTON_BOARD_PIN = 28
START_BUTTON_GPIO_NUM = 28

RECT_THRESHOLD = 8000
RECT_BINARY_THRESHOLD = (0, 72)
TARGET_WIDTH_CM = 25.0
TARGET_HEIGHT_CM = 29.7
TARGET_ASPECT = TARGET_WIDTH_CM / TARGET_HEIGHT_CM
TARGET_ASPECT_PENALTY_SCALE = 12000
TARGET_MIN_W = 44
TARGET_MIN_H = 44
TARGET_MIN_AREA = 3600
TARGET_CENTER_ALPHA = 0.90
TARGET_RESET_DIST_PX = 144
TARGET_STICKY_DIST_PX = 1
TARGET_LEAD_GAIN = 0.12
TARGET_LEAD_MAX_PX = 12
TARGET_MAX_JUMP_PX = 96
TARGET_MAX_SIZE_CHANGE_RATIO = 0.35
TARGET_EDGE_MARGIN_PX = 4
TARGET_EDGE_COMP_MIN_RATIO = 0.55
TARGET_MIN_OVERLAP_RATIO = 0.18
TARGET_INIT_CENTER_BIAS = 14
TARGET_NEAR_CENTER_PX = 180
TARGET_BORDER_SAMPLE_COUNT = 10
TARGET_BORDER_HIT_RATIO_MIN = 0.32
TARGET_BORDER_SCORE_SCALE = 9000
TARGET_MAX_MISS_FRAMES = 6
TARGET_DETECT_INTERVAL = 1

ALIGNED_TOLERANCE_PX = 6
# Target point is constrained to the same vertical line as screen center.
# Positive Z means the target point is below screen center; negative means above.
TARGET_POINT_OFFSET_Z = -40

DEBUG_MODE = True
DEBUG_TEXT_OVERLAY = False
FRAME_LOOP_DELAY_MS = 0
GC_INTERVAL = 180
MAX_CONSECUTIVE_SNAPSHOT_FAILURES = 5

BUILD_TAG = "2026-07-14-rect-center-v1"

STEPPER_AXIS_OVERRIDES = {
    "x": {
        "deadband": float(ALIGNED_TOLERANCE_PX),
        "error_full_scale": 100.0,
        "command_sign": 1,
        "pid_kp": 16.0,
        "pid_ki": 0.25,
        "pid_kd": 0.9,
        "integral_limit": 120.0,
        "integral_active_error": 36.0,
    },
    "y": {
        "deadband": float(ALIGNED_TOLERANCE_PX),
        "error_full_scale": 80.0,
        "command_sign": 1,
        "pid_kp": 16.0,
        "pid_ki": 0.25,
        "pid_kd": 0.9,
        "integral_limit": 120.0,
        "integral_active_error": 36.0,
    },
}


class RectTracker(RectTrackingMixin):
    frame_width = FRAME_WIDTH
    frame_height = FRAME_HEIGHT
    rect_binary_threshold = RECT_BINARY_THRESHOLD
    target_aspect = TARGET_ASPECT
    target_aspect_penalty_scale = TARGET_ASPECT_PENALTY_SCALE
    target_min_w = TARGET_MIN_W
    target_min_h = TARGET_MIN_H
    target_min_area = TARGET_MIN_AREA
    target_center_alpha = TARGET_CENTER_ALPHA
    target_reset_dist_px = TARGET_RESET_DIST_PX
    target_sticky_dist_px = TARGET_STICKY_DIST_PX
    target_lead_gain = TARGET_LEAD_GAIN
    target_lead_max_px = TARGET_LEAD_MAX_PX
    target_max_jump_px = TARGET_MAX_JUMP_PX
    target_max_size_change_ratio = TARGET_MAX_SIZE_CHANGE_RATIO
    target_edge_margin_px = TARGET_EDGE_MARGIN_PX
    target_edge_comp_min_ratio = TARGET_EDGE_COMP_MIN_RATIO
    target_min_overlap_ratio = TARGET_MIN_OVERLAP_RATIO
    target_init_center_bias = TARGET_INIT_CENTER_BIAS
    target_near_center_px = TARGET_NEAR_CENTER_PX
    target_border_sample_count = TARGET_BORDER_SAMPLE_COUNT
    target_border_hit_ratio_min = TARGET_BORDER_HIT_RATIO_MIN
    target_border_score_scale = TARGET_BORDER_SCORE_SCALE

    def __init__(self):
        self.frame_id = 0
        self.target_rect = None
        self.target_center = None
        self.target_found = False
        self.target_miss_count = 0
        self.last_target_rect = None
        self.last_target_center = None

    def detect(self, img):
        self.frame_id += 1
        if self.target_found and (self.frame_id % TARGET_DETECT_INTERVAL) != 0:
            return self.target_found, self.target_rect, self.target_center

        previous_rect = self.last_target_rect
        previous_center = self.last_target_center
        self.target_rect = None
        self.target_center = None
        self.target_found = False

        rect_img = self._prepare_rect_image(img)
        rects = rect_img.find_rects(threshold=RECT_THRESHOLD) or []
        chosen = self._select_best_rect(rect_img, rects, previous_center, previous_rect)
        if chosen is not None:
            _, _, center = chosen
            if not self._accept_center(center, previous_center):
                chosen = None

        if chosen is None:
            self.target_miss_count += 1
            if previous_rect and previous_center and self.target_miss_count <= TARGET_MAX_MISS_FRAMES:
                self.target_rect = previous_rect
                self.target_center = previous_center
                self.target_found = True
            else:
                self.last_target_rect = None
                self.last_target_center = None
            return self.target_found, self.target_rect, self.target_center

        self.target_miss_count = 0
        rect, _, center = chosen
        self.target_rect = rect
        self.target_center = self._smooth_center(
            center,
            previous_center,
            TARGET_CENTER_ALPHA,
            TARGET_RESET_DIST_PX,
            TARGET_STICKY_DIST_PX,
        )
        self.target_center = self._apply_motion_lead(
            self.target_center,
            previous_center,
            TARGET_LEAD_GAIN,
            TARGET_LEAD_MAX_PX,
        )
        self.target_found = True
        self.last_target_rect = self.target_rect
        self.last_target_center = self.target_center
        return self.target_found, self.target_rect, self.target_center


class RectCenterSystem(LoopStatsMixin):
    def __init__(self):
        self.tracker = RectTracker()
        self.motor = build_stepper_controller(STEPPER_AXIS_OVERRIDES)
        self.control_started = False
        self.start_button = StartButton(START_BUTTON_BOARD_PIN, START_BUTTON_GPIO_NUM)
        self._init_loop_stats(enable_fps=True)
        self.last_aligned = False
        self._aligned_latched = False

    def _update_start_button(self):
        if self.control_started:
            return
        if self.start_button.poll_pressed():
            self.control_started = True
            print("[Motor] start button pressed, stepper control enabled")

    def process_frame(self, img):
        self._mark_frame()
        self._update_start_button()

        found, rect, center = self.tracker.detect(img)
        screen_center = (FRAME_WIDTH // 2, FRAME_HEIGHT // 2)
        target_point = (screen_center[0], screen_center[1] + TARGET_POINT_OFFSET_Z)

        if not found or rect is None or center is None:
            self.last_aligned = False
            self._aligned_latched = False
            self.motor.stop()
            if DEBUG_MODE:
                self._draw_overlay(img, None, None, screen_center, target_point, False)
            return img

        dx = center[0] - target_point[0]
        dy = center[1] - target_point[1]
        aligned = abs(dx) <= ALIGNED_TOLERANCE_PX and abs(dy) <= ALIGNED_TOLERANCE_PX
        self.last_aligned = aligned
        self.motor.drive(dx, dy, allow_drive=self.control_started and (not aligned))
        self._aligned_latched = aligned

        if DEBUG_MODE:
            self._draw_overlay(img, rect, center, screen_center, target_point, aligned)
        return img

    def _draw_overlay(self, img, rect, center, screen_center, target_point, aligned):
        scx, scy = screen_center
        tx, ty = target_point
        img.draw_cross(scx, scy, color=(120, 120, 120), size=8, thickness=1)
        img.draw_line(scx, 0, scx, FRAME_HEIGHT - 1, color=(80, 80, 80), thickness=1)
        img.draw_cross(tx, ty, color=(255, 255, 0), size=10, thickness=2)
        img.draw_circle(tx, ty, ALIGNED_TOLERANCE_PX, color=(255, 255, 0), thickness=1)

        if rect is not None:
            x, y, w, h = rect
            color = (0, 255, 0) if aligned else (0, 180, 255)
            img.draw_rectangle(x, y, w, h, color=color, thickness=2)
        if center is not None:
            cx, cy = center
            color = (0, 255, 0) if aligned else (255, 0, 0)
            img.draw_cross(cx, cy, color=color, size=8, thickness=2)
            img.draw_line(cx, cy, tx, ty, color=(255, 255, 255), thickness=1)

        if DEBUG_TEXT_OVERLAY and center is not None:
            draw_text(img, 4, 4, "dx={} dy={}".format(center[0] - tx, center[1] - ty))
            draw_text(img, 4, 22, "fps={:.1f}".format(self.fps))
            draw_text(img, 4, 40, "aligned={}".format(1 if aligned else 0))
            draw_text(img, 4, 58, "z={}".format(TARGET_POINT_OFFSET_Z))
        if not self.control_started:
            draw_text(img, 4, FRAME_HEIGHT - 18, "PRESS GPIO28 TO START MOTOR", color=(255, 255, 0), scale=1)

def main():
    print("=" * 50)
    print("K230 Rect Center Mode")
    print("build:", BUILD_TAG)
    print("fast mode enabled")
    print("=" * 50)

    system = RectCenterSystem()

    print("[Display] init...")
    display_init(FRAME_WIDTH, FRAME_HEIGHT)
    kw = dict(camera_id=CAMERA_ID, width=FRAME_WIDTH, height=FRAME_HEIGHT,
              hmirror=SENSOR_HMIRROR, vflip=SENSOR_VFLIP)
    try:
        print("[Sensor] init...")
        sensor = camera_init(CAMERA_ID)
        camera_start(sensor, **kw)
    except Exception as e:
        print("[Sensor] start failed, retry by restart:", e)
        sensor = camera_restart(None, **kw)

    print("[System] ready")

    consecutive_snapshot_failures = 0
    try:
        while True:
            os.exitpoint()
            try:
                img = camera_snapshot(sensor)
                consecutive_snapshot_failures = 0
            except RuntimeError as e:
                consecutive_snapshot_failures += 1
                if consecutive_snapshot_failures >= MAX_CONSECUTIVE_SNAPSHOT_FAILURES:
                    print("[Sensor] snapshot failed repeatedly, restart:", e)
                    sensor = camera_restart(sensor, **kw)
                    consecutive_snapshot_failures = 0
                time.sleep_ms(FRAME_LOOP_DELAY_MS)
                continue

            img = system.process_frame(img)
            system.update_fps()
            Display.show_image(img)
            system.maybe_collect_gc(GC_INTERVAL)
            time.sleep_ms(FRAME_LOOP_DELAY_MS)
    except KeyboardInterrupt:
        print("\n[System] interrupted")
    except Exception as e:
        print("[Error]", e)
        sys.print_exception(e)
    finally:
        print("[System] cleanup...")
        camera_deinit(sensor)
        system.motor.deinit()
        print("[System] stopped")


if __name__ == "__main__":
    main()
