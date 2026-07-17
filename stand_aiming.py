from media.sensor import *
from media.display import *
from media.media import *
import gc
import math
import os
import sys
import time
from machine import FPIOA, Pin

try:
    from control_backends import BACKEND_LOCAL
    from control_backends import build_control_backend
    from control_backends import mode_code_for_name
    from control_backends import unit_code_for_name
    from dual_core_config import CONTROL_BACKEND
except ImportError:
    BACKEND_LOCAL = "local"
    CONTROL_BACKEND = BACKEND_LOCAL

    def build_control_backend(backend_name, axis_overrides=None, mode_code=0, unit_code=0):
        class _NoopControlBackend:
            ready = False

            def update(
                self,
                error_x,
                error_y,
                valid,
                control_enabled,
                target_x=None,
                target_y=None,
                sync_ok=True,
                aligned=False,
                state_name="IDLE",
            ):
                return

            def stop(self, state_name="STOPPED"):
                return

            def deinit(self):
                return

        return _NoopControlBackend()

    def mode_code_for_name(name):
        return 0

    def unit_code_for_name(name):
        return 0


CAMERA_ID = 2
FRAME_WIDTH = 400
FRAME_HEIGHT = 300
SENSOR_HMIRROR = True
SENSOR_VFLIP = True

START_BUTTON_BOARD_PIN = 28
START_BUTTON_GPIO_NUM = 28
BUTTON_DEBOUNCE_MS = 35

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

SNAPSHOT_RETRY_COUNT = 3
SNAPSHOT_RETRY_DELAY_MS = 3
SENSOR_WARMUP_FRAMES = 2
START_CAMERA_RETRY_COUNT = 2
START_CAMERA_RETRY_DELAY_MS = 10
START_CAMERA_SETTLE_MS = 28
START_CAMERA_SETTLE_STEP_MS = 18
ALLOW_CHANNEL_FALLBACK = False
MAX_CONSECUTIVE_SNAPSHOT_FAILURES = 5

BUILD_TAG = "2026-07-14-rect-center-v1"
CURRENT_SNAPSHOT_CHN = CAM_CHN_ID_1

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


def draw_text(img, x, y, text, color=(255, 255, 255), scale=1):
    text = str(text)
    if hasattr(img, "draw_string_advanced"):
        img.draw_string_advanced(x, y, max(16, 16 * scale), text, color=color)
    else:
        img.draw_string(x, y, text, color=color, scale=scale)


def _pin_pull_up_value():
    for name in ("PULL_UP", "PULLUP", "PULL_UP_ENABLE"):
        value = getattr(Pin, name, None)
        if value is not None:
            return value
    return None


def _map_board_pin_to_gpio(fpioa, board_pin, gpio_num):
    if not hasattr(fpioa, "set_function"):
        return
    for func_name in (
        "GPIO{}_FUNC".format(gpio_num),
        "GPIO{}".format(gpio_num),
        "GPIOHS{}".format(gpio_num),
        "GPIOHS{}_FUNC".format(gpio_num),
    ):
        func = getattr(fpioa, func_name, None)
        if func is not None:
            fpioa.set_function(board_pin, func)
            return


class StartButton:
    def __init__(self, board_pin, gpio_num):
        self.pin = None
        self.ready = False
        self.latched = False
        self.last_raw_value = 1
        self.stable_value = 1
        self.last_change_ms = 0
        self._init_pin(board_pin, gpio_num)

    def _init_pin(self, board_pin, gpio_num):
        try:
            fpioa = FPIOA()
            _map_board_pin_to_gpio(fpioa, board_pin, gpio_num)
            pull_up = _pin_pull_up_value()
            if pull_up is None:
                try:
                    self.pin = Pin(gpio_num, Pin.IN)
                except Exception:
                    self.pin = Pin(board_pin, Pin.IN)
            else:
                try:
                    self.pin = Pin(gpio_num, Pin.IN, pull_up)
                except Exception:
                    self.pin = Pin(board_pin, Pin.IN, pull_up)
            self.ready = self.pin is not None
        except Exception as e:
            print("[Key] start init failed:", e)
            self.pin = None
            self.ready = False

    def poll_pressed(self):
        if self.latched or self.pin is None:
            return self.latched
        now = time.ticks_ms()
        try:
            raw_value = self.pin.value()
        except Exception:
            return False
        if raw_value != self.last_raw_value:
            self.last_raw_value = raw_value
            self.last_change_ms = now
            return False
        if time.ticks_diff(now, self.last_change_ms) < BUTTON_DEBOUNCE_MS:
            return False
        if raw_value != self.stable_value:
            self.stable_value = raw_value
            if self.stable_value == 0:
                self.latched = True
                return True
        return False


def _snapshot_channel_name(snapshot_chn):
    if snapshot_chn == CAM_CHN_ID_1:
        return "chn1"
    return "chn0"


def _configure_sensor_for_channel(sensor, snapshot_chn):
    sensor.reset()
    try:
        sensor.set_hmirror(SENSOR_HMIRROR)
    except Exception:
        pass
    try:
        sensor.set_vflip(SENSOR_VFLIP)
    except Exception:
        pass
    if snapshot_chn == CAM_CHN_ID_1:
        sensor.set_framesize(Sensor.FHD)
        sensor.set_pixformat(Sensor.YUV420SP)
        sensor.set_framesize(width=FRAME_WIDTH, height=FRAME_HEIGHT, chn=CAM_CHN_ID_1)
        sensor.set_pixformat(Sensor.RGB565, chn=CAM_CHN_ID_1)
    else:
        sensor.set_framesize(width=FRAME_WIDTH, height=FRAME_HEIGHT)
        sensor.set_pixformat(Sensor.RGB565)


def _snapshot_once(sensor, snapshot_chn):
    if snapshot_chn == CAM_CHN_ID_1:
        return sensor.snapshot(chn=CAM_CHN_ID_1)
    return sensor.snapshot(chn=CAM_CHN_ID_0)


def init_camera_sensor(camera_id=None):
    if camera_id is None:
        camera_id = CAMERA_ID
    try:
        sensor = Sensor(id=camera_id)
    except OSError as e:
        if "already inited" not in str(e):
            raise
        Sensor.deinit()
        time.sleep_ms(20)
        sensor = Sensor(id=camera_id)
    return sensor


def _start_camera_on_channel(sensor, camera_id, snapshot_chn):
    if camera_id is None:
        camera_id = CAMERA_ID
    last_error = None
    for attempt in range(START_CAMERA_RETRY_COUNT):
        os.exitpoint()
        try:
            MediaManager.init()
            sensor.run()
            settle_ms = START_CAMERA_SETTLE_MS + attempt * START_CAMERA_SETTLE_STEP_MS
            time.sleep_ms(settle_ms)
            for _ in range(SENSOR_WARMUP_FRAMES):
                os.exitpoint()
                try:
                    _snapshot_once(sensor, snapshot_chn)
                    return True
                except Exception as e:
                    last_error = e
                    time.sleep_ms(SNAPSHOT_RETRY_DELAY_MS)
            last_error = RuntimeError(
                "camera id {} no warmup frames on {}".format(
                    camera_id, _snapshot_channel_name(snapshot_chn)
                )
            )
        except Exception as e:
            last_error = e

        try:
            sensor.stop()
        except Exception:
            pass
        try:
            MediaManager.deinit()
        except Exception:
            pass
        try:
            _configure_sensor_for_channel(sensor, snapshot_chn)
        except Exception:
            pass
        gc.collect()
        time.sleep_ms(START_CAMERA_RETRY_DELAY_MS)

    raise last_error


def start_camera(sensor, camera_id=None):
    global CURRENT_SNAPSHOT_CHN
    if camera_id is None:
        camera_id = CAMERA_ID
    last_error = None
    channels = (CAM_CHN_ID_1,)
    if ALLOW_CHANNEL_FALLBACK:
        channels = (CAM_CHN_ID_1, CAM_CHN_ID_0)
    for snapshot_chn in channels:
        os.exitpoint()
        try:
            _configure_sensor_for_channel(sensor, snapshot_chn)
            _start_camera_on_channel(sensor, camera_id, snapshot_chn)
            CURRENT_SNAPSHOT_CHN = snapshot_chn
            return True
        except Exception as e:
            last_error = e
            try:
                sensor.stop()
            except Exception:
                pass
            try:
                MediaManager.deinit()
            except Exception:
                pass
            gc.collect()
            time.sleep_ms(START_CAMERA_RETRY_DELAY_MS)
    raise last_error


def snapshot_with_retry(sensor):
    last_error = None
    for _ in range(SNAPSHOT_RETRY_COUNT):
        os.exitpoint()
        try:
            return _snapshot_once(sensor, CURRENT_SNAPSHOT_CHN)
        except RuntimeError as e:
            last_error = e
            time.sleep_ms(SNAPSHOT_RETRY_DELAY_MS)
    raise last_error


def init_preview_display():
    try:
        Display.init(
            Display.VIRT, width=FRAME_WIDTH, height=FRAME_HEIGHT, fps=100, to_ide=True
        )
        print("[Display] VIRT preview")
    except Exception as e:
        print("[Display] VIRT failed:", e)
        Display.init(Display.ST7701, to_ide=True)
        print("[Display] ST7701 preview")


def restart_camera(sensor):
    os.exitpoint()
    print("[Sensor] restarting...")
    try:
        if isinstance(sensor, Sensor):
            sensor.stop()
    except Exception:
        pass
    try:
        MediaManager.deinit()
    except Exception:
        pass
    try:
        Sensor.deinit()
    except Exception:
        pass
    gc.collect()
    time.sleep_ms(20)
    sensor = init_camera_sensor()
    start_camera(sensor)
    print("[Sensor] restart done")
    return sensor


class RectTracker:
    def __init__(self):
        self.frame_id = 0
        self.target_rect = None
        self.target_center = None
        self.target_found = False
        self.target_miss_count = 0
        self.last_target_rect = None
        self.last_target_center = None

    def _distance_sq(self, p0, p1):
        dx = p0[0] - p1[0]
        dy = p0[1] - p1[1]
        return dx * dx + dy * dy

    def _clamp_rect(self, x, y, w, h):
        x = max(0, min(FRAME_WIDTH - 1, int(x)))
        y = max(0, min(FRAME_HEIGHT - 1, int(y)))
        w = max(1, min(FRAME_WIDTH - x, int(w)))
        h = max(1, min(FRAME_HEIGHT - y, int(h)))
        return (x, y, w, h)

    def _clamp_point(self, point):
        return (
            max(0, min(FRAME_WIDTH - 1, int(point[0]))),
            max(0, min(FRAME_HEIGHT - 1, int(point[1]))),
        )

    def _smooth_center(self, current, last_center, alpha, reset_px, sticky_px):
        if last_center is None:
            return current
        dist_sq = self._distance_sq(current, last_center)
        if dist_sq <= (sticky_px * sticky_px):
            return last_center
        if dist_sq > (reset_px * reset_px):
            return current
        return (
            int(last_center[0] * (1 - alpha) + current[0] * alpha),
            int(last_center[1] * (1 - alpha) + current[1] * alpha),
        )

    def _apply_motion_lead(self, current, last_center, gain, max_px):
        if current is None or last_center is None or gain <= 0:
            return current
        dx = current[0] - last_center[0]
        dy = current[1] - last_center[1]
        lead_x = int(dx * gain)
        lead_y = int(dy * gain)
        if lead_x > max_px:
            lead_x = max_px
        elif lead_x < -max_px:
            lead_x = -max_px
        if lead_y > max_px:
            lead_y = max_px
        elif lead_y < -max_px:
            lead_y = -max_px
        return self._clamp_point((current[0] + lead_x, current[1] + lead_y))

    def _rect_aspect_error(self, w, h):
        aspect = w / max(h, 1)
        target_inv = 1.0 / TARGET_ASPECT
        return min(abs(aspect - TARGET_ASPECT), abs(aspect - target_inv))

    def _rect_center_from_corners(self, corners):
        sx = 0
        sy = 0
        points = []
        for p in corners:
            sx += p[0]
            sy += p[1]
            points.append((p[0], p[1]))
        avg_center = (sx / 4, sy / 4)

        points.sort(key=lambda p: math.atan2(p[1] - avg_center[1], p[0] - avg_center[0]))
        p0 = points[0]
        p1 = points[1]
        p2 = points[2]
        p3 = points[3]

        x1 = p0[0]
        y1 = p0[1]
        x2 = p2[0]
        y2 = p2[1]
        x3 = p1[0]
        y3 = p1[1]
        x4 = p3[0]
        y4 = p3[1]

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1:
            return self._clamp_point(avg_center)

        det1 = x1 * y2 - y1 * x2
        det2 = x3 * y4 - y3 * x4
        center_x = (det1 * (x3 - x4) - (x1 - x2) * det2) / denom
        center_y = (det1 * (y3 - y4) - (y1 - y2) * det2) / denom
        if center_x < -8 or center_x > (FRAME_WIDTH + 8) or center_y < -8 or center_y > (FRAME_HEIGHT + 8):
            return self._clamp_point(avg_center)
        return self._clamp_point((center_x, center_y))

    def _rect_size_change_ok(self, rect, last_rect):
        if last_rect is None:
            return True
        _, _, w, h = rect
        _, _, last_w, last_h = last_rect
        if last_w <= 0 or last_h <= 0:
            return True
        dw = abs(w - last_w) / last_w
        dh = abs(h - last_h) / last_h
        return dw <= TARGET_MAX_SIZE_CHANGE_RATIO and dh <= TARGET_MAX_SIZE_CHANGE_RATIO

    def _compensate_edge_rect(self, rect, last_rect):
        if last_rect is None:
            return rect
        x, y, w, h = rect
        _, _, last_w, last_h = last_rect
        if last_w <= 0 or last_h <= 0:
            return rect

        x2 = x + w
        y2 = y + h
        if x <= TARGET_EDGE_MARGIN_PX and w < int(last_w * TARGET_EDGE_COMP_MIN_RATIO):
            w = last_w
        elif x2 >= (FRAME_WIDTH - TARGET_EDGE_MARGIN_PX) and w < int(last_w * TARGET_EDGE_COMP_MIN_RATIO):
            x = max(0, x2 - last_w)
            w = FRAME_WIDTH - x if x + last_w > FRAME_WIDTH else last_w

        if y <= TARGET_EDGE_MARGIN_PX and h < int(last_h * TARGET_EDGE_COMP_MIN_RATIO):
            h = last_h
        elif y2 >= (FRAME_HEIGHT - TARGET_EDGE_MARGIN_PX) and h < int(last_h * TARGET_EDGE_COMP_MIN_RATIO):
            y = max(0, y2 - last_h)
            h = FRAME_HEIGHT - y if y + last_h > FRAME_HEIGHT else last_h

        return self._clamp_rect(x, y, w, h)

    def _rect_overlap_ratio(self, rect_a, rect_b):
        if rect_a is None or rect_b is None:
            return 0.0
        ax, ay, aw, ah = rect_a
        bx, by, bw, bh = rect_b
        ax2 = ax + aw
        ay2 = ay + ah
        bx2 = bx + bw
        by2 = by + bh
        ix1 = max(ax, bx)
        iy1 = max(ay, by)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = ix2 - ix1
        ih = iy2 - iy1
        if iw <= 0 or ih <= 0:
            return 0.0
        inter = iw * ih
        min_area = min(max(1, aw * ah), max(1, bw * bh))
        return inter / min_area

    def _rect_border_hit_ratio(self, rect_img, rect):
        x, y, w, h = rect
        if w <= 4 or h <= 4:
            return 0.0
        inset = max(1, min(6, min(w, h) // 12))
        x1 = x + inset
        y1 = y + inset
        x2 = x + w - 1 - inset
        y2 = y + h - 1 - inset
        if x2 <= x1 or y2 <= y1:
            return 0.0

        hits = 0
        total = 0
        steps = max(4, TARGET_BORDER_SAMPLE_COUNT)
        for idx in range(steps):
            t = idx / max(1, steps - 1)
            sx = int(x1 + (x2 - x1) * t)
            sy = int(y1 + (y2 - y1) * t)
            points = ((sx, y1), (sx, y2), (x1, sy), (x2, sy))
            for px, py in points:
                total += 1
                try:
                    if rect_img.get_pixel(px, py):
                        hits += 1
                except Exception:
                    pass
        if total <= 0:
            return 0.0
        return hits / total

    def _accept_center(self, candidate_center, last_center):
        if candidate_center is None or last_center is None:
            return True
        return self._distance_sq(candidate_center, last_center) <= (TARGET_MAX_JUMP_PX * TARGET_MAX_JUMP_PX)

    def _prepare_rect_image(self, img):
        rect_img = img.to_grayscale()
        rect_img.binary([RECT_BINARY_THRESHOLD])
        return rect_img

    def _select_best_rect(self, rect_img, rects, reference_center, reference_rect):
        best = None
        best_score = None
        image_center = (FRAME_WIDTH // 2, FRAME_HEIGHT // 2)

        for r in rects:
            raw_rect = r.rect()
            corners = r.corners()
            if raw_rect is None or corners is None or len(corners) != 4:
                continue

            rect = self._compensate_edge_rect(raw_rect, reference_rect)
            x, y, w, h = rect
            if w < TARGET_MIN_W or h < TARGET_MIN_H:
                continue
            area = w * h
            if area < TARGET_MIN_AREA:
                continue
            if not self._rect_size_change_ok(rect, reference_rect):
                continue

            center = self._rect_center_from_corners(corners)
            border_hit_ratio = self._rect_border_hit_ratio(rect_img, rect)
            if border_hit_ratio < TARGET_BORDER_HIT_RATIO_MIN:
                continue
            if reference_center is not None:
                jump_sq = self._distance_sq(center, reference_center)
                if jump_sq > (TARGET_RESET_DIST_PX * TARGET_RESET_DIST_PX):
                    continue
            if reference_rect is not None:
                overlap_ratio = self._rect_overlap_ratio(rect, reference_rect)
                if overlap_ratio < TARGET_MIN_OVERLAP_RATIO and (
                    reference_center is None
                    or self._distance_sq(center, reference_center) > (TARGET_STICKY_DIST_PX * TARGET_STICKY_DIST_PX)
                ):
                    continue

            aspect_penalty = int(self._rect_aspect_error(w, h) * TARGET_ASPECT_PENALTY_SCALE)
            if reference_center is not None:
                distance_penalty = self._distance_sq(center, reference_center) // 10
            else:
                distance_penalty = self._distance_sq(center, image_center) // TARGET_INIT_CENTER_BIAS
            edge_penalty = 0
            if x <= 2 or y <= 2 or (x + w) >= (FRAME_WIDTH - 2) or (y + h) >= (FRAME_HEIGHT - 2):
                edge_penalty = 3600
            center_bias_bonus = 0
            if self._distance_sq(center, image_center) <= (TARGET_NEAR_CENTER_PX * TARGET_NEAR_CENTER_PX):
                center_bias_bonus = 2000
            border_score_bonus = int(border_hit_ratio * TARGET_BORDER_SCORE_SCALE)

            score = area - aspect_penalty - distance_penalty - edge_penalty + center_bias_bonus + border_score_bonus
            if best_score is None or score > best_score:
                best_score = score
                best = (rect, center)

        return best

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
            _, center = chosen
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
        rect, center = chosen
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


class RectCenterSystem:
    def __init__(self):
        self.tracker = RectTracker()
        self.control = build_control_backend(
            CONTROL_BACKEND,
            axis_overrides=STEPPER_AXIS_OVERRIDES,
            mode_code=mode_code_for_name("stand"),
            unit_code=unit_code_for_name("pixel"),
        )
        self.control_started = False
        self.start_button = StartButton(START_BUTTON_BOARD_PIN, START_BUTTON_GPIO_NUM)
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_time = time.ticks_ms()
        self.gc_counter = 0
        self.last_aligned = False
        self._aligned_latched = False

    def _update_start_button(self):
        if self.control_started:
            return
        if self.start_button.poll_pressed():
            self.control_started = True
            print("[Motor] start button pressed, stepper control enabled")

    def process_frame(self, img):
        self.frame_count += 1
        self.gc_counter += 1
        self._update_start_button()

        found, rect, center = self.tracker.detect(img)
        screen_center = (FRAME_WIDTH // 2, FRAME_HEIGHT // 2)
        target_point = (screen_center[0], screen_center[1] + TARGET_POINT_OFFSET_Z)

        if not found or rect is None or center is None:
            self.last_aligned = False
            self._aligned_latched = False
            self.control.update(
                error_x=None,
                error_y=None,
                valid=False,
                control_enabled=self.control_started,
                aligned=False,
                state_name="STOPPED",
            )
            if DEBUG_MODE:
                self._draw_overlay(img, None, None, screen_center, target_point, False)
            return img

        dx = center[0] - target_point[0]
        dy = center[1] - target_point[1]
        aligned = abs(dx) <= ALIGNED_TOLERANCE_PX and abs(dy) <= ALIGNED_TOLERANCE_PX
        self.last_aligned = aligned
        self.control.update(
            error_x=dx,
            error_y=dy,
            valid=True,
            control_enabled=self.control_started,
            aligned=aligned,
            state_name="TRACKING",
        )
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

    def update_fps(self):
        current_time = time.ticks_ms()
        dt = time.ticks_diff(current_time, self.last_fps_time)
        if dt >= 1000:
            self.fps = self.frame_count * 1000 / dt
            self.frame_count = 0
            self.last_fps_time = current_time

    def maybe_collect_gc(self):
        if self.gc_counter >= GC_INTERVAL:
            gc.collect()
            self.gc_counter = 0


def main():
    print("=" * 50)
    print("K230 Rect Center Mode")
    print("build:", BUILD_TAG)
    print("fast mode enabled")
    print("=" * 50)

    system = RectCenterSystem()

    print("[Display] init...")
    init_preview_display()
    try:
        print("[Sensor] init...")
        sensor = init_camera_sensor()
        start_camera(sensor)
    except Exception as e:
        print("[Sensor] start failed, retry by restart:", e)
        sensor = restart_camera(None)

    print("[System] ready")

    consecutive_snapshot_failures = 0
    try:
        while True:
            os.exitpoint()
            try:
                img = snapshot_with_retry(sensor)
                consecutive_snapshot_failures = 0
            except RuntimeError as e:
                consecutive_snapshot_failures += 1
                if consecutive_snapshot_failures >= MAX_CONSECUTIVE_SNAPSHOT_FAILURES:
                    print("[Sensor] snapshot failed repeatedly, restart:", e)
                    sensor = restart_camera(sensor)
                    consecutive_snapshot_failures = 0
                time.sleep_ms(FRAME_LOOP_DELAY_MS)
                continue

            img = system.process_frame(img)
            system.update_fps()
            Display.show_image(img)
            system.maybe_collect_gc()
            time.sleep_ms(FRAME_LOOP_DELAY_MS)
    except KeyboardInterrupt:
        print("\n[System] interrupted")
    except Exception as e:
        print("[Error]", e)
        sys.print_exception(e)
    finally:
        print("[System] cleanup...")
        if isinstance(sensor, Sensor):
            sensor.stop()
        Display.deinit()
        os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
        time.sleep_ms(100)
        MediaManager.deinit()
        Sensor.deinit()
        system.control.deinit()
        print("[System] stopped")


if __name__ == "__main__":
    main()
