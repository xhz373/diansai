from media.sensor import *
from media.display import *
from media.media import *
import gc
import math
import os
import sys
import time
from machine import FPIOA, Pin, UART

try:
    from k230_common import load_calibration
except ImportError:
    def load_calibration(default_red, default_black, default_violet, default_bright=None):
        return (
            False,
            tuple(default_red),
            tuple(default_black),
            tuple(default_violet),
            default_bright,
        )


CAMERA_ID = 2
FRAME_WIDTH = 416
FRAME_HEIGHT = 234
SENSOR_HMIRROR = True
SENSOR_VFLIP = True

TARGET_WIDTH_CM = 25.0
TARGET_HEIGHT_CM = 29.7
CIRCLE_RADIUS_CM = 6.0
TARGET_OUTER_DIAMETER_CM = 21.0

RED_THRESHOLD = (41, 100, -28, 6, -14, 14)
BLACK_THRESHOLD = (22, 69, -23, -3, -22, 16)
RING_THRESHOLD = (43, 74, -28, -1, -17, 19)
VIOLET_THRESHOLD = (92, 100, -15, 6, -9, 11)

RECT_THRESHOLD = 8000
RECT_BINARY_THRESHOLD = (0, 72)
TARGET_MIN_W = 44
TARGET_MIN_H = 44
TARGET_MIN_AREA = 3600
TARGET_ASPECT = TARGET_WIDTH_CM / TARGET_HEIGHT_CM
TARGET_ASPECT_PENALTY_SCALE = 12000
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
BULLSEYE_ROI_RATIO = 0.42
BULLSEYE_GATE_RATIO = 0.60
BULLSEYE_BLEND_ALPHA = 0.65
BULLSEYE_CENTER_ALPHA = 0.84
BULLSEYE_LEAD_GAIN = 0.18
BULLSEYE_LEAD_MAX_PX = 10

NUM_WAYPOINTS = 360
MAX_SYNC_ERROR_DEG = 90
START_ALIGN_TOL_CM = 1.2
START_ALIGN_HOLD_FRAMES = 3
CIRCLE_SPEED_DEG_PER_SEC = 20.0

PID_KP = 0.8
PID_KI = 0.1
PID_KD = 0.05

UART_ID = 2
TX_PIN = 5
RX_PIN = 6
UART_BAUDRATE = 115200
START_BUTTON_BOARD_PIN = 28
START_BUTTON_GPIO_NUM = 28
BUTTON_DEBOUNCE_MS = 35

DEBUG_MODE = True
DEBUG_TEXT_OVERLAY = False
TARGET_DETECT_INTERVAL = 1
BULLSEYE_DETECT_INTERVAL = 1
RING_DETECT_INTERVAL = 3
PREDICT_TRACK_MAX_FRAMES = 45
TARGET_MAX_MISS_FRAMES = 6
LASER_MAX_MISS_FRAMES = 2
LASER_STICKY_PX = 2
LASER_PREDICT_ROI_MARGIN_PX = 28
CIRCLE_RADIUS_SMOOTHING_ALPHA = 0.45
RING_RADIUS_CLUSTER_TOL_CM = 0.35
RING_RADIUS_MIN_CM = 1.5
RING_RADIUS_MAX_CM = 12.0
RING_TARGET_INDEX_FROM_CENTER = 2
RING_MIN_CLUSTER_SAMPLES = 2
RING_MIN_CLUSTER_WEIGHT = 3.0
RING_BLOB_MIN_SIZE_PX = 2
RING_BLOB_MAX_SIZE_DIV = 8
POINT_HISTORY_LEN = 1
DISPLAY_FPS = 30
FRAME_LOOP_DELAY_MS = 0
SNAPSHOT_RETRY_COUNT = 3
SNAPSHOT_RETRY_DELAY_MS = 3
SENSOR_WARMUP_FRAMES = 2
START_CAMERA_RETRY_COUNT = 2
START_CAMERA_RETRY_DELAY_MS = 10
START_CAMERA_SETTLE_MS = 28
START_CAMERA_SETTLE_STEP_MS = 18
ALLOW_CHANNEL_FALLBACK = False
OVERLAY_RING_SAMPLE_COUNT = 18
GC_FRAME_INTERVAL = 180
BUILD_TAG = "2026-07-14-circle-fastboot-v31"
CURRENT_SNAPSHOT_CHN = CAM_CHN_ID_1
ERROR_JUMP_MAX_CM = 3.0
ERROR_JUMP_REJECT_FRAMES = 2


def apply_calibration():
    global RED_THRESHOLD, BLACK_THRESHOLD, VIOLET_THRESHOLD
    ok, red, black, violet, _ = load_calibration(
        RED_THRESHOLD, BLACK_THRESHOLD, VIOLET_THRESHOLD
    )
    RED_THRESHOLD = red
    BLACK_THRESHOLD = black
    VIOLET_THRESHOLD = violet
    if ok:
        print("[Calib] thresholds applied")
    else:
        print("[Calib] using built-in thresholds")


def _snapshot_channel_name(snapshot_chn):
    if snapshot_chn == CAM_CHN_ID_1:
        return "chn1"
    return "chn0"


def _configure_sensor_for_channel(sensor, camera_id, snapshot_chn):
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
            _configure_sensor_for_channel(sensor, camera_id, snapshot_chn)
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
            _configure_sensor_for_channel(sensor, camera_id, snapshot_chn)
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
            gc.collect()
            time.sleep_ms(SNAPSHOT_RETRY_DELAY_MS)
    raise last_error


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


def init_preview_display():
    try:
        Display.init(
            Display.VIRT, width=FRAME_WIDTH, height=FRAME_HEIGHT, fps=100, to_ide=True
        )
        print("[Display] VIRT preview")
    except Exception as e:
        print(f"[Display] VIRT failed: {e}")
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


class CircleTracker:
    def __init__(self, radius_cm=CIRCLE_RADIUS_CM, num_points=NUM_WAYPOINTS):
        self.radius_cm = radius_cm
        self.num_points = num_points
        self.reset()

    def get_current_target(self):
        self.update()
        angle = math.radians(self.current_angle_deg)
        return {
            "index": int(self.current_angle_deg) % 360,
            "angle_deg": self.current_angle_deg,
            "dx_cm": self.radius_cm * math.cos(angle),
            "dy_cm": self.radius_cm * math.sin(angle),
        }

    def start(self):
        self.start_ms = time.ticks_ms()
        self.current_angle_deg = 0.0
        self.lap_count = 0

    def update(self):
        if self.start_ms is None:
            return
        elapsed_ms = max(0, time.ticks_diff(time.ticks_ms(), self.start_ms))
        total_angle_deg = (elapsed_ms * CIRCLE_SPEED_DEG_PER_SEC) / 1000.0
        new_lap_count = int(total_angle_deg // 360.0)
        if new_lap_count > self.lap_count:
            self.lap_count = new_lap_count
            print(f"[Circle] completed lap {self.lap_count}")
        self.current_angle_deg = total_angle_deg % 360.0

    def calculate_sync_error(self, laser_angle_deg):
        target = self.get_current_target()
        if target is None:
            return 0.0

        error = laser_angle_deg - target["angle_deg"]
        while error > 180:
            error -= 360
        while error < -180:
            error += 360
        return error

    def reset(self):
        self.start_ms = None
        self.current_angle_deg = 0.0
        self.lap_count = 0


class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral = 10.0
        self.reset()

    def compute(self, error, dt=0.033):
        if dt <= 0:
            dt = 0.001

        self.integral += error * dt
        self.integral = max(-self.max_integral, min(self.max_integral, self.integral))
        derivative = (error - self.last_error) / dt
        self.last_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0


class TargetDetector:
    def __init__(self):
        self.target_rect = None
        self.target_center = None
        self.target_diameter_px = 0
        self.bullseye_center = None
        self.laser_spot = None
        self.target_found = False
        self.bullseye_found = False
        self.laser_found = False
        self.pixel_to_cm_x = 0.0
        self.pixel_to_cm_y = 0.0
        self.target_plane_corners_cm = None
        self.target_to_image_h = None
        self.image_to_target_h = None
        self.bullseye_plane_cm = None
        self.frame_id = 0
        self.target_miss_count = 0
        self.last_target_rect = None
        self.last_target_center = None
        self.last_target_diameter_px = 0
        self.last_target_corners = None
        self.last_bullseye_center = None
        self.bullseye_miss_count = 0
        self.last_laser_spot = None
        self.laser_miss_count = 0
        self.circle_radius_px = 0.0
        self.last_circle_radius_px = 0.0
        self.detected_ring_radius_cm = CIRCLE_RADIUS_CM
        self.last_detected_ring_radius_cm = 0.0
        self.target_center_history = []
        self.bullseye_center_history = []
        self.laser_spot_history = []
        self.circle_radius_history = []
        self.ring_radius_cm_history = []
        self.frozen_target_rect = None
        self.frozen_target_center = None
        self.frozen_target_diameter_px = 0
        self.frozen_bullseye_center = None
        self.predict_miss_count = 0

    def detect_all(self, img):
        self.frame_id += 1
        self._detect_target(img)
        if self.target_found:
            self._detect_bullseye(img)
            self._update_scale()
            self._detect_ring_radius(img)
            self._detect_laser(img)
            self._refresh_prediction_anchor()
        else:
            if self._prediction_available() and self.predict_miss_count < PREDICT_TRACK_MAX_FRAMES:
                self.predict_miss_count += 1
                self.target_rect = self.frozen_target_rect
                self.target_center = self.frozen_target_center
                self.target_diameter_px = self.frozen_target_diameter_px
                self.bullseye_center = self.frozen_bullseye_center
                self.bullseye_found = self.bullseye_plane_cm is not None
                if self.last_detected_ring_radius_cm > 0:
                    self.detected_ring_radius_cm = self.last_detected_ring_radius_cm
                else:
                    self.detected_ring_radius_cm = CIRCLE_RADIUS_CM
                self._detect_laser(img)
            else:
                self.target_center = None
                self.target_diameter_px = 0
                self.bullseye_center = None
                self.laser_spot = None
                self.bullseye_found = False
                self.laser_found = False
                self.detected_ring_radius_cm = CIRCLE_RADIUS_CM

    def _clamp_rect(self, x, y, w, h):
        x = max(0, min(FRAME_WIDTH - 1, int(x)))
        y = max(0, min(FRAME_HEIGHT - 1, int(y)))
        w = max(1, min(FRAME_WIDTH - x, int(w)))
        h = max(1, min(FRAME_HEIGHT - y, int(h)))
        return (x, y, w, h)

    def _distance_sq(self, p0, p1):
        dx = p0[0] - p1[0]
        dy = p0[1] - p1[1]
        return dx * dx + dy * dy

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

    def _smooth_scalar(self, current, last_value, alpha):
        if last_value <= 0:
            return current
        return last_value * (1 - alpha) + current * alpha

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

    def _solve_linear_system(self, matrix, values):
        size = len(values)
        a = []
        for row in range(size):
            current = []
            for col in range(size):
                current.append(float(matrix[row][col]))
            current.append(float(values[row]))
            a.append(current)

        for col in range(size):
            pivot = col
            pivot_abs = abs(a[pivot][col])
            for row in range(col + 1, size):
                value_abs = abs(a[row][col])
                if value_abs > pivot_abs:
                    pivot = row
                    pivot_abs = value_abs
            if pivot_abs < 1e-6:
                return None
            if pivot != col:
                tmp = a[col]
                a[col] = a[pivot]
                a[pivot] = tmp

            pivot_value = a[col][col]
            for idx in range(col, size + 1):
                a[col][idx] /= pivot_value

            for row in range(size):
                if row == col:
                    continue
                factor = a[row][col]
                for idx in range(col, size + 1):
                    a[row][idx] -= factor * a[col][idx]

        result = []
        for row in range(size):
            result.append(a[row][size])
        return tuple(result)

    def _compute_homography(self, src_points, dst_points):
        matrix = []
        values = []
        for idx in range(4):
            x = float(src_points[idx][0])
            y = float(src_points[idx][1])
            u = float(dst_points[idx][0])
            v = float(dst_points[idx][1])
            matrix.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
            values.append(u)
            matrix.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
            values.append(v)

        solution = self._solve_linear_system(matrix, values)
        if solution is None:
            return None
        return (
            (solution[0], solution[1], solution[2]),
            (solution[3], solution[4], solution[5]),
            (solution[6], solution[7], 1.0),
        )

    def _apply_homography(self, h, x, y):
        if h is None:
            return None
        denom = h[2][0] * x + h[2][1] * y + h[2][2]
        if abs(denom) < 1e-6:
            return None
        out_x = (h[0][0] * x + h[0][1] * y + h[0][2]) / denom
        out_y = (h[1][0] * x + h[1][1] * y + h[1][2]) / denom
        return (out_x, out_y)

    def _normalize_corners(self, corners):
        sx = 0
        sy = 0
        points = []
        for p in corners:
            sx += p[0]
            sy += p[1]
            points.append((float(p[0]), float(p[1])))
        center = (sx / 4, sy / 4)
        points.sort(key=lambda p: math.atan2(p[1] - center[1], p[0] - center[0]))

        top_left_idx = 0
        best_score = points[0][0] + points[0][1]
        for idx in range(1, 4):
            score = points[idx][0] + points[idx][1]
            if score < best_score:
                best_score = score
                top_left_idx = idx

        ordered = []
        for idx in range(4):
            ordered.append(points[(top_left_idx + idx) % 4])
        return ordered

    def _plane_size_cm_for_corners(self, corners):
        top = math.sqrt((corners[1][0] - corners[0][0]) ** 2 + (corners[1][1] - corners[0][1]) ** 2)
        right = math.sqrt((corners[2][0] - corners[1][0]) ** 2 + (corners[2][1] - corners[1][1]) ** 2)
        bottom = math.sqrt((corners[2][0] - corners[3][0]) ** 2 + (corners[2][1] - corners[3][1]) ** 2)
        left = math.sqrt((corners[3][0] - corners[0][0]) ** 2 + (corners[3][1] - corners[0][1]) ** 2)
        width_px = (top + bottom) * 0.5
        height_px = (left + right) * 0.5
        aspect = width_px / max(height_px, 1e-6)
        normal_error = abs(aspect - TARGET_ASPECT)
        swapped_error = abs(aspect - (1.0 / TARGET_ASPECT))
        if swapped_error < normal_error:
            return TARGET_HEIGHT_CM, TARGET_WIDTH_CM
        return TARGET_WIDTH_CM, TARGET_HEIGHT_CM

    def target_plane_cm_to_image(self, dx_cm, dy_cm):
        if self.target_to_image_h is None:
            return None
        projected = self._apply_homography(self.target_to_image_h, dx_cm, dy_cm)
        if projected is None:
            return None
        return self._clamp_point(projected)

    def _point_to_target_plane_cm(self, point):
        if self.image_to_target_h is None:
            return None
        projected = self._apply_homography(self.image_to_target_h, point[0], point[1])
        if projected is None:
            return None
        return projected

    def target_offset_cm_to_image_point(self, dx_cm, dy_cm):
        if self.bullseye_plane_cm is None:
            return None
        return self.target_plane_cm_to_image(
            self.bullseye_plane_cm[0] + dx_cm,
            self.bullseye_plane_cm[1] + dy_cm,
        )

    def _prediction_available(self):
        return (
            self.frozen_target_rect is not None
            and self.image_to_target_h is not None
            and self.target_to_image_h is not None
            and self.bullseye_plane_cm is not None
        )

    def _refresh_prediction_anchor(self):
        if not self.bullseye_found:
            return
        if not self._prediction_available() and self.target_rect is None:
            return
        if self.target_rect is not None:
            self.frozen_target_rect = self.target_rect
        if self.target_center is not None:
            self.frozen_target_center = self.target_center
        if self.target_diameter_px > 0:
            self.frozen_target_diameter_px = self.target_diameter_px
        if self.bullseye_center is not None:
            self.frozen_bullseye_center = self.bullseye_center
        self.predict_miss_count = 0

    def _expand_rect(self, rect, margin):
        x, y, w, h = rect
        return self._clamp_rect(x - margin, y - margin, w + margin * 2, h + margin * 2)

    def _push_point_history(self, history, point):
        history.append(point)
        if len(history) > POINT_HISTORY_LEN:
            history.pop(0)
        return history

    def _filter_point_history(self, history):
        if not history:
            return None
        xs = sorted(point[0] for point in history)
        ys = sorted(point[1] for point in history)
        mid = len(history) // 2
        return (xs[mid], ys[mid])

    def _push_scalar_history(self, history, value):
        history.append(value)
        if len(history) > POINT_HISTORY_LEN:
            history.pop(0)
        return history

    def _filter_scalar_history(self, history):
        if not history:
            return 0.0
        values = sorted(history)
        return values[len(values) // 2]

    def _pixel_to_cm_avg(self):
        values = []
        if self.pixel_to_cm_x > 0:
            values.append(self.pixel_to_cm_x)
        if self.pixel_to_cm_y > 0:
            values.append(self.pixel_to_cm_y)
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _offset_cm_to_image_point(self, dx_cm, dy_cm):
        if self.bullseye_center is None:
            return None
        pixel_to_cm = self._pixel_to_cm_avg()
        if pixel_to_cm <= 0:
            return None
        cx, cy = self.bullseye_center
        x = cx + (dx_cm / pixel_to_cm)
        y = cy + (dy_cm / pixel_to_cm)
        return self._clamp_point((x, y))

    def _effective_ring_radius_cm(self):
        if self.detected_ring_radius_cm > 0:
            return self.detected_ring_radius_cm
        return CIRCLE_RADIUS_CM

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

    def _clamp_point(self, point):
        return (
            max(0, min(FRAME_WIDTH - 1, int(point[0]))),
            max(0, min(FRAME_HEIGHT - 1, int(point[1]))),
        )

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
            x2 = x + w
        elif x2 >= (FRAME_WIDTH - TARGET_EDGE_MARGIN_PX) and w < int(last_w * TARGET_EDGE_COMP_MIN_RATIO):
            x = max(0, x2 - last_w)
            w = FRAME_WIDTH - x if x + last_w > FRAME_WIDTH else last_w

        if y <= TARGET_EDGE_MARGIN_PX and h < int(last_h * TARGET_EDGE_COMP_MIN_RATIO):
            h = last_h
            y2 = y + h
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
            points = (
                (sx, y1),
                (sx, y2),
                (x1, sy),
                (x2, sy),
            )
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
                best = (rect, corners, center)
        return best

    def _accept_center(self, candidate_center, last_center):
        if candidate_center is None or last_center is None:
            return True
        return self._distance_sq(candidate_center, last_center) <= (TARGET_MAX_JUMP_PX * TARGET_MAX_JUMP_PX)

    def _prepare_rect_image(self, img):
        rect_img = img.to_grayscale()
        rect_img.binary([RECT_BINARY_THRESHOLD])
        return rect_img

    def _refine_bullseye_center(self, img):
        if not (self.target_rect and self.target_center):
            return self.target_center

        x, y, w, h = self.target_rect
        roi_w = max(12, int(w * BULLSEYE_ROI_RATIO))
        roi_h = max(12, int(h * BULLSEYE_ROI_RATIO))
        cx, cy = self.target_center
        roi_x = max(x, cx - roi_w // 2)
        roi_y = max(y, cy - roi_h // 2)
        roi_x2 = min(x + w, roi_x + roi_w)
        roi_y2 = min(y + h, roi_y + roi_h)
        roi = (
            roi_x,
            roi_y,
            max(1, roi_x2 - roi_x),
            max(1, roi_y2 - roi_y),
        )

        blobs = img.find_blobs(
            [BLACK_THRESHOLD],
            roi=roi,
            pixels_threshold=3,
            area_threshold=3,
            merge=True,
        ) or []
        if not blobs:
            return self.target_center

        gate_px = max(6, int(min(w, h) * BULLSEYE_GATE_RATIO * 0.5))
        gate_sq = gate_px * gate_px
        weighted_x = 0.0
        weighted_y = 0.0
        total_weight = 0.0
        for blob in blobs:
            bx = blob.cx()
            by = blob.cy()
            dist_sq = self._distance_sq((bx, by), self.target_center)
            if dist_sq > gate_sq:
                continue
            weight = max(1.0, float(blob.pixels()))
            weighted_x += bx * weight
            weighted_y += by * weight
            total_weight += weight

        if total_weight <= 0:
            return self.target_center

        refined = (
            int(weighted_x / total_weight),
            int(weighted_y / total_weight),
        )
        return (
            int(self.target_center[0] * (1 - BULLSEYE_BLEND_ALPHA) + refined[0] * BULLSEYE_BLEND_ALPHA),
            int(self.target_center[1] * (1 - BULLSEYE_BLEND_ALPHA) + refined[1] * BULLSEYE_BLEND_ALPHA),
        )

    def _detect_target(self, img):
        if self.target_found and (self.frame_id % TARGET_DETECT_INTERVAL) != 0:
            return

        previous_rect = self.last_target_rect
        previous_center = self.last_target_center
        previous_diameter = self.last_target_diameter_px
        previous_corners = self.last_target_corners

        self.target_found = False
        self.target_rect = None
        self.target_center = None
        self.target_diameter_px = 0

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
                self.target_diameter_px = previous_diameter
                self.last_target_corners = previous_corners
                self.target_found = True
            else:
                self.last_target_rect = None
                self.last_target_center = None
                self.last_target_diameter_px = 0
                self.last_target_corners = None
                self.target_center_history = []
                self.bullseye_center_history = []
                self.laser_spot_history = []
                self.circle_radius_history = []
                self.ring_radius_cm_history = []
                self.circle_radius_px = 0.0
                self.last_circle_radius_px = 0.0
                self.detected_ring_radius_cm = CIRCLE_RADIUS_CM
                self.last_detected_ring_radius_cm = 0.0
            return

        self.target_miss_count = 0
        rect, corners, center = chosen
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
        self.target_diameter_px = min(rect[2], rect[3])
        self.target_found = True
        self.last_target_rect = self.target_rect
        self.last_target_corners = corners
        self.last_target_center = self.target_center
        self.last_target_diameter_px = self.target_diameter_px

    def _detect_bullseye(self, img):
        self.bullseye_found = False
        self.bullseye_center = None
        if not (self.target_found and self.target_rect and self.target_center):
            return
        if (
            self.last_bullseye_center is not None
            and (self.frame_id % BULLSEYE_DETECT_INTERVAL) != 0
        ):
            self.bullseye_center = self.last_bullseye_center
            self.bullseye_found = True
            return

        refined_center = self._refine_bullseye_center(img)
        self.bullseye_center = self._smooth_center(
            refined_center,
            self.last_bullseye_center,
            BULLSEYE_CENTER_ALPHA,
            TARGET_RESET_DIST_PX,
            TARGET_STICKY_DIST_PX,
        )
        self.bullseye_center = self._apply_motion_lead(
            self.bullseye_center,
            self.last_bullseye_center,
            BULLSEYE_LEAD_GAIN,
            BULLSEYE_LEAD_MAX_PX,
        )
        self.last_bullseye_center = self.bullseye_center
        self.bullseye_found = True
        self.bullseye_miss_count = 0

    def _detect_laser(self, img):
        self.laser_found = False
        self.laser_spot = None
        if self.target_rect is None:
            return
        roi = self.target_rect
        if not self.target_found:
            margin = max(LASER_PREDICT_ROI_MARGIN_PX, self.target_diameter_px // 2)
            roi = self._expand_rect(roi, margin)

        blobs = img.find_blobs(
            [VIOLET_THRESHOLD],
            roi=roi,
            pixels_threshold=2,
            area_threshold=2,
            merge=True,
        )
        if blobs:
            reference = self.last_laser_spot or self.target_center
            gate_px = max(5, self.target_diameter_px // 10)
            gate_sq = gate_px * gate_px
            candidate_blobs = [
                blob
                for blob in blobs
                if self._distance_sq((blob.cx(), blob.cy()), reference) <= gate_sq
            ]
            if not candidate_blobs:
                candidate_blobs = blobs
            best = min(
                candidate_blobs,
                key=lambda b: (
                    self._distance_sq((b.cx(), b.cy()), reference),
                    -b.density(),
                ),
            )
            raw_spot = (best.cx(), best.cy())
            if self.last_laser_spot is None:
                self.laser_spot = raw_spot
            else:
                self.laser_spot = self._smooth_center(
                    raw_spot,
                    self.last_laser_spot,
                    0.9,
                    LASER_STICKY_PX * 3,
                    LASER_STICKY_PX,
                )
            self._push_point_history(self.laser_spot_history, self.laser_spot)
            self.laser_spot = self._filter_point_history(self.laser_spot_history)
            self.laser_found = True
            self.last_laser_spot = self.laser_spot
            self.laser_miss_count = 0
        else:
            self.laser_miss_count += 1
            if self.last_laser_spot is not None and self.laser_miss_count <= LASER_MAX_MISS_FRAMES:
                self.laser_spot = self.last_laser_spot
                self._push_point_history(self.laser_spot_history, self.laser_spot)
                self.laser_spot = self._filter_point_history(self.laser_spot_history)
                self.laser_found = True
            else:
                self.laser_spot_history = []

    def _detect_ring_radius(self, img):
        if not (self.target_found and self.target_rect and self.bullseye_center):
            self.detected_ring_radius_cm = CIRCLE_RADIUS_CM
            return
        if self.bullseye_plane_cm is None:
            self.detected_ring_radius_cm = CIRCLE_RADIUS_CM
            return
        if (
            self.last_detected_ring_radius_cm > 0
            and (self.frame_id % RING_DETECT_INTERVAL) != 0
        ):
            self.detected_ring_radius_cm = self.last_detected_ring_radius_cm
            return

        max_blob_size = max(RING_BLOB_MIN_SIZE_PX + 1, self.target_diameter_px // RING_BLOB_MAX_SIZE_DIV)
        black_blobs = img.find_blobs(
            [RING_THRESHOLD],
            roi=self.target_rect,
            pixels_threshold=1,
            area_threshold=1,
            merge=False,
        ) or []

        samples = []
        for blob in black_blobs:
            size = max(blob.w(), blob.h())
            if size < RING_BLOB_MIN_SIZE_PX or size > max_blob_size:
                continue
            plane_point = self._point_to_target_plane_cm((blob.cx(), blob.cy()))
            if plane_point is None:
                continue
            dx_cm = plane_point[0] - self.bullseye_plane_cm[0]
            dy_cm = plane_point[1] - self.bullseye_plane_cm[1]
            radius_cm = math.sqrt(dx_cm * dx_cm + dy_cm * dy_cm)
            if radius_cm < RING_RADIUS_MIN_CM or radius_cm > RING_RADIUS_MAX_CM:
                continue
            samples.append((radius_cm, max(1, min(blob.w(), blob.h()))))

        if not samples:
            if self.last_detected_ring_radius_cm > 0:
                self.detected_ring_radius_cm = self.last_detected_ring_radius_cm
            else:
                self.detected_ring_radius_cm = CIRCLE_RADIUS_CM
            return

        samples.sort(key=lambda item: item[0])
        clusters = []
        for radius_cm, weight in samples:
            if not clusters or abs(radius_cm - clusters[-1]["radius"]) > RING_RADIUS_CLUSTER_TOL_CM:
                clusters.append({
                    "radius": radius_cm,
                    "weight_sum": float(weight),
                    "sample_count": 1,
                })
            else:
                cluster = clusters[-1]
                total_weight = cluster["weight_sum"] + weight
                cluster["radius"] = (
                    cluster["radius"] * cluster["weight_sum"] + radius_cm * weight
                ) / total_weight
                cluster["weight_sum"] = total_weight
                cluster["sample_count"] += 1

        candidate_clusters = []
        for cluster in clusters:
            if (
                cluster["sample_count"] >= RING_MIN_CLUSTER_SAMPLES
                or cluster["weight_sum"] >= RING_MIN_CLUSTER_WEIGHT
            ):
                candidate_clusters.append(cluster)

        if not candidate_clusters:
            candidate_clusters = clusters

        candidate_clusters.sort(key=lambda cluster: cluster["radius"])

        best_cluster = None
        target_index = RING_TARGET_INDEX_FROM_CENTER - 1
        if 0 <= target_index < len(candidate_clusters):
            best_cluster = candidate_clusters[target_index]
        elif self.last_detected_ring_radius_cm > 0 and candidate_clusters:
            best_cluster = min(
                candidate_clusters,
                key=lambda cluster: abs(cluster["radius"] - self.last_detected_ring_radius_cm),
            )
        elif candidate_clusters:
            best_cluster = candidate_clusters[-1]

        if best_cluster is None:
            self.detected_ring_radius_cm = CIRCLE_RADIUS_CM
            return

        detected_radius = best_cluster["radius"]
        if self.last_detected_ring_radius_cm > 0:
            detected_radius = self._smooth_scalar(
                detected_radius,
                self.last_detected_ring_radius_cm,
                CIRCLE_RADIUS_SMOOTHING_ALPHA,
            )
        self._push_scalar_history(self.ring_radius_cm_history, detected_radius)
        detected_radius = self._filter_scalar_history(self.ring_radius_cm_history)
        self.detected_ring_radius_cm = detected_radius
        self.last_detected_ring_radius_cm = detected_radius

    def _update_scale(self):
        self.pixel_to_cm_x = 0.0
        self.pixel_to_cm_y = 0.0
        self.target_to_image_h = None
        self.image_to_target_h = None
        self.bullseye_plane_cm = None
        if not (self.target_found and self.last_target_corners):
            return

        corners = self.last_target_corners
        ordered_corners = self._normalize_corners(corners)
        width_cm, height_cm = self._plane_size_cm_for_corners(ordered_corners)
        plane_corners = (
            (-width_cm * 0.5, height_cm * 0.5),
            (width_cm * 0.5, height_cm * 0.5),
            (width_cm * 0.5, -height_cm * 0.5),
            (-width_cm * 0.5, -height_cm * 0.5),
        )
        self.target_plane_corners_cm = plane_corners
        self.target_to_image_h = self._compute_homography(plane_corners, ordered_corners)
        self.image_to_target_h = self._compute_homography(ordered_corners, plane_corners)
        edges = []
        for idx in range(4):
            p0 = ordered_corners[idx]
            p1 = ordered_corners[(idx + 1) % 4]
            dx = p0[0] - p1[0]
            dy = p0[1] - p1[1]
            edges.append(math.sqrt(dx * dx + dy * dy))

        width_px = (edges[0] + edges[2]) * 0.5
        height_px = (edges[1] + edges[3]) * 0.5
        if width_px <= 0 or height_px <= 0:
            width_px = self.target_rect[2]
            height_px = self.target_rect[3]
        if width_px <= 0 or height_px <= 0:
            return

        self.pixel_to_cm_x = width_cm / width_px
        self.pixel_to_cm_y = height_cm / height_px

        radius_px = (
            (CIRCLE_RADIUS_CM / self.pixel_to_cm_x)
            + (CIRCLE_RADIUS_CM / self.pixel_to_cm_y)
        ) * 0.5
        self.circle_radius_px = self._smooth_scalar(
            radius_px,
            self.last_circle_radius_px,
            CIRCLE_RADIUS_SMOOTHING_ALPHA,
        )
        self._push_scalar_history(
            self.circle_radius_history,
            self.circle_radius_px,
        )
        self.circle_radius_px = self._filter_scalar_history(
            self.circle_radius_history
        )
        self.last_circle_radius_px = self.circle_radius_px
        if self.bullseye_found and self.bullseye_center:
            self.bullseye_plane_cm = self._point_to_target_plane_cm(self.bullseye_center)

    def get_laser_position_cm(self):
        if not self.bullseye_found or not self.laser_found:
            return 0.0, 0.0, 0.0
        if self.bullseye_plane_cm is None:
            return 0.0, 0.0, 0.0
        laser_plane_cm = self._point_to_target_plane_cm(self.laser_spot)
        if laser_plane_cm is None:
            return 0.0, 0.0, 0.0
        dx_cm = laser_plane_cm[0] - self.bullseye_plane_cm[0]
        dy_cm = laser_plane_cm[1] - self.bullseye_plane_cm[1]
        angle_deg = math.degrees(math.atan2(-dy_cm, dx_cm))
        return dx_cm, dy_cm, angle_deg

    def get_error_from_target(self, target_dx_cm, target_dy_cm):
        laser_dx, laser_dy, _ = self.get_laser_position_cm()
        return target_dx_cm - laser_dx, target_dy_cm - laser_dy


class CircleModeSystem:
    def __init__(self):
        self.detector = TargetDetector()
        self.tracker = CircleTracker()
        self.uart = None
        self.comm_started = False
        self.start_button = StartButton(START_BUTTON_BOARD_PIN, START_BUTTON_GPIO_NUM)
        self.frame_count = 0
        self.gc_counter = 0
        self.state = "IDLE"
        self.start_align_frames = 0
        self.last_sent_error = None
        self.error_jump_count = 0

    def _control_quality(self):
        if not self.detector.target_found:
            return False, "target_lost"
        if not self.detector.bullseye_found or self.detector.bullseye_plane_cm is None:
            return False, "bullseye_lost"
        if not self.detector.laser_found:
            return False, "laser_lost"
        if self.detector.image_to_target_h is None:
            return False, "mapping_lost"
        return True, "ok"

    def _error_distance_cm(self, error_a, error_b):
        dx = error_a[0] - error_b[0]
        dy = error_a[1] - error_b[1]
        return math.sqrt(dx * dx + dy * dy)

    def _filter_error_jump(self, error_x, error_y):
        current_error = (error_x, error_y)
        if self.last_sent_error is None:
            self.last_sent_error = current_error
            self.error_jump_count = 0
            return error_x, error_y, True

        if self._error_distance_cm(current_error, self.last_sent_error) <= ERROR_JUMP_MAX_CM:
            self.last_sent_error = current_error
            self.error_jump_count = 0
            return error_x, error_y, True

        self.error_jump_count += 1
        if self.error_jump_count <= ERROR_JUMP_REJECT_FRAMES:
            return self.last_sent_error[0], self.last_sent_error[1], False

        self.last_sent_error = current_error
        self.error_jump_count = 0
        return error_x, error_y, True

    def init_uart(self):
        try:
            fpioa = FPIOA()
            fpioa.set_function(TX_PIN, fpioa.UART2_TXD)
            fpioa.set_function(RX_PIN, fpioa.UART2_RXD)
            self.uart = UART(UART_ID, baudrate=UART_BAUDRATE)
            print(f"[UART] OK ({UART_BAUDRATE})")
            return True
        except Exception as e:
            print(f"[UART] failed: {e}")
            return False

    def send_error_command(self, error_x, error_y, sync_ok):
        if self.uart is None or not self.comm_started:
            return
        sync_flag = 1 if sync_ok else 0
        self.uart.write(
            "CIRC,{:.3f},{:.3f},{}\n".format(error_x, error_y, sync_flag)
        )

    def send_status(self, status):
        if self.uart and self.comm_started:
            self.uart.write("STS,{}\n".format(status))

    def _poll_uart_commands(self):
        commands = []
        if (not self.uart) or (not self.comm_started):
            return commands
        while self.uart.any():
            try:
                data = self.uart.readline()
            except Exception:
                break
            if not data:
                break
            try:
                command = data.decode("utf-8").strip()
            except Exception:
                continue
            if command:
                commands.append(command)
        return commands

    def _start_point_error_cm(self):
        target = self.tracker.get_current_target()
        if (
            target is None
            or not self.detector.target_found
            or not self.detector.bullseye_found
            or not self.detector.laser_found
        ):
            return None
        error_x, error_y = self.detector.get_error_from_target(
            target["dx_cm"], target["dy_cm"]
        )
        return math.sqrt(error_x * error_x + error_y * error_y)

    def _check_start_alignment(self):
        start_error_cm = self._start_point_error_cm()
        if start_error_cm is None or start_error_cm > START_ALIGN_TOL_CM:
            self.start_align_frames = 0
            return False
        self.start_align_frames += 1
        return self.start_align_frames >= START_ALIGN_HOLD_FRAMES

    def _update_start_button(self):
        if self.comm_started:
            return
        if self.start_button.poll_pressed():
            self.comm_started = True
            print("[Comm] start button pressed, UART enabled")
            if self.state != "IDLE":
                self.send_status(self.state)

    def process_frame(self, img):
        self.frame_count += 1
        self.gc_counter += 1
        self.detector.detect_all(img)
        self._update_start_button()
        commands = self._poll_uart_commands()

        if self.state == "IDLE":
            if self.detector.target_found and self.detector.bullseye_found:
                self.state = "WAITING"
                self.start_align_frames = 0
                self.tracker.reset()
                self.send_status("WAITING")
                print("[State] target ready -> WAITING")
        elif self.state == "WAITING":
            if self._check_start_alignment():
                self.state = "RUNNING"
                self.tracker.reset()
                self.tracker.start()
                self.start_align_frames = 0
                self.send_status("RUNNING")
                print("[State] laser aligned with start point -> RUNNING")
        elif self.state == "RUNNING":
            if "STOP_CIRCLE" in commands or "STOP" in commands:
                self.state = "WAITING"
                self.tracker.reset()
                self.start_align_frames = 0
                self.last_sent_error = None
                self.error_jump_count = 0
                self.send_status("STOPPED")
                print("[State] stop command -> WAITING")
            else:
                self._handle_running()

        if DEBUG_MODE:
            self._draw_overlay(img)
        return img

    def _handle_running(self):
        target = self.tracker.get_current_target()
        if target is None:
            return

        target_dx = target["dx_cm"]
        target_dy = target["dy_cm"]

        error_x, error_y = self.detector.get_error_from_target(
            target_dx, target_dy
        )
        _, _, laser_angle = self.detector.get_laser_position_cm()
        sync_error = self.tracker.calculate_sync_error(laser_angle)
        quality_ok, _ = self._control_quality()
        sync_ok = quality_ok and abs(sync_error) < MAX_SYNC_ERROR_DEG

        if quality_ok:
            filtered_x, filtered_y, stable_ok = self._filter_error_jump(error_x, error_y)
            self.send_error_command(filtered_x, filtered_y, sync_ok and stable_ok)
        else:
            self.last_sent_error = None
            self.error_jump_count = 0
            self.send_error_command(0.0, 0.0, False)

    def _draw_overlay(self, img):
        if self.detector.target_found and self.detector.target_rect:
            x, y, w, h = self.detector.target_rect
            img.draw_rectangle(x, y, w, h, color=(0, 255, 0), thickness=2)
        if self.detector.target_found and self.detector.target_center:
            cx, cy = self.detector.target_center
            img.draw_cross(cx, cy, color=(0, 255, 0), size=8)

        if self.detector.target_found and self.detector.target_center:
            if self.detector.bullseye_plane_cm is not None:
                prev_point = None
                sample_count = OVERLAY_RING_SAMPLE_COUNT
                radius_cm = self.detector._effective_ring_radius_cm()
                for idx in range(sample_count + 1):
                    angle = 2 * math.pi * idx / sample_count
                    sample_point = self.detector.target_offset_cm_to_image_point(
                        radius_cm * math.cos(angle),
                        radius_cm * math.sin(angle),
                    )
                    if prev_point is not None and sample_point is not None:
                        img.draw_line(
                            prev_point[0],
                            prev_point[1],
                            sample_point[0],
                            sample_point[1],
                            color=(0, 0, 255),
                            thickness=2,
                        )
                    prev_point = sample_point

                target = self.tracker.get_current_target()
                if target:
                    target_dx = target["dx_cm"]
                    target_dy = target["dy_cm"]
                    target_point = self.detector.target_offset_cm_to_image_point(target_dx, target_dy)
                    if target_point is not None:
                        img.draw_circle(
                            target_point[0],
                            target_point[1],
                            4,
                            color=(0, 255, 255),
                            thickness=2,
                        )

        if self.detector.laser_found and self.detector.laser_spot:
            lx, ly = self.detector.laser_spot
            img.draw_circle(lx, ly, 6, color=(255, 255, 0), thickness=2)

        if DEBUG_TEXT_OVERLAY:
            state_colors = {
                "IDLE": (128, 128, 128),
                "WAITING": (255, 255, 0),
                "RUNNING": (0, 255, 0),
                "COMPLETE": (0, 255, 255),
                "ERROR": (255, 0, 0),
            }
            color = state_colors.get(self.state, (255, 255, 255))
            draw_text(img, 10, 10, f"State: {self.state}", color=color, scale=2)

            if self.state == "WAITING":
                start_error_cm = self._start_point_error_cm()
                if start_error_cm is None:
                    text = "Align laser to start point"
                else:
                    text = "Start err: {:.1f}cm".format(start_error_cm)
                draw_text(img, 10, 40, text, color=(255, 255, 255), scale=1)
                draw_text(
                    img,
                    10,
                    60,
                    "Hold aligned: {}/{}".format(
                        self.start_align_frames,
                        START_ALIGN_HOLD_FRAMES,
                    ),
                    color=(255, 255, 255),
                    scale=1,
                )

            if self.state == "RUNNING":
                target = self.tracker.get_current_target()
                if target:
                    target_dx = target["dx_cm"]
                    target_dy = target["dy_cm"]
                    draw_text(
                        img,
                        10,
                        40,
                        f"Target: {target_dx:.1f},{target_dy:.1f}",
                        color=(255, 255, 255),
                        scale=1,
                    )
                    draw_text(
                        img,
                        10,
                        60,
                        f"Lap: {self.tracker.lap_count} Angle: {target['angle_deg']:.1f}",
                        color=(255, 255, 255),
                        scale=1,
                    )
        if not self.comm_started:
            draw_text(
                img,
                10,
                FRAME_HEIGHT - 20,
                "PRESS GPIO28 TO START UART",
                color=(255, 255, 0),
                scale=1,
            )


def main():
    print("=" * 50)
    print("K230 Circle Mode - laser arc tracking")
    print("build:", BUILD_TAG)
    print("Competition task: circle mode")
    print("=" * 50)
    apply_calibration()

    init_preview_display()
    sensor = init_camera_sensor()
    start_camera(sensor)

    system = CircleModeSystem()
    system.init_uart()
    print("system ready")

    try:
        while True:
            os.exitpoint()
            try:
                img = snapshot_with_retry(sensor)
            except RuntimeError as e:
                print(f"[Sensor] snapshot failed, retry by restart: {e}")
                sensor = restart_camera(sensor)
                img = snapshot_with_retry(sensor)
            img = system.process_frame(img)
            Display.show_image(img)
            time.sleep_ms(FRAME_LOOP_DELAY_MS)

            if system.gc_counter >= GC_FRAME_INTERVAL:
                gc.collect()
                system.gc_counter = 0
    except KeyboardInterrupt:
        print("\nuser interrupted")
    except Exception as e:
        print(f"error: {e}")
        sys.print_exception(e)
    finally:
        if isinstance(sensor, Sensor):
            sensor.stop()
        Display.deinit()
        os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
        time.sleep_ms(100)
        MediaManager.deinit()
        Sensor.deinit()
        if system.uart:
            system.uart.deinit()
        print("system stopped")


if __name__ == "__main__":
    main()
