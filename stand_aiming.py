import gc
import math
import os
import sys
sys.path.insert(0, '/sdcard/app')
import time

from common_hw import (DebouncedButton as StartButton, Display, draw_text,
                       camera_init, camera_start, camera_snapshot,
                       camera_restart, camera_deinit, display_init)
from vision_utils import (clamp_rect, dist_sq, smooth_center, apply_motion_lead,
                          rect_aspect_error, rect_center_from_corners,
                          rect_size_change_ok, compensate_edge_rect,
                          rect_overlap_ratio, rect_border_hit_ratio,
                          normalize_corners, compute_homography,
                          apply_homography, log_info)
from pitch_search import PitchSearchController

try:
    import cv_lite
except ImportError:
    cv_lite = None

try:
    import image as image_module
except ImportError:
    image_module = None

try:
    from k230_common import build_stepper_controller
except ImportError:
    def build_stepper_controller(axis_overrides=None):
        class _NoopStepperController:
            ready = False
            def drive(self, *a, **kw): pass
            def drive_velocity(self, *a, **kw): pass
            def stop(self): pass
            def disable(self): pass
            def deinit(self): pass
        return _NoopStepperController()

# ==========================================
# 常量配置
# ==========================================
CAMERA_ID = 2
FRAME_WIDTH = 400
FRAME_HEIGHT = 300
SENSOR_HMIRROR = False
SENSOR_VFLIP = False
START_BUTTON_BOARD_PIN = 28
START_BUTTON_GPIO_NUM = 28
RECT_THRESHOLD = 8000
RECT_DARK_THRESHOLDS = ((0, 72), (0, 90))
RECT_BORDER_THRESHOLD = RECT_DARK_THRESHOLDS[0]
RECT_TRACK_THRESHOLD = 25000
RECT_GLOBAL_THRESHOLD = 20000
RECT_TRACK_MAX_REGIONS = 2
RECT_REACQUIRE_MAX_REGIONS = 4
TARGET_WIDTH_CM = 25.0
TARGET_HEIGHT_CM = 29.7
TARGET_ASPECT = TARGET_WIDTH_CM / TARGET_HEIGHT_CM
TARGET_ASPECT_PENALTY_SCALE = 12000
TARGET_MIN_W = 44
TARGET_MIN_H = 44
TARGET_MIN_AREA = 3600
TARGET_MAX_MISS_FRAMES = 3
TARGET_REACQUIRE_FRAMES = 24
TARGET_REACQUIRE_GLOBAL_AFTER = 4
TARGET_DETECT_INTERVAL = 1
TARGET_STABLE_FRAMES = 3
TARGET_CENTER_ALPHA = 1.00
TARGET_CENTER_RESET_PX = 72
TARGET_CENTER_STICKY_PX = 2
TARGET_CORNER_ALPHA = 1.00
TARGET_CORNER_RESET_PX = 72
TARGET_CORNER_STEP_LIMIT_PX = 96
CONTROL_CENTER_ALPHA_IDLE = 0.40
CONTROL_CENTER_ALPHA_DRIVE = 0.50
CONTROL_CORNER_ALPHA_IDLE = 0.42
CONTROL_CORNER_ALPHA_DRIVE = 0.50
CONTROL_FILTER_RESET_PX = 96
CONTROL_FILTER_STICKY_PX = 2
CONTROL_FILTER_DRIVE_ERROR_CM = 0.35
CONTROL_ERROR_ALPHA_IDLE = 0.65
CONTROL_ERROR_ALPHA_DRIVE = 0.35
CONTROL_ERROR_RESET_CM = 2.5
MAX_AIM_ERROR_CM = 2.0
ALIGNED_TOLERANCE_CM = 1.2
LASER_DOT_X_PX = 188
LASER_DOT_Y_PX = 135
DEBUG_MODE = True
DEBUG_TEXT_OVERLAY = True
FRAME_LOOP_DELAY_MS = 0
GC_INTERVAL = 60
MAX_CONSECUTIVE_SNAPSHOT_FAILURES = 5
BUILD_TAG = "2026-VFINAL-FIXED-STABLE"
AUTO_START = True
MOTOR_CONTROL_ENABLED = True
PITCH_SEARCH_ENABLED = False
PITCH_SEARCH_START_DELAY_MS = 500
PITCH_SEARCH_ERROR_CM = 2.0
PITCH_SEARCH_SEGMENTS = ((1, 700), (0, 150), (-1, 1400), (0, 150), (1, 700), (0, 800))
YAW_SEARCH_ENABLED = True
YAW_SEARCH_START_DELAY_MS = 200
YAW_SEARCH_FREQ_HZ = 150  #测试完改回300
YAW_SEARCH_DIRECTION = -1
YAW_SEARCH_HALF_TURN_STEPS = 1600
YAW_SEARCH_REVERSE_PAUSE_MS = 120

STEPPER_AXIS_OVERRIDES = {
    "x": {
        "deadband": float(ALIGNED_TOLERANCE_CM),
        "error_full_scale": 20.0,
        "command_sign": 1,
        "pid_kp": 0.1,
        "pid_ki": 0.0,
        "pid_kd": 0.0,      # 彻底关掉 Kd
        "min_freq": 80,
        "max_freq": 200,
        "manual_max_freq": 1000,
        "ramp_hz_per_s": 300.0,
        "integral_limit": 10.0,
        "integral_active_error": 3.0,
    },
    "y": {
        "deadband": float(ALIGNED_TOLERANCE_CM) * 1.35,
        "error_full_scale": 20.0,
        "command_sign": 1,
        "pid_kp": 0.07,
        "pid_ki": 0.0,
        "pid_kd": 0.0,
        "min_freq": 60,
        "max_freq": 180,
        "ramp_hz_per_s": 120.0,
        "integral_limit": 10.0,
        "integral_active_error": 3.0,
    },
}

# ==========================================
# 核心追踪与检测组件（已移除卡尔曼滤波）
# ==========================================
class RectTracker:
    def __init__(self):
        self.frame_id = 0
        self.target_rect = None
        self.target_center = None
        self.target_found = False
        self.target_fresh = False
        self.target_miss_count = 0
        self.last_target_center = None
        self.last_target_corners = None
        self.target_corners = None
        self.search_anchor_rect = None

    def _sort_corners_fixed(self, pts):
        """严格的拓扑四角排序（左上、右上、右下、左下），防止倾斜时顶点对调闪烁"""
        pts_sorted_x = sorted(pts, key=lambda p: p[0])
        left_pts = pts_sorted_x[:2]
        right_pts = pts_sorted_x[2:]
        
        left_pts_sorted_y = sorted(left_pts, key=lambda p: p[1])
        tl = left_pts_sorted_y[0]
        bl = left_pts_sorted_y[1]
        
        right_pts_sorted_y = sorted(right_pts, key=lambda p: p[1])
        tr = right_pts_sorted_y[0]
        br = right_pts_sorted_y[1]
        
        return (tl, tr, br, bl)

    def _smooth_corners(self, corners, previous_corners):
        current = self._sort_corners_fixed(corners)
        if previous_corners is None or len(previous_corners) != 4:
            return tuple((int(p[0]), int(p[1])) for p in current)
        
        previous = self._sort_corners_fixed(previous_corners)
        smoothed = []
        reset_sq = TARGET_CORNER_RESET_PX * TARGET_CORNER_RESET_PX
        for idx in range(4):
            prev_x, prev_y = previous[idx]
            curr_x, curr_y = current[idx]
            dx = curr_x - prev_x
            dy = curr_y - prev_y
            if dx * dx + dy * dy > reset_sq:
                smoothed.append((int(curr_x), int(curr_y)))
                continue

            limited_x = prev_x + max(-TARGET_CORNER_STEP_LIMIT_PX, min(TARGET_CORNER_STEP_LIMIT_PX, dx))
            limited_y = prev_y + max(-TARGET_CORNER_STEP_LIMIT_PX, min(TARGET_CORNER_STEP_LIMIT_PX, dy))
            smoothed.append((
                int(prev_x * (1.0 - TARGET_CORNER_ALPHA) + limited_x * TARGET_CORNER_ALPHA),
                int(prev_y * (1.0 - TARGET_CORNER_ALPHA) + limited_y * TARGET_CORNER_ALPHA),
            ))
        return tuple(smoothed)

    def detect(self, img, force_global=False):
        self.frame_id += 1
        if self.target_found and (self.frame_id % TARGET_DETECT_INTERVAL) != 0:
            return self.target_found, self.target_rect, self.target_center

        previous_center = None if force_global else self.last_target_center
        previous_corners = self.last_target_corners
        previous_rect = None if force_global else (self.target_rect or self.search_anchor_rect)
        
        self.target_found = False
        self.target_fresh = False

        # 【终极防爆优化】锁死追踪 ROI，绝不放开全局边缘扫描
        if previous_rect is not None:
            px, py, pw, ph = previous_rect
            pad = 30 if self.target_rect is not None else 80
            rx = max(0, px - pad)
            ry = max(0, py - pad)
            rw = min(FRAME_WIDTH - rx, pw + pad * 2)
            rh = min(FRAME_HEIGHT - ry, ph + pad * 2)
            search_roi = (rx, ry, rw, rh)
        else:
            search_roi = (40, 30, FRAME_WIDTH - 80, FRAME_HEIGHT - 60)

        # 【物理防爆防御】高阈值 + max_regions=2 限制线段总数，外加 try-except 保护机制
        rects = []
        rect_keys = []
        scan_threshold = RECT_TRACK_THRESHOLD if previous_rect is not None else RECT_GLOBAL_THRESHOLD
        scan_max_regions = RECT_TRACK_MAX_REGIONS if previous_rect is not None else RECT_REACQUIRE_MAX_REGIONS
        try:
            local_rects = img.find_rects(
                roi=search_roi,
                threshold=scan_threshold,
                max_regions=scan_max_regions
            ) or []
            for r in local_rects:
                key = r.rect()
                if key not in rect_keys:
                    rect_keys.append(key)
                    rects.append(r)
        except (RuntimeError, MemoryError):
            gc.collect()

        if self.target_rect is None and self.target_miss_count >= TARGET_REACQUIRE_GLOBAL_AFTER:
            global_roi = (40, 30, FRAME_WIDTH - 80, FRAME_HEIGHT - 60)
            if global_roi != search_roi:
                try:
                    global_rects = img.find_rects(
                        roi=global_roi,
                        threshold=RECT_GLOBAL_THRESHOLD,
                        max_regions=RECT_REACQUIRE_MAX_REGIONS
                    ) or []
                    for r in global_rects:
                        key = r.rect()
                        if key not in rect_keys:
                            rect_keys.append(key)
                            rects.append(r)
                except (RuntimeError, MemoryError):
                    gc.collect()

        best_rect_obj = None
        best_score = None
        best_corners = None
        
        if rects:
            for r in rects:
                x, y, w, h = r.rect()
                if w < TARGET_MIN_W or h < TARGET_MIN_H: continue
                if x <= 1 or y <= 1 or x + w >= FRAME_WIDTH - 1 or y + h >= FRAME_HEIGHT - 1: continue
                
                current_aspect = float(w) / float(h)
                aspect_error = abs(current_aspect - TARGET_ASPECT)
                if aspect_error > 0.22: continue  
                
                c = r.corners()
                if not c or len(c) != 4: continue
                
                score = w * h
                score -= int(aspect_error * TARGET_ASPECT_PENALTY_SCALE)
                
                if previous_center:
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                    d_sq = (cx - previous_center[0])**2 + (cy - previous_center[1])**2
                    score -= int(d_sq * 2.0)
                
                if best_score is None or score > best_score:
                    best_score = score
                    best_rect_obj = r
                    best_corners = c

        # 丢失记忆与重置
        if not best_rect_obj or best_corners is None:
            self.target_miss_count += 1
            if self.target_miss_count <= TARGET_MAX_MISS_FRAMES and previous_center:
                self.target_found = True
                self.target_fresh = False 
                return self.target_found, self.target_rect, self.target_center
            else:
                self.target_rect = None
                self.target_center = None
                self.target_corners = None
                self.last_target_center = None
                self.last_target_corners = None
                if self.target_miss_count > TARGET_REACQUIRE_FRAMES:
                    self.search_anchor_rect = None
                return self.target_found, self.target_rect, self.target_center

        self.target_miss_count = 0
        self.target_found = True
        self.target_fresh = True

        # 计算当前的几何测量数据，直接作为输出值使用
        meas_cx = sum(p[0] for p in best_corners) / 4.0
        meas_cy = sum(p[1] for p in best_corners) / 4.0
        xs = [p[0] for p in best_corners]
        ys = [p[1] for p in best_corners]
        meas_w = max(xs) - min(xs)
        meas_h = max(ys) - min(ys)

        # 移除卡尔曼滤波与门限拦截，直接赋予输出
        self.target_center = (int(meas_cx), int(meas_cy))
        
        rx = int(meas_cx - meas_w / 2.0)
        ry = int(meas_cy - meas_h / 2.0)
        self.target_rect = (rx, ry, int(meas_w), int(meas_h))
        self.search_anchor_rect = self.target_rect

        # 同时平滑四角，确保你后面的 Homography 透视映射矩阵极其丝滑
        self.target_corners = self._sort_corners_fixed(best_corners)

        self.last_target_center = self.target_center
        self.last_target_corners = self.target_corners
        
        return self.target_found, self.target_rect, self.target_center

# ==========================================
# 系统与控制逻辑
# ==========================================
class YawSearchController:
    def __init__(self, enabled=True, start_delay_ms=300, search_freq_hz=500,
                 direction=-1, half_turn_steps=1600, reverse_pause_ms=120):
        self.enabled = bool(enabled)
        self.start_delay_ms = max(0, int(start_delay_ms))
        self.search_freq_hz = abs(float(search_freq_hz))
        self.base_direction = 1 if direction >= 0 else -1
        self.direction = self.base_direction
        self.sweep_direction = self.base_direction
        self.half_turn_steps = max(1, int(half_turn_steps))
        self.reverse_pause_ms = max(0, int(reverse_pause_ms))
        self._missing_since_ms = None
        self._sweep_started_ms = None
        self._phase_started_ms = None
        self._phase = "idle"
        self._active = False

    def reset(self):
        was_active = self._active
        self._missing_since_ms = None
        self._sweep_started_ms = None
        self._phase_started_ms = None
        self.direction = self.base_direction
        self.sweep_direction = self.base_direction
        self._phase = "idle"
        self._active = False
        return was_active

    def is_searching(self):
        return self._missing_since_ms is not None

    def _half_turn_ms(self):
        return max(1, int(self.half_turn_steps * 1000.0 / max(1.0, self.search_freq_hz)))

    def _start_sweep(self, now_ms):
        self._active = True
        self._phase = "sweep"
        self.direction = self.sweep_direction
        self._sweep_started_ms = now_ms
        self._phase_started_ms = now_ms

    def _drive_search_motor(self, motor, command_hz):
        if hasattr(motor, "drive_velocity"):
            motor.drive_velocity(command_hz, 0.0, allow_drive=True)
            return

        x_axis = getattr(motor, "x_axis", None)
        y_axis = getattr(motor, "y_axis", None)
        if y_axis is not None:
            y_axis.stop()
        if x_axis is None:
            motor.drive(command_hz, 0.0, allow_drive=True)
            return

        if (not getattr(x_axis, "ready", False)) or (not getattr(x_axis, "output_enabled", True)):
            x_axis.stop()
            return

        signed_freq = float(command_hz) * float(getattr(x_axis, "command_sign", 1))
        target_freq = abs(signed_freq)
        if target_freq <= 0.0:
            x_axis.stop()
            return

        if hasattr(x_axis, "_reset_pid"):
            x_axis._reset_pid()
        x_axis._last_update_ms = time.ticks_ms()
        x_axis._current_freq = target_freq
        x_axis._write_enable(True)
        x_axis._set_direction(signed_freq >= 0.0)
        x_axis._set_pwm(target_freq, getattr(x_axis, "step_duty", 50))

    def update(self, motor, allow_drive=True):
        if not allow_drive:
            motor.stop()
            self.reset()
            return "CONTROL DISABLED -> HOLD"

        if not self.enabled:
            motor.stop()
            self.reset()
            return "NO RECT -> MOTOR HOLD"

        now_ms = time.ticks_ms()
        if self._missing_since_ms is None:
            self._missing_since_ms = now_ms
            motor.stop()
            return "NO RECT -> YAW SEARCH WAIT"

        if (not self._active) and time.ticks_diff(now_ms, self._missing_since_ms) < self.start_delay_ms:
            motor.stop()
            return "NO RECT -> YAW SEARCH WAIT"

        if not self._active:
            self._start_sweep(now_ms)

        if self._phase == "pause":
            if time.ticks_diff(now_ms, self._phase_started_ms) < self.reverse_pause_ms:
                motor.stop()
                return "YAW SEARCH AT ORIGIN"
            self._start_sweep(now_ms)

        if self._phase_started_ms is None:
            self._phase_started_ms = now_ms

        if time.ticks_diff(now_ms, self._phase_started_ms) >= self._half_turn_ms():
            motor.stop()
            if self._phase == "sweep":
                self._phase = "return"
                self.direction = -self.sweep_direction
                self._phase_started_ms = now_ms
                return "YAW SEARCH RETURN ORIGIN"
            self.sweep_direction = -self.sweep_direction
            self.direction = self.sweep_direction
            self._phase = "pause"
            self._phase_started_ms = now_ms
            return "YAW SEARCH AT ORIGIN"

        command = self.search_freq_hz * self.direction
        self._drive_search_motor(motor, command)
        if self._phase == "return":
            return "YAW SEARCH RETURN ORIGIN"
        return "YAW SEARCH CCW" if self.direction < 0 else "YAW SEARCH CW"


class RectCenterSystem:
    def __init__(self):
        self.tracker = RectTracker()
        self.motor = build_stepper_controller(STEPPER_AXIS_OVERRIDES)
        self.control_started = AUTO_START and MOTOR_CONTROL_ENABLED
        if not MOTOR_CONTROL_ENABLED:
            self.motor.disable()
            print("[Motor] DISABLED: driver released for vision tuning")
        self.start_button = StartButton(START_BUTTON_BOARD_PIN, START_BUTTON_GPIO_NUM)
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_time = time.ticks_ms()
        self.gc_counter = 0
        self.last_aligned = False
        self._aligned_latched = False
        self._control_state = None
        self.filtered_control_center = None
        self.filtered_control_corners = None
        self.filtered_dx = None
        self.filtered_dy = None
        self._drive_filter_active = False
        self.pitch_search = PitchSearchController(
            enabled=PITCH_SEARCH_ENABLED,
            start_delay_ms=PITCH_SEARCH_START_DELAY_MS,
            search_error=PITCH_SEARCH_ERROR_CM,
            segments=PITCH_SEARCH_SEGMENTS,
        )
        self.yaw_search = YawSearchController(
            enabled=YAW_SEARCH_ENABLED,
            start_delay_ms=YAW_SEARCH_START_DELAY_MS,
            search_freq_hz=YAW_SEARCH_FREQ_HZ,
            direction=YAW_SEARCH_DIRECTION,
            half_turn_steps=YAW_SEARCH_HALF_TURN_STEPS,
            reverse_pause_ms=YAW_SEARCH_REVERSE_PAUSE_MS,
        )

    def _report_control_state(self, state, dx=None, dy=None):
        if state == self._control_state: return
        self._control_state = state
        if dx is None or dy is None:
            print("[Aim] {}".format(state))
        else:
            print("[Aim] {} dx={:.2f}cm dy={:.2f}cm".format(state, dx, dy))

    def _update_start_button(self):
        if self.control_started or not MOTOR_CONTROL_ENABLED: return
        if self.start_button.poll_pressed():
            self.control_started = True
            print("[Motor] start button pressed, stepper control enabled")

    def _compute_aim_error(self, rect, corners, target_point, fallback_center):
        if corners is not None and len(corners) == 4:
            try:
                ordered = normalize_corners(corners)
                plane = ((0.0, 0.0), (TARGET_WIDTH_CM, 0.0),
                         (TARGET_WIDTH_CM, TARGET_HEIGHT_CM), (0.0, TARGET_HEIGHT_CM))
                image_to_plane = compute_homography(ordered, plane)
                projected = apply_homography(image_to_plane, target_point[0], target_point[1])
                if projected is not None:
                    px, py = projected
                    margin = max(TARGET_WIDTH_CM, TARGET_HEIGHT_CM)
                    if (-margin <= px <= TARGET_WIDTH_CM + margin and -margin <= py <= TARGET_HEIGHT_CM + margin):
                        return (TARGET_WIDTH_CM * 0.5 - px, TARGET_HEIGHT_CM * 0.5 - py)
            except Exception:
                pass

        _rx, _ry, rw, rh = rect
        px_per_cm_x = rw / TARGET_WIDTH_CM if rw > 0 else 4.0
        px_per_cm_y = rh / TARGET_HEIGHT_CM if rh > 0 else 4.0
        dx_px = fallback_center[0] - target_point[0]
        dy_px = fallback_center[1] - target_point[1]
        return (dx_px / px_per_cm_x, dy_px / px_per_cm_y)

    def _filter_control_error(self, dx, dy):
        if self.filtered_dx is None or self.filtered_dy is None:
            self.filtered_dx = dx
            self.filtered_dy = dy
            return dx, dy

        delta_x = dx - self.filtered_dx
        delta_y = dy - self.filtered_dy
        if delta_x * delta_x + delta_y * delta_y > CONTROL_ERROR_RESET_CM * CONTROL_ERROR_RESET_CM:
            self.filtered_dx = dx
            self.filtered_dy = dy
            return dx, dy

        alpha = CONTROL_ERROR_ALPHA_DRIVE if self._drive_filter_active else CONTROL_ERROR_ALPHA_IDLE
        self.filtered_dx = self.filtered_dx * (1.0 - alpha) + dx * alpha
        self.filtered_dy = self.filtered_dy * (1.0 - alpha) + dy * alpha
        return self.filtered_dx, self.filtered_dy

    def _smooth_control_center(self, center):
        if center is None:
            return None
        alpha = CONTROL_CENTER_ALPHA_DRIVE if self._drive_filter_active else CONTROL_CENTER_ALPHA_IDLE
        self.filtered_control_center = smooth_center(
            center,
            self.filtered_control_center,
            alpha,
            CONTROL_FILTER_RESET_PX,
            CONTROL_FILTER_STICKY_PX,
        )
        return self.filtered_control_center

    def _smooth_control_corners(self, corners):
        if corners is None or len(corners) != 4:
            return None

        current = normalize_corners(corners)
        if self.filtered_control_corners is None or len(self.filtered_control_corners) != 4:
            self.filtered_control_corners = tuple((int(p[0]), int(p[1])) for p in current)
            return self.filtered_control_corners

        alpha = CONTROL_CORNER_ALPHA_DRIVE if self._drive_filter_active else CONTROL_CORNER_ALPHA_IDLE
        previous = normalize_corners(self.filtered_control_corners)
        reset_sq = CONTROL_FILTER_RESET_PX * CONTROL_FILTER_RESET_PX
        smoothed = []
        for idx in range(4):
            prev_x, prev_y = previous[idx]
            curr_x, curr_y = current[idx]
            dx = curr_x - prev_x
            dy = curr_y - prev_y
            if dx * dx + dy * dy > reset_sq:
                smoothed.append((int(curr_x), int(curr_y)))
                continue

            limited_x = prev_x + max(-TARGET_CORNER_STEP_LIMIT_PX, min(TARGET_CORNER_STEP_LIMIT_PX, dx))
            limited_y = prev_y + max(-TARGET_CORNER_STEP_LIMIT_PX, min(TARGET_CORNER_STEP_LIMIT_PX, dy))
            smoothed.append((
                int(prev_x * (1.0 - alpha) + limited_x * alpha),
                int(prev_y * (1.0 - alpha) + limited_y * alpha),
            ))

        self.filtered_control_corners = tuple(smoothed)
        return self.filtered_control_corners

    def process_frame(self, img):
        self.frame_count += 1
        self.gc_counter += 1
        self._update_start_button()

        force_global_search = YAW_SEARCH_ENABLED and self.yaw_search.is_searching()
        found, rect, center = self.tracker.detect(img, force_global=force_global_search)
        screen_center = (FRAME_WIDTH // 2, FRAME_HEIGHT // 2)

        if not found or rect is None or center is None:
            self.last_aligned = False
            self._aligned_latched = False
            self.filtered_control_center = None
            self.filtered_control_corners = None
            self.filtered_dx = None
            self.filtered_dy = None
            self._drive_filter_active = False
            if YAW_SEARCH_ENABLED:
                search_state = self.yaw_search.update(self.motor, allow_drive=self.control_started)
            else:
                search_state = self.pitch_search.update(self.motor, allow_drive=self.control_started)
            self._report_control_state(search_state)
            if DEBUG_MODE:
                self._draw_overlay(img, None, None, screen_center,
                                   (LASER_DOT_X_PX, LASER_DOT_Y_PX), False, 8, status_text=search_state)
            return img

        search_was_active = self.yaw_search.reset()
        pitch_search_was_active = self.pitch_search.reset()
        if search_was_active or pitch_search_was_active:
            self.motor.stop()
            if search_was_active:
                self._report_control_state("TARGET FOUND -> SEARCH STOP")
                if DEBUG_MODE:
                    self._draw_overlay(img, rect, center, screen_center,
                                       (LASER_DOT_X_PX, LASER_DOT_Y_PX), False, 8,
                                       target_corners=self.tracker.target_corners)
                return img

        _rx, _ry, rw, rh = rect
        px_per_cm_x = rw / TARGET_WIDTH_CM if rw > 0 else 4.0
        px_per_cm_y = rh / TARGET_HEIGHT_CM if rh > 0 else 4.0

        target_point_px = (LASER_DOT_X_PX, LASER_DOT_Y_PX)
        target_fresh = self.tracker.target_fresh
        dx, dy = self._compute_aim_error(rect, self.tracker.target_corners, target_point_px, center)

        dx = max(-MAX_AIM_ERROR_CM, min(MAX_AIM_ERROR_CM, dx))
        dy = max(-MAX_AIM_ERROR_CM, min(MAX_AIM_ERROR_CM, dy))
        self._drive_filter_active = (
            self.control_started
            and target_fresh
            and (abs(dx) > CONTROL_FILTER_DRIVE_ERROR_CM or abs(dy) > CONTROL_FILTER_DRIVE_ERROR_CM)
        )
        dx, dy = self._filter_control_error(dx, dy)

        tolerance_px = int(ALIGNED_TOLERANCE_CM * max(px_per_cm_x, px_per_cm_y))
        aligned = (target_fresh and abs(dx) <= ALIGNED_TOLERANCE_CM and abs(dy) <= ALIGNED_TOLERANCE_CM)
        self.last_aligned = aligned
        
        self.motor.drive(dx, dy, allow_drive=self.control_started and target_fresh and (not aligned))
        self._aligned_latched = aligned
        
        if not self.control_started: self._report_control_state("CONTROL DISABLED", dx, dy)
        elif not target_fresh: self._report_control_state("TRACK HOLD", dx, dy)
        elif aligned: self._report_control_state("ALIGNED -> MOTOR HOLD", dx, dy)
        else: self._report_control_state("DRIVING", dx, dy)

        if DEBUG_MODE:
            self._draw_overlay(img, rect, center, screen_center, target_point_px,
                               aligned, tolerance_px, dx, dy, target_corners=self.tracker.target_corners)
        return img

    def _draw_overlay(self, img, rect, center, screen_center, target_point,
                       aligned, tolerance_px=8, dx_cm=0.0, dy_cm=0.0,
                       target_corners=None, status_text=None):
        scx, scy = screen_center
        tx, ty = target_point
        img.draw_cross(scx, scy, color=(120, 120, 120), size=8, thickness=1)
        img.draw_line(scx, 0, scx, FRAME_HEIGHT - 1, color=(80, 80, 80), thickness=1)
        img.draw_cross(tx, ty, color=(255, 255, 0), size=10, thickness=2)
        img.draw_circle(tx, ty, tolerance_px, color=(255, 255, 0), thickness=1)

        if rect is not None:
            x, y, w, h = rect
            color = (0, 255, 0) if aligned else (0, 180, 255)
            if target_corners is not None and len(target_corners) == 4:
                points = target_corners
                for idx in range(4):
                    p0 = points[idx]
                    p1 = points[(idx + 1) % 4]
                    img.draw_line(int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1]), color=color, thickness=2)
            else:
                img.draw_rectangle(x, y, w, h, color=color, thickness=2)
                
        if center is not None:
            cx, cy = center
            color = (0, 255, 0) if aligned else (255, 0, 0)
            img.draw_cross(cx, cy, color=color, size=8, thickness=2)
            img.draw_line(int(cx), int(cy), tx, ty, color=(255, 255, 255), thickness=1)

        if DEBUG_TEXT_OVERLAY and center is not None:
            draw_text(img, 4, 4, "dx={:.1f}cm dy={:.1f}cm".format(dx_cm, dy_cm))
            draw_text(img, 4, 22, "fps={:.1f}".format(self.fps))
            draw_text(img, 4, 40, "aligned={}".format(1 if aligned else 0))
            draw_text(img, 4, 58, "Laser X/Y: {},{}".format(tx, ty))
        elif DEBUG_TEXT_OVERLAY:
            draw_text(img, 4, 4, status_text or "NO RECT - SEARCH WAIT", color=(255, 255, 0), scale=1)

        if not self.control_started:
            draw_text(img, 4, FRAME_HEIGHT - 18, "PRESS GPIO28 TO START", color=(255, 255, 0), scale=1)

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

# ==========================================
# 主运行入口
# ==========================================
def main():
    sensor = None
    try:
        os.exitpoint(os.EXITPOINT_ENABLE)
    except Exception:
        pass

    print("=" * 50)
    print("K230 Rect Center Mode - FINAL TUNED")
    print("build:", BUILD_TAG)
    print("=" * 50)

    try:
        from machine import FPIOA
        from common_hw import map_gpio, map_pwm
        fpioa = FPIOA()
        map_gpio(fpioa, 43, 43)
        map_gpio(fpioa, 27, 27)
        map_pwm(fpioa, 42, 0)
        map_gpio(fpioa, 53, 53)
        map_gpio(fpioa, 35, 35)
        map_pwm(fpioa, 52, 4)
        print("[System] Stepper pins mapped successfully.")
    except Exception as e:
        print("[System] Stepper pin mapping warning:", e)

    system = RectCenterSystem()

    print("[Display] init...")
    display_init(FRAME_WIDTH, FRAME_HEIGHT)
    kw = dict(camera_id=CAMERA_ID, width=FRAME_WIDTH, height=FRAME_HEIGHT,
              hmirror=SENSOR_HMIRROR, vflip=SENSOR_VFLIP)

    try:
        print("[Sensor] init...")
        sensor = camera_init(CAMERA_ID)
        camera_start(sensor, **kw)
        
        try:
            sensor.set_auto_gain(False)
            sensor.set_auto_whitebal(False)
            sensor.set_auto_exposure(False, exposure_us=15000)
            print("[Sensor] 固定曝光成功锁死！")
        except Exception as e:
            print("[Sensor] 锁曝光跳过:", e)
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

            os.exitpoint()
            img = system.process_frame(img)
            os.exitpoint()
            system.update_fps()

            if hasattr(Display, 'show_image'):
                Display.show_image(img)
            else:
                Display.show(img)

            os.exitpoint()
            system.maybe_collect_gc()
            time.sleep_ms(FRAME_LOOP_DELAY_MS)
            
    except KeyboardInterrupt:
        print("\n[System] interrupted")
    except Exception as e:
        print("[Error]", e)
        sys.print_exception(e)
    finally:
        print("[System] cleanup...")
        try:
            system.motor.stop()
        except Exception:
            pass
        camera_deinit(sensor)
        system.motor.deinit()
        print("[System] stopped")

if __name__ == "__main__":
    main()
