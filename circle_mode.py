import gc
import math
import os
import sys
import time

from common_hw import (DebouncedButton as StartButton, Display, draw_text,
                        camera_init, camera_start, camera_snapshot,
                        camera_restart, camera_deinit,
                        display_init)
from vision_utils import (CircleState, clamp_point, clamp_rect, dist_sq,
                           smooth_center, smooth_scalar, apply_motion_lead,
                           push_point_history, filter_point_history,
                           push_scalar_history, filter_scalar_history,
                           rect_aspect_error, rect_center_from_corners,
                           rect_size_change_ok, compensate_edge_rect,
                           rect_overlap_ratio, rect_border_hit_ratio,
                           expand_rect, compute_homography, apply_homography,
                           normalize_corners, plane_size_cm_for_corners,
                           log_info, log_warn, log_error)

try:
    from k230_common import build_stepper_controller, load_calibration
except ImportError:
    def build_stepper_controller(axis_overrides=None):
        class _NoopStepperController:
            ready = False

            def drive(self, error_x, error_y, allow_drive=True):
                return

            def stop(self):
                return

            def deinit(self):
                return

        return _NoopStepperController()

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

START_BUTTON_BOARD_PIN = 28
START_BUTTON_GPIO_NUM = 28

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
FRAME_LOOP_DELAY_MS = 0
OVERLAY_RING_SAMPLE_COUNT = 18
GC_FRAME_INTERVAL = 180
BUILD_TAG = "2026-07-14-circle-fastboot-v31"

# ── Testing without button board ──────────────────────────────────
# Set True to skip GPIO28 start-button wait.
AUTO_START = True
ERROR_JUMP_MAX_CM = 3.0
ERROR_JUMP_REJECT_FRAMES = 2

STEPPER_AXIS_OVERRIDES = {
    "x": {
        "deadband": 0.20,
        "error_full_scale": 4.0,
        "command_sign": 1,
        "pid_kp": 220.0,
        "pid_ki": 12.0,
        "pid_kd": 3.0,
        "integral_limit": 5.0,
        "integral_active_error": 2.2,
    },
    "y": {
        "deadband": 0.20,
        "error_full_scale": 4.0,
        "command_sign": 1,
        "pid_kp": 220.0,
        "pid_ki": 12.0,
        "pid_kd": 3.0,
        "integral_limit": 5.0,
        "integral_active_error": 2.2,
    },
}


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

    # geometry/coordinate helpers from vision_utils:
    #   clamp_point, clamp_rect, dist_sq, smooth_center, smooth_scalar,
    #   apply_motion_lead, rect_aspect_error, rect_center_from_corners,
    #   rect_size_change_ok, compensate_edge_rect, rect_overlap_ratio,
    #   rect_border_hit_ratio, expand_rect, compute_homography,
    #   apply_homography, normalize_corners, plane_size_cm_for_corners,
    #   push_point_history, filter_point_history,
    #   push_scalar_history, filter_scalar_history

    def _clamp_point(self, point):
        return clamp_point(point, FRAME_WIDTH, FRAME_HEIGHT)

    def target_plane_cm_to_image(self, dx_cm, dy_cm):
        if self.target_to_image_h is None:
            return None
        projected = apply_homography(self.target_to_image_h, dx_cm, dy_cm)
        if projected is None:
            return None
        return self._clamp_point(projected)

    def _point_to_target_plane_cm(self, point):
        if self.image_to_target_h is None:
            return None
        projected = apply_homography(self.image_to_target_h, point[0], point[1])
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

    # ── methods that need self state (not in vision_utils) ──────────

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


    def _select_best_rect(self, rect_img, rects, reference_center, reference_rect):
        best = None
        best_score = None
        image_center = (FRAME_WIDTH // 2, FRAME_HEIGHT // 2)

        for r in rects:
            raw_rect = r.rect()
            corners = r.corners()
            if raw_rect is None or corners is None or len(corners) != 4:
                continue
            rect = compensate_edge_rect(
                raw_rect,
                reference_rect,
                TARGET_EDGE_MARGIN_PX,
                TARGET_EDGE_COMP_MIN_RATIO,
                FRAME_WIDTH,
                FRAME_HEIGHT,
            )
            x, y, w, h = rect
            if w < TARGET_MIN_W or h < TARGET_MIN_H:
                continue
            area = w * h
            if area < TARGET_MIN_AREA:
                continue
            if not rect_size_change_ok(rect, reference_rect, TARGET_MAX_SIZE_CHANGE_RATIO):
                continue

            center = rect_center_from_corners(corners, FRAME_WIDTH, FRAME_HEIGHT)
            border_hit_ratio = rect_border_hit_ratio(
                rect_img,
                rect,
                TARGET_BORDER_SAMPLE_COUNT,
                corners,
            )
            if border_hit_ratio < TARGET_BORDER_HIT_RATIO_MIN:
                continue
            if reference_center is not None:
                jump_sq = dist_sq(center, reference_center)
                if jump_sq > (TARGET_RESET_DIST_PX * TARGET_RESET_DIST_PX):
                    continue
            if reference_rect is not None:
                overlap_ratio = rect_overlap_ratio(rect, reference_rect)
                if overlap_ratio < TARGET_MIN_OVERLAP_RATIO and (
                    reference_center is None
                    or dist_sq(center, reference_center) > (TARGET_STICKY_DIST_PX * TARGET_STICKY_DIST_PX)
                ):
                    continue
            aspect_penalty = int(
                rect_aspect_error(w, h, TARGET_ASPECT) * TARGET_ASPECT_PENALTY_SCALE
            )
            if reference_center is not None:
                distance_penalty = dist_sq(center, reference_center) // 10
            else:
                distance_penalty = dist_sq(center, image_center) // TARGET_INIT_CENTER_BIAS
            edge_penalty = 0
            if x <= 2 or y <= 2 or (x + w) >= (FRAME_WIDTH - 2) or (y + h) >= (FRAME_HEIGHT - 2):
                edge_penalty = 3600
            center_bias_bonus = 0
            if dist_sq(center, image_center) <= (TARGET_NEAR_CENTER_PX * TARGET_NEAR_CENTER_PX):
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
        return dist_sq(candidate_center, last_center) <= (TARGET_MAX_JUMP_PX * TARGET_MAX_JUMP_PX)

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
            distance_sq = dist_sq((bx, by), self.target_center)
            if distance_sq > gate_sq:
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
        self.target_center = smooth_center(
            center,
            previous_center,
            TARGET_CENTER_ALPHA,
            TARGET_RESET_DIST_PX,
            TARGET_STICKY_DIST_PX,
        )
        self.target_center = apply_motion_lead(
            self.target_center,
            previous_center,
            TARGET_LEAD_GAIN,
            TARGET_LEAD_MAX_PX,
            FRAME_WIDTH,
            FRAME_HEIGHT,
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
        self.bullseye_center = smooth_center(
            refined_center,
            self.last_bullseye_center,
            BULLSEYE_CENTER_ALPHA,
            TARGET_RESET_DIST_PX,
            TARGET_STICKY_DIST_PX,
        )
        self.bullseye_center = apply_motion_lead(
            self.bullseye_center,
            self.last_bullseye_center,
            BULLSEYE_LEAD_GAIN,
            BULLSEYE_LEAD_MAX_PX,
            FRAME_WIDTH,
            FRAME_HEIGHT,
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
            roi = expand_rect(roi, margin, FRAME_WIDTH, FRAME_HEIGHT)

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
                if dist_sq((blob.cx(), blob.cy()), reference) <= gate_sq
            ]
            if not candidate_blobs:
                candidate_blobs = blobs
            best = min(
                candidate_blobs,
                key=lambda b: (
                    dist_sq((b.cx(), b.cy()), reference),
                    -b.density(),
                ),
            )
            raw_spot = (best.cx(), best.cy())
            if self.last_laser_spot is None:
                self.laser_spot = raw_spot
            else:
                self.laser_spot = smooth_center(
                    raw_spot,
                    self.last_laser_spot,
                    0.9,
                    LASER_STICKY_PX * 3,
                    LASER_STICKY_PX,
                )
            push_point_history(self.laser_spot_history, self.laser_spot, LASER_POINT_HISTORY_LEN)
            self.laser_spot = filter_point_history(self.laser_spot_history)
            self.laser_found = True
            self.last_laser_spot = self.laser_spot
            self.laser_miss_count = 0
        else:
            self.laser_miss_count += 1
            if self.last_laser_spot is not None and self.laser_miss_count <= LASER_MAX_MISS_FRAMES:
                self.laser_spot = self.last_laser_spot
                push_point_history(self.laser_spot_history, self.laser_spot, LASER_POINT_HISTORY_LEN)
                self.laser_spot = filter_point_history(self.laser_spot_history)
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
            detected_radius = smooth_scalar(
                detected_radius,
                self.last_detected_ring_radius_cm,
                CIRCLE_RADIUS_SMOOTHING_ALPHA,
            )
        push_scalar_history(self.ring_radius_cm_history, detected_radius)
        detected_radius = filter_scalar_history(self.ring_radius_cm_history)
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
        ordered_corners = normalize_corners(corners)
        width_cm, height_cm = plane_size_cm_for_corners(
            ordered_corners,
            TARGET_ASPECT,
            TARGET_WIDTH_CM,
            TARGET_HEIGHT_CM,
        )
        plane_corners = (
            (-width_cm * 0.5, height_cm * 0.5),
            (width_cm * 0.5, height_cm * 0.5),
            (width_cm * 0.5, -height_cm * 0.5),
            (-width_cm * 0.5, -height_cm * 0.5),
        )
        self.target_plane_corners_cm = plane_corners
        self.target_to_image_h = compute_homography(plane_corners, ordered_corners)
        self.image_to_target_h = compute_homography(ordered_corners, plane_corners)
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
        self.circle_radius_px = smooth_scalar(
            radius_px,
            self.last_circle_radius_px,
            CIRCLE_RADIUS_SMOOTHING_ALPHA,
        )
        push_scalar_history(
            self.circle_radius_history,
            self.circle_radius_px,
        )
        self.circle_radius_px = filter_scalar_history(
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
        self.motor = build_stepper_controller(STEPPER_AXIS_OVERRIDES)
        self.control_started = AUTO_START
        self.start_button = StartButton(START_BUTTON_BOARD_PIN, START_BUTTON_GPIO_NUM)
        self.frame_count = 0
        self.gc_counter = 0
        self.state = CircleState.IDLE
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
        if self.control_started:
            return
        if self.start_button.poll_pressed():
            self.control_started = True
            print("[Motor] start button pressed, stepper control enabled")

    def process_frame(self, img):
        self.frame_count += 1
        self.gc_counter += 1
        self.detector.detect_all(img)
        self._update_start_button()

        if self.state == CircleState.IDLE:
            self.motor.stop()
            if self.detector.target_found and self.detector.bullseye_found:
                self.state = CircleState.WAITING
                self.start_align_frames = 0
                self.tracker.reset()
                print("[State] target ready -> WAITING")
        elif self.state == CircleState.WAITING:
            self.motor.stop()
            if self._check_start_alignment():
                self.state = CircleState.RUNNING
                self.tracker.reset()
                self.tracker.start()
                self.start_align_frames = 0
                print("[State] laser aligned with start point -> RUNNING")
        elif self.state == CircleState.RUNNING:
            if not self.control_started:
                self.state = CircleState.WAITING
                self.tracker.reset()
                self.start_align_frames = 0
                self.last_sent_error = None
                self.error_jump_count = 0
                self.motor.stop()
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
            self.motor.drive(
                filtered_x,
                filtered_y,
                allow_drive=self.control_started and sync_ok and stable_ok,
            )
        else:
            self.last_sent_error = None
            self.error_jump_count = 0
            self.motor.stop()

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
                CircleState.IDLE: (128, 128, 128),
                CircleState.WAITING: (255, 255, 0),
                CircleState.RUNNING: (0, 255, 0),
                "COMPLETE": (0, 255, 255),
                "ERROR": (255, 0, 0),
            }
            color = state_colors.get(self.state, (255, 255, 255))
            draw_text(img, 10, 10, "State: " + CircleState.name(self.state), color=color, scale=2)

            if self.state == CircleState.WAITING:
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

            if self.state == CircleState.RUNNING:
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
        if not self.control_started:
            draw_text(
                img,
                10,
                FRAME_HEIGHT - 20,
                "PRESS GPIO28 TO START MOTOR",
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

    display_init(FRAME_WIDTH, FRAME_HEIGHT)
    kw = dict(camera_id=CAMERA_ID, width=FRAME_WIDTH, height=FRAME_HEIGHT,
              hmirror=SENSOR_HMIRROR, vflip=SENSOR_VFLIP)
    sensor = camera_init(CAMERA_ID)
    camera_start(sensor, **kw)

    system = CircleModeSystem()
    print("system ready")

    try:
        while True:
            os.exitpoint()
            try:
                img = camera_snapshot(sensor)
            except RuntimeError as e:
                print("[Sensor] snapshot failed, retry:", e)
                sensor = camera_restart(sensor, **kw)
                img = camera_snapshot(sensor)
            img = system.process_frame(img)
            Display.show_image(img)
            time.sleep_ms(FRAME_LOOP_DELAY_MS)

            if system.gc_counter >= GC_FRAME_INTERVAL:
                gc.collect()
                system.gc_counter = 0
    except KeyboardInterrupt:
        print("\nuser interrupted")
    except Exception as e:
        print("error:", e)
        sys.print_exception(e)
    finally:
        camera_deinit(sensor)
        system.motor.deinit()
        print("system stopped")


if __name__ == "__main__":
    main()
