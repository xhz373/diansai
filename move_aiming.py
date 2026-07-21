import gc
import math
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
    from common_hw import (DebouncedButton as StartButton, Display, draw_text,
                            camera_init, camera_start, camera_snapshot,
                            camera_restart, camera_deinit,
                            display_init)
except ImportError:
    raise ImportError(
        "common_hw not found; copy common_hw.py to the same folder as move_aiming.py, "
        "/flash, /flash/app, /flash/lib, /sdcard, /sdcard/app, or /sdcard/lib before running."
    )

try:
    from vision_utils import (clamp_rect, dist_sq, smooth_center, smooth_scalar,
                               apply_motion_lead, push_point_history, filter_point_history,
                               push_scalar_history, filter_scalar_history,
                               rect_aspect_error, rect_center_from_corners,
                               rect_size_change_ok, compensate_edge_rect,
                               rect_overlap_ratio, rect_border_hit_ratio,
                               expand_rect, compute_homography, apply_homography,
                               normalize_corners, plane_size_cm_for_corners,
                               log_info)
except ImportError:
    raise ImportError(
        "vision_utils not found; copy vision_utils.py with move_aiming.py dependencies "
        "to /flash, /flash/app, /sdcard, or /sdcard/app before running."
    )

try:
    from k230_common import build_stepper_controller, load_calibration
except ImportError:
    def build_stepper_controller(axis_overrides=None):
        class _NoopStepperController:
            ready = False
            def drive(self, *a, **kw): pass
            def stop(self): pass
            def deinit(self): pass
        return _NoopStepperController()

    def load_calibration(*a):
        return (False,) + tuple(a[1:])


CAMERA_ID = 2
FRAME_WIDTH = 400
FRAME_HEIGHT = 300
SENSOR_HMIRROR = True
SENSOR_VFLIP = True
TARGET_DETECT_INTERVAL = 1
BULLSEYE_DETECT_INTERVAL = 1
GC_INTERVAL = 180

TARGET_WIDTH_CM = 25.0
TARGET_HEIGHT_CM = 29.7
CIRCLE_RADIUS_CM = 6.0
TARGET_OUTER_DIAMETER_CM = 21.0

RED_THRESHOLD = (41, 100, -28, 6, -14, 14)
BLACK_THRESHOLD = (22, 69, -23, -3, -22, 16)
VIOLET_THRESHOLD = (92, 100, -15, 6, -9, 11)

RECT_THRESHOLD = 8000
RECT_BINARY_THRESHOLD = (0, 72)
RECT_TRACK_THRESHOLD = 25000
RECT_GLOBAL_THRESHOLD = 20000
RECT_TRACK_MAX_REGIONS = 2
RECT_REACQUIRE_MAX_REGIONS = 4
RECT_TRACK_PAD_PX = 30
RECT_REACQUIRE_PAD_PX = 80
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

START_BUTTON_BOARD_PIN = 28
START_BUTTON_GPIO_NUM = 28
DEBUG_MODE = True
DEBUG_TEXT_OVERLAY = False
FRAME_LOOP_DELAY_MS = 0
MAX_CONSECUTIVE_SNAPSHOT_FAILURES = 5
TARGET_MAX_MISS_FRAMES = 6
TARGET_REACQUIRE_FRAMES = 24
TARGET_REACQUIRE_GLOBAL_AFTER = 4
LASER_MAX_MISS_FRAMES = 2
LASER_STICKY_PX = 2
PREDICT_TRACK_MAX_FRAMES = 30
LASER_PREDICT_ROI_MARGIN_PX = 24
CIRCLE_RADIUS_SMOOTHING_ALPHA = 0.45
TARGET_POINT_HISTORY_LEN = 3
LASER_POINT_HISTORY_LEN = 1
RADIUS_HISTORY_LEN = 3
VISUAL_RATE_FILTER_ALPHA = 0.45
TURN_BOOST_ENTER_RATE_CM_S = 18.0
TURN_BOOST_EXIT_RATE_CM_S = 8.0
TURN_BOOST_MIN_FRAMES = 2
TURN_BOOST_EXIT_FRAMES = 3
TURN_BOOST_GROWTH_DELTA_CM = 0.35
TURN_BOOST_EXIT_ERROR_CM = 1.2
YAW_LEAD_GAIN_NORMAL = 0.035
YAW_LEAD_GAIN_TURN = 0.075
YAW_LEAD_MAX_CM = 2.0
LOSS_HOLD_MS = 280
LOSS_REACQUIRE_MS = 1600
LOSS_REACQUIRE_YAW_CMD_CM = 6.0
LOSS_REACQUIRE_PITCH_HOLD_RATIO = 0.35
HOLD_COMMAND_DECAY = 0.35
PITCH_LEAD_GAIN = 0.0
NORMAL_YAW_CMD_LIMIT_CM = 5.0
TURN_BOOST_YAW_CMD_LIMIT_CM = 7.0
PITCH_CMD_LIMIT_CM = 5.0
BUILD_TAG = "2026-07-20-aim-turnboost-v20"

# ── Testing without button board ──────────────────────────────────
# Set True to skip GPIO28 start-button wait.
AUTO_START = True

STEPPER_AXIS_OVERRIDES = {
    "x": {
        "deadband": 0.25,
        "error_full_scale": 5.0,
        "command_sign": -1,
        "pid_kp": 0.1,
        "pid_ki": 0.0,
        "pid_kd": 0.0,
        "integral_limit": 6.0,
        "integral_active_error": 2.8,
    },
    "y": {
        "deadband": 0.25,
        "error_full_scale": 5.0,
        "command_sign": -1,
        "pid_kp": 0.1,
        "pid_ki": 0.0,
        "pid_kd": 0.0,
        "integral_limit": 6.0,
        "integral_active_error": 2.8,
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
        self.pixel_to_cm_ratio_x = 0.0
        self.pixel_to_cm_ratio_y = 0.0
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
        self.search_anchor_rect = None
        self.last_bullseye_center = None
        self.bullseye_miss_count = 0
        self.last_laser_spot = None
        self.laser_miss_count = 0
        self.circle_radius_px = 0.0
        self.last_circle_radius_px = 0.0
        self.target_center_history = []
        self.bullseye_center_history = []
        self.laser_spot_history = []
        self.circle_radius_history = []
        self.frozen_target_rect = None
        self.frozen_target_center = None
        self.frozen_target_diameter_px = 0
        self.frozen_bullseye_center = None
        self.predict_miss_count = 0
        self.frame_timestamp_ms = 0
        self.target_visual_valid = False
        self.bullseye_visual_valid = False
        self.laser_visual_valid = False
        self.prediction_active = False

    def detect_all(self, img):
        self.frame_timestamp_ms = time.ticks_ms()
        self.target_visual_valid = False
        self.bullseye_visual_valid = False
        self.laser_visual_valid = False
        self.prediction_active = False
        self.frame_id += 1
        self._detect_target(img)
        if self.target_found:
            self._detect_bullseye(img)
            self.calibrate_scale()
            self._detect_laser(img)
            self._refresh_prediction_anchor()
        else:
            if self._prediction_available() and self.predict_miss_count < PREDICT_TRACK_MAX_FRAMES:
                self.prediction_active = True
                self.predict_miss_count += 1
                self.target_rect = self.frozen_target_rect
                self.target_center = self.frozen_target_center
                self.target_diameter_px = self.frozen_target_diameter_px
                self.bullseye_center = self.frozen_bullseye_center
                self.bullseye_found = self.bullseye_plane_cm is not None
                self._detect_laser(img)
            else:
                self.target_center = None
                self.target_diameter_px = 0
                self.bullseye_center = None
                self.laser_spot = None
                self.bullseye_found = False
                self.laser_found = False
    def _clamp_point(self, point):
        return (
            max(0, min(FRAME_WIDTH - 1, int(point[0]))),
            max(0, min(FRAME_HEIGHT - 1, int(point[1]))),
        )
    def _point_to_target_plane_cm(self, point):
        if self.image_to_target_h is None:
            return None
        projected = apply_homography(self.image_to_target_h, point[0], point[1])
        if projected is None:
            return None
        return projected

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
        return clamp_rect(
            x - margin,
            y - margin,
            w + margin * 2,
            h + margin * 2,
            FRAME_WIDTH,
            FRAME_HEIGHT,
        )

    def _target_search_roi(self, reference_rect, target_is_active):
        if reference_rect is None:
            return (40, 30, FRAME_WIDTH - 80, FRAME_HEIGHT - 60)

        x, y, w, h = reference_rect
        pad = RECT_TRACK_PAD_PX if target_is_active else RECT_REACQUIRE_PAD_PX
        return clamp_rect(
            x - pad,
            y - pad,
            w + pad * 2,
            h + pad * 2,
            FRAME_WIDTH,
            FRAME_HEIGHT,
        )

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

        previous_rect = self.last_target_rect or self.search_anchor_rect
        previous_center = self.last_target_center
        previous_diameter = self.last_target_diameter_px
        previous_corners = self.last_target_corners
        self.target_rect = None
        self.target_center = None
        self.target_diameter_px = 0
        self.target_found = False
        self.target_visual_valid = False

        rect_img = self._prepare_rect_image(img)
        search_roi = self._target_search_roi(previous_rect, self.last_target_rect is not None)
        scan_threshold = RECT_TRACK_THRESHOLD if previous_rect is not None else RECT_GLOBAL_THRESHOLD
        scan_max_regions = (
            RECT_TRACK_MAX_REGIONS if previous_rect is not None else RECT_REACQUIRE_MAX_REGIONS
        )
        try:
            rects = rect_img.find_rects(
                roi=search_roi,
                threshold=scan_threshold,
                max_regions=scan_max_regions,
            ) or []
        except (RuntimeError, MemoryError):
            gc.collect()
            rects = []
        chosen = self._select_best_rect(rect_img, rects, previous_center, previous_rect)
        used_global_reacquire = False
        if chosen is None and self.target_miss_count >= TARGET_REACQUIRE_GLOBAL_AFTER:
            global_roi = (40, 30, FRAME_WIDTH - 80, FRAME_HEIGHT - 60)
            if global_roi != search_roi:
                try:
                    global_rects = rect_img.find_rects(
                        roi=global_roi,
                        threshold=RECT_GLOBAL_THRESHOLD,
                        max_regions=RECT_REACQUIRE_MAX_REGIONS,
                    ) or []
                except (RuntimeError, MemoryError):
                    gc.collect()
                    global_rects = []
                chosen = self._select_best_rect(rect_img, global_rects, None, None)
                used_global_reacquire = chosen is not None
        if chosen is not None:
            _, _, center = chosen
            if not used_global_reacquire and not self._accept_center(center, previous_center):
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
                if self.target_miss_count > TARGET_REACQUIRE_FRAMES:
                    self.search_anchor_rect = None
                self.target_center_history = []
                self.bullseye_center_history = []
                self.laser_spot_history = []
                self.circle_radius_history = []
                self.circle_radius_px = 0.0
                self.last_circle_radius_px = 0.0
            return None

        self.target_miss_count = 0
        rect, corners, center = chosen
        self.target_rect = rect
        self.search_anchor_rect = rect
        self.target_center = smooth_center(
            center,
            previous_center,
            TARGET_CENTER_ALPHA,
            TARGET_RESET_DIST_PX,
            TARGET_STICKY_DIST_PX,
        )
        self.target_diameter_px = min(rect[2], rect[3])
        self.target_found = True
        self.target_visual_valid = True
        self.last_target_rect = self.target_rect
        self.last_target_corners = corners
        self.last_target_center = self.target_center
        self.last_target_diameter_px = self.target_diameter_px

        return self.target_rect

    def _detect_bullseye(self, img):
        self.bullseye_found = False
        self.bullseye_center = None
        self.bullseye_visual_valid = False
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
        self.bullseye_visual_valid = True
        self.bullseye_miss_count = 0

    def _detect_laser(self, img):
        self.laser_spot = None
        self.laser_found = False
        self.laser_visual_valid = False
        if self.target_rect is None:
            return None
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
            self.laser_spot = smooth_center(
                raw_spot,
                self.last_laser_spot,
                0.9,
                LASER_STICKY_PX * 3,
                LASER_STICKY_PX,
            )
            push_point_history(
                self.laser_spot_history,
                self.laser_spot,
                LASER_POINT_HISTORY_LEN,
            )
            self.laser_spot = filter_point_history(self.laser_spot_history)
            self.laser_found = True
            self.laser_visual_valid = True
            self.last_laser_spot = self.laser_spot
            self.laser_miss_count = 0
            return self.laser_spot

        self.laser_miss_count += 1
        if self.last_laser_spot is not None and self.laser_miss_count <= LASER_MAX_MISS_FRAMES:
            self.laser_spot = self.last_laser_spot
            push_point_history(
                self.laser_spot_history,
                self.laser_spot,
                LASER_POINT_HISTORY_LEN,
            )
            self.laser_spot = filter_point_history(self.laser_spot_history)
            self.laser_found = True
            return self.laser_spot

        self.laser_spot_history = []
        return None

    def calibrate_scale(self):
        self.pixel_to_cm_ratio_x = 0.0
        self.pixel_to_cm_ratio_y = 0.0
        self.target_to_image_h = None
        self.image_to_target_h = None
        self.bullseye_plane_cm = None
        if not (self.target_found and self.last_target_corners):
            return False

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
            return False

        self.pixel_to_cm_ratio_x = width_cm / width_px
        self.pixel_to_cm_ratio_y = height_cm / height_px

        radius_px = (
            (CIRCLE_RADIUS_CM / self.pixel_to_cm_ratio_x)
            + (CIRCLE_RADIUS_CM / self.pixel_to_cm_ratio_y)
        ) * 0.5
        self.circle_radius_px = smooth_scalar(
            radius_px,
            self.last_circle_radius_px,
            CIRCLE_RADIUS_SMOOTHING_ALPHA,
        )
        push_scalar_history(
            self.circle_radius_history,
            self.circle_radius_px,
            RADIUS_HISTORY_LEN,
        )
        self.circle_radius_px = filter_scalar_history(self.circle_radius_history)
        self.last_circle_radius_px = self.circle_radius_px
        if self.bullseye_found and self.bullseye_center:
            self.bullseye_plane_cm = self._point_to_target_plane_cm(self.bullseye_center)
        return True

    def pixel_offset_to_cm(self, pixel_dx, pixel_dy):
        return (
            pixel_dx * self.pixel_to_cm_ratio_x,
            pixel_dy * self.pixel_to_cm_ratio_y,
        )

    def get_offset_info(self):
        if not self.bullseye_found or not self.laser_found:
            return None

        bx, by = self.bullseye_center
        lx, ly = self.laser_spot
        laser_plane_cm = self._point_to_target_plane_cm(self.laser_spot)
        if self.bullseye_plane_cm is not None and laser_plane_cm is not None:
            dx_cm = laser_plane_cm[0] - self.bullseye_plane_cm[0]
            dy_cm = laser_plane_cm[1] - self.bullseye_plane_cm[1]
        else:
            dx_cm, dy_cm = self.pixel_offset_to_cm(lx - bx, ly - by)
        distance_cm = math.sqrt(dx_cm * dx_cm + dy_cm * dy_cm)
        angle_deg = math.degrees(math.atan2(-dy_cm, dx_cm))
        return {
            "timestamp_ms": self.frame_timestamp_ms,
            "dx_cm": dx_cm,
            "dy_cm": dy_cm,
            "distance_cm": distance_cm,
            "angle_deg": angle_deg,
            "laser_px": (lx, ly),
            "target_px": (bx, by),
            "bullseye_plane_cm": self.bullseye_plane_cm,
            "laser_plane_cm": laser_plane_cm,
            "target_valid": self.target_visual_valid,
            "bullseye_valid": self.bullseye_visual_valid,
            "laser_valid": self.laser_visual_valid,
            "prediction_active": self.prediction_active,
        }


class CircleTrajectory:
    def __init__(self, radius_cm=CIRCLE_RADIUS_CM, points_per_circle=360):
        self.radius_cm = radius_cm
        self.points_per_circle = points_per_circle
        self.angle_step = 2 * math.pi / points_per_circle
        self.reset()

    def get_target_point(self):
        dx = self.radius_cm * math.cos(self.current_angle)
        dy = self.radius_cm * math.sin(self.current_angle)
        self.current_angle += self.angle_step
        if self.current_angle >= 2 * math.pi:
            self.current_angle -= 2 * math.pi
        return dx, dy

    def reset(self):
        self.current_angle = 0.0


class AimingSystem:
    def __init__(self):
        self.detector = TargetDetector()
        self.trajectory = CircleTrajectory(CIRCLE_RADIUS_CM)
        self.motor = build_stepper_controller(STEPPER_AXIS_OVERRIDES)
        self.control_started = AUTO_START
        self.start_button = StartButton(START_BUTTON_BOARD_PIN, START_BUTTON_GPIO_NUM)
        self.mode = "aim"
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_time = time.ticks_ms()
        self.gc_counter = 0
        self.control_state = "aim"
        self.turn_boost_active = False
        self.turn_boost_enter_count = 0
        self.turn_boost_exit_count = 0
        self.dx_growth_count = 0
        self.last_turn_eval_dx_cm = None
        self.last_rate_sample = None
        self.last_visual_dx_rate = 0.0
        self.last_visual_dy_rate = 0.0
        self.last_valid_offset_ms = None
        self.last_valid_base_dx_cm = 0.0
        self.last_valid_base_dy_cm = 0.0
        self.last_yaw_lead_cm = 0.0
        self.last_pitch_lead_cm = 0.0
        self.last_cmd_dx_cm = 0.0
        self.last_cmd_dy_cm = 0.0
        self.last_reacquire_yaw_dir = 0

    def _update_start_button(self):
        if self.control_started:
            return
        if self.start_button.poll_pressed():
            self.control_started = True
            print("[Motor] start button pressed, stepper control enabled")

    def _compute_visual_rates(self, offset_info):
        if offset_info is None:
            self.last_visual_dx_rate = 0.0
            self.last_visual_dy_rate = 0.0
            self.last_rate_sample = None
            return 0.0, 0.0

        current_sample = {
            "timestamp_ms": offset_info.get("timestamp_ms", time.ticks_ms()),
            "dx_cm": offset_info["dx_cm"],
            "dy_cm": offset_info["dy_cm"],
        }
        previous_sample = self.last_rate_sample
        self.last_rate_sample = current_sample
        if previous_sample is None:
            self.last_visual_dx_rate = 0.0
            self.last_visual_dy_rate = 0.0
            return 0.0, 0.0

        dt_ms = max(1, time.ticks_diff(current_sample["timestamp_ms"], previous_sample["timestamp_ms"]))
        raw_dx_rate = (current_sample["dx_cm"] - previous_sample["dx_cm"]) * 1000.0 / dt_ms
        raw_dy_rate = (current_sample["dy_cm"] - previous_sample["dy_cm"]) * 1000.0 / dt_ms
        alpha = VISUAL_RATE_FILTER_ALPHA
        self.last_visual_dx_rate = self.last_visual_dx_rate * (1.0 - alpha) + raw_dx_rate * alpha
        self.last_visual_dy_rate = self.last_visual_dy_rate * (1.0 - alpha) + raw_dy_rate * alpha
        return self.last_visual_dx_rate, self.last_visual_dy_rate

    def _compute_yaw_lead(self, dx_cm_rate):
        gain = YAW_LEAD_GAIN_TURN if self.turn_boost_active else YAW_LEAD_GAIN_NORMAL
        yaw_lead_cm = dx_cm_rate * gain
        if yaw_lead_cm > YAW_LEAD_MAX_CM:
            yaw_lead_cm = YAW_LEAD_MAX_CM
        elif yaw_lead_cm < -YAW_LEAD_MAX_CM:
            yaw_lead_cm = -YAW_LEAD_MAX_CM
        return yaw_lead_cm

    def _update_turn_boost_state(self, dx_cm, dx_cm_rate):
        growing = False
        if self.last_turn_eval_dx_cm is not None:
            same_side = dx_cm * self.last_turn_eval_dx_cm > 0.0
            error_growing = abs(dx_cm) > (abs(self.last_turn_eval_dx_cm) + TURN_BOOST_GROWTH_DELTA_CM)
            if same_side and error_growing:
                self.dx_growth_count += 1
            else:
                self.dx_growth_count = 0
            growing = self.dx_growth_count >= TURN_BOOST_MIN_FRAMES
        self.last_turn_eval_dx_cm = dx_cm

        rate_trigger = abs(dx_cm_rate) >= TURN_BOOST_ENTER_RATE_CM_S
        if self.turn_boost_active:
            calm_rate = abs(dx_cm_rate) <= TURN_BOOST_EXIT_RATE_CM_S
            calm_error = abs(dx_cm) <= TURN_BOOST_EXIT_ERROR_CM
            if calm_rate and calm_error:
                self.turn_boost_exit_count += 1
            else:
                self.turn_boost_exit_count = 0
            if self.turn_boost_exit_count >= TURN_BOOST_EXIT_FRAMES:
                self.turn_boost_active = False
                self.turn_boost_enter_count = 0
                self.turn_boost_exit_count = 0
                self.dx_growth_count = 0
        else:
            if rate_trigger or growing:
                self.turn_boost_enter_count += 1
            else:
                self.turn_boost_enter_count = 0
            if self.turn_boost_enter_count >= TURN_BOOST_MIN_FRAMES:
                self.turn_boost_active = True
                self.turn_boost_exit_count = 0

        return self.turn_boost_active

    def _compute_hold_command(self, now_ms):
        if self.last_valid_offset_ms is None:
            return None
        hold_age_ms = time.ticks_diff(now_ms, self.last_valid_offset_ms)
        if hold_age_ms < 0 or hold_age_ms > LOSS_HOLD_MS:
            return None

        age_ratio = hold_age_ms / float(max(1, LOSS_HOLD_MS))
        lead_scale = HOLD_COMMAND_DECAY + (1.0 - HOLD_COMMAND_DECAY) * (1.0 - age_ratio)
        cmd_dx = self.last_valid_base_dx_cm + self.last_yaw_lead_cm * lead_scale
        cmd_dy = self.last_valid_base_dy_cm + self.last_pitch_lead_cm * lead_scale
        yaw_limit = TURN_BOOST_YAW_CMD_LIMIT_CM if self.turn_boost_active else NORMAL_YAW_CMD_LIMIT_CM
        cmd_dx = max(-yaw_limit, min(yaw_limit, cmd_dx))
        cmd_dy = max(-PITCH_CMD_LIMIT_CM, min(PITCH_CMD_LIMIT_CM, cmd_dy))
        return {
            "cmd_dx_cm": cmd_dx,
            "cmd_dy_cm": cmd_dy,
            "hold_age_ms": hold_age_ms,
            "yaw_lead_cm": self.last_yaw_lead_cm * lead_scale,
            "pitch_lead_cm": self.last_pitch_lead_cm * lead_scale,
        }

    def _compute_reacquire_command(self, now_ms):
        if self.last_valid_offset_ms is None:
            return None
        reacquire_age_ms = time.ticks_diff(now_ms, self.last_valid_offset_ms)
        if reacquire_age_ms <= LOSS_HOLD_MS or reacquire_age_ms > LOSS_REACQUIRE_MS:
            return None

        yaw_dir = self.last_reacquire_yaw_dir
        if yaw_dir == 0:
            if abs(self.last_cmd_dx_cm) > 0.2:
                yaw_dir = 1 if self.last_cmd_dx_cm > 0.0 else -1
            elif abs(self.last_valid_base_dx_cm) > 0.2:
                yaw_dir = 1 if self.last_valid_base_dx_cm > 0.0 else -1
            elif abs(self.last_yaw_lead_cm) > 0.1:
                yaw_dir = 1 if self.last_yaw_lead_cm > 0.0 else -1
            else:
                return None

        cmd_dx = yaw_dir * LOSS_REACQUIRE_YAW_CMD_CM
        cmd_dy = self.last_valid_base_dy_cm * LOSS_REACQUIRE_PITCH_HOLD_RATIO
        cmd_dy = max(-PITCH_CMD_LIMIT_CM, min(PITCH_CMD_LIMIT_CM, cmd_dy))
        return {
            "cmd_dx_cm": cmd_dx,
            "cmd_dy_cm": cmd_dy,
            "reacquire_age_ms": reacquire_age_ms,
            "yaw_dir": yaw_dir,
        }

    def process_frame(self, img):
        self.frame_count += 1
        self.gc_counter += 1
        self._update_start_button()

        self.detector.detect_all(img)
        offset_info = self.detector.get_offset_info()
        now_ms = time.ticks_ms()
        control_debug = {
            "state": self.control_state,
            "dx_cm": None,
            "dy_cm": None,
            "dx_rate_cm_s": self.last_visual_dx_rate,
            "dy_rate_cm_s": self.last_visual_dy_rate,
            "yaw_lead_cm": self.last_yaw_lead_cm,
            "hold_age_ms": None,
            "cmd_dx_cm": self.last_cmd_dx_cm,
            "cmd_dy_cm": self.last_cmd_dy_cm,
        }

        if self.mode == "aim":
            if offset_info is None:
                hold_command = self._compute_hold_command(now_ms)
                if hold_command is not None:
                    self.control_state = "loss_hold"
                    self.motor.drive(
                        hold_command["cmd_dx_cm"],
                        hold_command["cmd_dy_cm"],
                        allow_drive=self.control_started,
                    )
                    self.last_cmd_dx_cm = hold_command["cmd_dx_cm"]
                    self.last_cmd_dy_cm = hold_command["cmd_dy_cm"]
                    control_debug.update({
                        "state": self.control_state,
                        "dx_cm": self.last_valid_base_dx_cm,
                        "dy_cm": self.last_valid_base_dy_cm,
                        "dx_rate_cm_s": self.last_visual_dx_rate,
                        "dy_rate_cm_s": self.last_visual_dy_rate,
                        "yaw_lead_cm": hold_command["yaw_lead_cm"],
                        "hold_age_ms": hold_command["hold_age_ms"],
                        "cmd_dx_cm": hold_command["cmd_dx_cm"],
                        "cmd_dy_cm": hold_command["cmd_dy_cm"],
                    })
                else:
                    reacquire_command = self._compute_reacquire_command(now_ms)
                    if reacquire_command is not None:
                        self.control_state = "loss_reacquire"
                        self.motor.drive(
                            reacquire_command["cmd_dx_cm"],
                            reacquire_command["cmd_dy_cm"],
                            allow_drive=self.control_started,
                        )
                        self.last_cmd_dx_cm = reacquire_command["cmd_dx_cm"]
                        self.last_cmd_dy_cm = reacquire_command["cmd_dy_cm"]
                        control_debug.update({
                            "state": self.control_state,
                            "dx_cm": self.last_valid_base_dx_cm,
                            "dy_cm": self.last_valid_base_dy_cm,
                            "dx_rate_cm_s": self.last_visual_dx_rate,
                            "dy_rate_cm_s": self.last_visual_dy_rate,
                            "yaw_lead_cm": self.last_yaw_lead_cm,
                            "hold_age_ms": reacquire_command["reacquire_age_ms"],
                            "cmd_dx_cm": reacquire_command["cmd_dx_cm"],
                            "cmd_dy_cm": reacquire_command["cmd_dy_cm"],
                        })
                    else:
                        self.control_state = "aim"
                        self.turn_boost_active = False
                        self.turn_boost_enter_count = 0
                        self.turn_boost_exit_count = 0
                        self.dx_growth_count = 0
                        self.last_turn_eval_dx_cm = None
                        self.motor.stop()
                        control_debug.update({
                            "state": self.control_state,
                            "dx_rate_cm_s": self.last_visual_dx_rate,
                            "dy_rate_cm_s": self.last_visual_dy_rate,
                            "yaw_lead_cm": 0.0,
                            "cmd_dx_cm": 0.0,
                            "cmd_dy_cm": 0.0,
                        })
            else:
                dx_rate_cm_s, dy_rate_cm_s = self._compute_visual_rates(offset_info)
                self._update_turn_boost_state(offset_info["dx_cm"], dx_rate_cm_s)
                yaw_lead_cm = self._compute_yaw_lead(dx_rate_cm_s)
                pitch_lead_cm = dy_rate_cm_s * PITCH_LEAD_GAIN
                cmd_dx_cm = offset_info["dx_cm"] + yaw_lead_cm
                cmd_dy_cm = offset_info["dy_cm"] + pitch_lead_cm
                yaw_limit = TURN_BOOST_YAW_CMD_LIMIT_CM if self.turn_boost_active else NORMAL_YAW_CMD_LIMIT_CM
                cmd_dx_cm = max(-yaw_limit, min(yaw_limit, cmd_dx_cm))
                cmd_dy_cm = max(-PITCH_CMD_LIMIT_CM, min(PITCH_CMD_LIMIT_CM, cmd_dy_cm))
                self.motor.drive(
                    cmd_dx_cm,
                    cmd_dy_cm,
                    allow_drive=self.control_started,
                )
                self.last_valid_offset_ms = offset_info["timestamp_ms"]
                self.last_valid_base_dx_cm = offset_info["dx_cm"]
                self.last_valid_base_dy_cm = offset_info["dy_cm"]
                self.last_yaw_lead_cm = yaw_lead_cm
                self.last_pitch_lead_cm = pitch_lead_cm
                self.last_cmd_dx_cm = cmd_dx_cm
                self.last_cmd_dy_cm = cmd_dy_cm
                if abs(cmd_dx_cm) > 0.2:
                    self.last_reacquire_yaw_dir = 1 if cmd_dx_cm > 0.0 else -1
                elif abs(offset_info["dx_cm"]) > 0.2:
                    self.last_reacquire_yaw_dir = 1 if offset_info["dx_cm"] > 0.0 else -1
                self.control_state = "turn_boost" if self.turn_boost_active else "aim"
                control_debug.update({
                    "state": self.control_state,
                    "dx_cm": offset_info["dx_cm"],
                    "dy_cm": offset_info["dy_cm"],
                    "dx_rate_cm_s": dx_rate_cm_s,
                    "dy_rate_cm_s": dy_rate_cm_s,
                    "yaw_lead_cm": yaw_lead_cm,
                    "hold_age_ms": max(0, time.ticks_diff(now_ms, offset_info["timestamp_ms"])),
                    "cmd_dx_cm": cmd_dx_cm,
                    "cmd_dy_cm": cmd_dy_cm,
                })
        elif self.mode == "circle":
            target_dx, target_dy = self.trajectory.get_target_point()
            if offset_info is None:
                self.motor.stop()
                control_debug.update({
                    "state": "circle",
                    "dx_rate_cm_s": self.last_visual_dx_rate,
                    "dy_rate_cm_s": self.last_visual_dy_rate,
                })
            else:
                self.motor.drive(
                    target_dx - offset_info["dx_cm"],
                    target_dy - offset_info["dy_cm"],
                    allow_drive=self.control_started,
                )
                control_debug.update({
                    "state": "circle",
                    "dx_cm": offset_info["dx_cm"],
                    "dy_cm": offset_info["dy_cm"],
                    "dx_rate_cm_s": self.last_visual_dx_rate,
                    "dy_rate_cm_s": self.last_visual_dy_rate,
                    "yaw_lead_cm": 0.0,
                    "hold_age_ms": max(0, time.ticks_diff(now_ms, offset_info["timestamp_ms"])),
                    "cmd_dx_cm": target_dx - offset_info["dx_cm"],
                    "cmd_dy_cm": target_dy - offset_info["dy_cm"],
                })
            self.control_state = "circle"
        else:
            self.control_state = "idle"
            control_debug.update({
                "state": "idle",
                "dx_rate_cm_s": self.last_visual_dx_rate,
                "dy_rate_cm_s": self.last_visual_dy_rate,
            })

        if DEBUG_MODE:
            self._draw_debug_overlay(img, offset_info, control_debug)

        return img

    def _draw_debug_overlay(self, img, offset_info, control_debug):
        if self.detector.target_found and self.detector.target_rect:
            x, y, w, h = self.detector.target_rect
            img.draw_rectangle(x, y, w, h, color=(0, 255, 0), thickness=1)
        if self.detector.target_found and self.detector.target_center:
            cx, cy = self.detector.target_center
            img.draw_cross(cx, cy, color=(0, 255, 0), size=8, thickness=1)

        if self.detector.laser_found and self.detector.laser_spot:
            cx, cy = self.detector.laser_spot
            img.draw_circle(cx, cy, 4, color=(255, 255, 0), thickness=1)
            if self.detector.bullseye_found and self.detector.bullseye_center:
                bx, by = self.detector.bullseye_center
                img.draw_line(cx, cy, bx, by, color=(0, 255, 255), thickness=1)

        if DEBUG_TEXT_OVERLAY:
            if offset_info:
                draw_text(
                    img,
                    4,
                    4,
                    "D={:.1f} A={:.0f}".format(
                        offset_info["distance_cm"], offset_info["angle_deg"]
                    ),
                    color=(255, 255, 255),
                    scale=1,
                )
                draw_text(
                    img,
                    4,
                    22,
                    "dx={:.1f} dy={:.1f}".format(
                        control_debug["dx_cm"], control_debug["dy_cm"]
                    ),
                    color=(255, 255, 255),
                    scale=1,
                )
                draw_text(
                    img,
                    4,
                    40,
                    "vx={:.1f} lead={:.1f}".format(
                        control_debug["dx_rate_cm_s"], control_debug["yaw_lead_cm"]
                    ),
                    color=(255, 255, 255),
                    scale=1,
                )
                draw_text(
                    img,
                    4,
                    58,
                    "age={}ms {}".format(
                        int(control_debug["hold_age_ms"] or 0),
                        control_debug["state"].upper(),
                    ),
                    color=(255, 255, 0),
                    scale=1,
                )
            else:
                draw_text(
                    img,
                    4,
                    4,
                    "STATE {}".format(control_debug["state"].upper()),
                    color=(255, 255, 0),
                    scale=1,
                )
                draw_text(
                    img,
                    4,
                    22,
                    "vx={:.1f} lead={:.1f}".format(
                        control_debug["dx_rate_cm_s"], control_debug["yaw_lead_cm"]
                    ),
                    color=(255, 255, 255),
                    scale=1,
                )
                if control_debug["hold_age_ms"] is not None:
                    draw_text(
                        img,
                        4,
                        40,
                        "hold={}ms".format(int(control_debug["hold_age_ms"])),
                        color=(255, 255, 0),
                        scale=1,
                    )

            draw_text(
                img,
                4,
                FRAME_HEIGHT - 16,
                "Mode: {}/{}".format(self.mode.upper(), control_debug["state"].upper()),
                color=(0, 255, 0),
                scale=1,
            )
            draw_text(
                img,
                FRAME_WIDTH - 70,
                4,
                "FPS:{:.1f}".format(self.fps),
                color=(255, 255, 255),
                scale=1,
            )
        if not self.control_started:
            draw_text(
                img,
                4,
                FRAME_HEIGHT - 16,
                "PRESS GPIO28 TO START MOTOR",
                color=(255, 255, 0),
                scale=1,
            )

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

    def set_mode(self, mode):
        if mode in ("aim", "circle", "idle"):
            self.mode = mode
            if mode == "circle":
                self.trajectory.reset()
            if mode == "idle":
                self.motor.stop()
            print(f"[Mode] {mode}")


def main():
    print("=" * 50)
    print("K230 aiming system")
    print("build:", BUILD_TAG)
    print("fast mode enabled")
    print("=" * 50)
    apply_calibration()

    aiming_system = AimingSystem()

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

    aiming_system.set_mode("aim")
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
            img = aiming_system.process_frame(img)
            aiming_system.update_fps()
            Display.show_image(img)
            aiming_system.maybe_collect_gc()
            time.sleep_ms(FRAME_LOOP_DELAY_MS)
    except KeyboardInterrupt:
        print("\n[System] interrupted")
    except Exception as e:
        print("[Error]", e)
        sys.print_exception(e)
    finally:
        print("[System] cleanup...")
        camera_deinit(sensor)
        aiming_system.motor.deinit()
        print("[System] stopped")


def main_test():
    print("K230 aiming system - test mode")
    apply_calibration()

    kw = dict(camera_id=CAMERA_ID, width=FRAME_WIDTH, height=FRAME_HEIGHT,
              hmirror=SENSOR_HMIRROR, vflip=SENSOR_VFLIP)
    sensor = camera_init(CAMERA_ID)
    display_init(FRAME_WIDTH, FRAME_HEIGHT)
    try:
        camera_start(sensor, **kw)
    except Exception as e:
        print("[Sensor] start failed in test, retry:", e)
        sensor = camera_restart(sensor, **kw)

    detector = TargetDetector()
    clock = time.clock()
    frame_count = 0

    try:
        while True:
            os.exitpoint()
            clock.tick()
            frame_count += 1

            try:
                img = camera_snapshot(sensor)
            except RuntimeError as e:
                print("[Sensor] snapshot failed in test, restart:", e)
                sensor = camera_restart(sensor, **kw)
                img = camera_snapshot(sensor)
            detector.detect_all(img)
            offset = detector.get_offset_info()

            if frame_count % 30 == 0:
                print("-" * 40)
                print("target:{} bullseye:{} laser:{}".format(
                    detector.target_found, detector.bullseye_found, detector.laser_found))
                if offset:
                    print("offset dx={:.2f} dy={:.2f} dist={:.2f} angle={:.1f}".format(
                        offset["dx_cm"], offset["dy_cm"], offset["distance_cm"], offset["angle_deg"]))
                print("FPS: {:.1f}".format(clock.fps()))

            if frame_count % GC_INTERVAL == 0:
                gc.collect()
            time.sleep_ms(FRAME_LOOP_DELAY_MS)
    except KeyboardInterrupt:
        print("\nstop test")
    except Exception as e:
        print("error:", e)
        sys.print_exception(e)
    finally:
        camera_deinit(sensor)
        print("test finished")


if __name__ == "__main__":
    main()
