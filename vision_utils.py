"""
vision_utils.py -- Shared vision/geometry/filtering/logging for K230 gimbal.

All pure functions: no class state, no side effects.  Import from this module
instead of duplicating code across the three run-mode scripts.
"""
import math
import sys
import time

# ═══════════════════════════════════════════════════════════════
#  Logging
# ═══════════════════════════════════════════════════════════════

_LOG_LEVEL = 3   # 0=silent  1=error  2=warn  3=info  4=debug


def log_level(level=None):
    """Get or set the current log level (0-4)."""
    global _LOG_LEVEL
    if level is not None:
        _LOG_LEVEL = int(level)
    return _LOG_LEVEL


def _log(tag, level, *args):
    if level <= _LOG_LEVEL:
        print("[{}]".format(tag), *args)


def log_error(*args):
    _log("ERR", 1, *args)


def log_warn(*args):
    _log("WARN", 2, *args)


def log_info(*args):
    _log("INFO", 3, *args)


def log_debug(*args):
    _log("DBG", 4, *args)


# ═══════════════════════════════════════════════════════════════
#  Pure math helpers
# ═══════════════════════════════════════════════════════════════

def dist_sq(p0, p1):
    """Squared Euclidean distance between two (x, y) tuples."""
    dx = p0[0] - p1[0]
    dy = p0[1] - p1[1]
    return dx * dx + dy * dy


def clamp_point(point, frame_w, frame_h):
    """Clamp an (x, y) tuple inside [0, frame_w-1] × [0, frame_h-1]."""
    return (
        max(0, min(frame_w - 1, int(point[0]))),
        max(0, min(frame_h - 1, int(point[1]))),
    )


def clamp_rect(x, y, w, h, frame_w, frame_h):
    """Clamp a rectangle inside image bounds, shrinking w/h if needed."""
    x = max(0, min(frame_w - 1, int(x)))
    y = max(0, min(frame_h - 1, int(y)))
    w = max(1, min(frame_w - x, int(w)))
    h = max(1, min(frame_h - y, int(h)))
    return (x, y, w, h)


# ═══════════════════════════════════════════════════════════════
#  EMA / motion-lead smoothing
# ═══════════════════════════════════════════════════════════════

def smooth_center(current, last_center, alpha, reset_px, sticky_px):
    """Exponential moving average with sticky zone and reset threshold.

    - Distance ≤ *sticky_px* → return *last_center* unchanged.
    - Distance > *reset_px*  → return *current* (instant jump).
    - Otherwise              → EMA blend.
    """
    if last_center is None:
        return current
    d2 = dist_sq(current, last_center)
    if d2 <= (sticky_px * sticky_px):
        return last_center
    if d2 > (reset_px * reset_px):
        return current
    return (
        int(last_center[0] * (1 - alpha) + current[0] * alpha),
        int(last_center[1] * (1 - alpha) + current[1] * alpha),
    )


def smooth_scalar(current, last_value, alpha):
    """Scalar EMA — if *last_value* ≤ 0, return *current* unchanged."""
    if last_value <= 0:
        return current
    return last_value * (1 - alpha) + current * alpha


def apply_motion_lead(current, last_center, gain, max_px, frame_w, frame_h):
    """Predict next position using velocity between *last_center* and *current*."""
    if current is None or last_center is None or gain <= 0:
        return current
    dx = current[0] - last_center[0]
    dy = current[1] - last_center[1]
    lead_x = int(dx * gain)
    lead_y = int(dy * gain)
    if lead_x > max_px:      lead_x = max_px
    elif lead_x < -max_px:   lead_x = -max_px
    if lead_y > max_px:      lead_y = max_px
    elif lead_y < -max_px:   lead_y = -max_px
    return clamp_point((current[0] + lead_x, current[1] + lead_y),
                       frame_w, frame_h)


# ═══════════════════════════════════════════════════════════════
#  History buffers (median-of-3 filter)
# ═══════════════════════════════════════════════════════════════

def push_point_history(history, point, max_len):
    history.append(point)
    if len(history) > max_len:
        history.pop(0)
    return history


def filter_point_history(history):
    """Median of (x, y) history — x and y filtered independently."""
    if not history:
        return None
    xs = sorted(pt[0] for pt in history)
    ys = sorted(pt[1] for pt in history)
    mid = len(history) // 2
    return (xs[mid], ys[mid])


def push_scalar_history(history, value, max_len=1):
    history.append(value)
    if len(history) > max_len:
        history.pop(0)
    return history


def filter_scalar_history(history):
    """Median value of scalar history."""
    if not history:
        return 0.0
    vals = sorted(history)
    return vals[len(vals) // 2]


# ═══════════════════════════════════════════════════════════════
#  Rectangle geometry
# ═══════════════════════════════════════════════════════════════

def rect_aspect_error(w, h, target_aspect):
    """Penalty for deviating from expected aspect ratio."""
    aspect = w / max(h, 1)
    target_inv = 1.0 / target_aspect
    return min(abs(aspect - target_aspect), abs(aspect - target_inv))


def rect_size_change_ok(rect, last_rect, max_ratio):
    """Return True if width/height change is within *max_ratio*."""
    if last_rect is None:
        return True
    _, _, w, h = rect
    _, _, lw, lh = last_rect
    if lw <= 0 or lh <= 0:
        return True
    dw = abs(w - lw) / lw
    dh = abs(h - lh) / lh
    return dw <= max_ratio and dh <= max_ratio


def rect_overlap_ratio(rect_a, rect_b):
    """IoU-like ratio: intersection area / min(area_a, area_b)."""
    if rect_a is None or rect_b is None:
        return 0.0
    ax, ay, aw, ah = rect_a
    bx, by, bw, bh = rect_b
    ix1 = max(ax, bx);    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw);  iy2 = min(ay + ah, by + bh)
    iw = ix2 - ix1;        ih = iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    min_area = min(max(1, aw * ah), max(1, bw * bh))
    return inter / min_area


def compensate_edge_rect(rect, last_rect, edge_margin, comp_min_ratio,
                          frame_w, frame_h):
    """If a rect is near the image edge and shrank suspiciously, borrow
    size from *last_rect*."""
    if last_rect is None:
        return rect
    x, y, w, h = rect
    _, _, lw, lh = last_rect
    if lw <= 0 or lh <= 0:
        return rect

    x2 = x + w;   y2 = y + h
    if x <= edge_margin and w < int(lw * comp_min_ratio):
        w = lw
    elif x2 >= (frame_w - edge_margin) and w < int(lw * comp_min_ratio):
        x = max(0, x2 - lw)
        w = frame_w - x if x + lw > frame_w else lw
    if y <= edge_margin and h < int(lh * comp_min_ratio):
        h = lh
    elif y2 >= (frame_h - edge_margin) and h < int(lh * comp_min_ratio):
        y = max(0, y2 - lh)
        h = frame_h - y if y + lh > frame_h else lh
    return clamp_rect(x, y, w, h, frame_w, frame_h)


def rect_border_hit_ratio(rect_img, rect, border_sample_count, corners=None):
    """Ratio of sampled border pixels that are true (binary image).

    When corners are available, sample the actual quadrilateral edges instead
    of the horizontal/vertical bounding box. This keeps rotated targets from
    being rejected by the border check.
    """
    if corners is not None and len(corners) == 4:
        points = normalize_corners(corners)
        hits = 0
        total = 0
        steps = max(4, border_sample_count)
        for idx in range(4):
            p0 = points[idx]
            p1 = points[(idx + 1) % 4]
            for sample in range(steps):
                t = sample / max(1, steps - 1)
                px = int(p0[0] + (p1[0] - p0[0]) * t)
                py = int(p0[1] + (p1[1] - p0[1]) * t)
                total += 1
                try:
                    if rect_img.get_pixel(px, py):
                        hits += 1
                except Exception:
                    pass
        return hits / total if total > 0 else 0.0

    x, y, w, h = rect
    if w <= 4 or h <= 4:
        return 0.0
    inset = max(1, min(6, min(w, h) // 12))
    x1 = x + inset;   y1 = y + inset
    x2 = x + w - 1 - inset;   y2 = y + h - 1 - inset
    if x2 <= x1 or y2 <= y1:
        return 0.0

    hits = 0;  total = 0
    steps = max(4, border_sample_count)
    for idx in range(steps):
        t = idx / max(1, steps - 1)
        sx = int(x1 + (x2 - x1) * t)
        sy = int(y1 + (y2 - y1) * t)
        for px, py in ((sx, y1), (sx, y2), (x1, sy), (x2, sy)):
            total += 1
            try:
                if rect_img.get_pixel(px, py):
                    hits += 1
            except Exception:
                pass
    if total <= 0:
        return 0.0
    return hits / total


def expand_rect(rect, margin, frame_w, frame_h):
    x, y, w, h = rect
    return clamp_rect(x - margin, y - margin, w + margin * 2, h + margin * 2,
                      frame_w, frame_h)


def rect_center_from_corners(corners, frame_w, frame_h):
    """Intersection of the two diagonals of a quadrilateral (sub-pixel)."""
    sx = 0;  sy = 0
    pts = [(p[0], p[1]) for p in corners]
    for p in pts:
        sx += p[0];  sy += p[1]
    avg = (sx / 4, sy / 4)

    pts.sort(key=lambda p: math.atan2(p[1] - avg[1], p[0] - avg[0]))
    p0, p1, p2, p3 = pts[0], pts[1], pts[2], pts[3]
    x1, y1 = p0;  x2, y2 = p2   # diagonal 0-2
    x3, y3 = p1;  x4, y4 = p3   # diagonal 1-3

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1:
        return clamp_point(avg, frame_w, frame_h)

    det1 = x1 * y2 - y1 * x2
    det2 = x3 * y4 - y3 * x4
    cx = (det1 * (x3 - x4) - (x1 - x2) * det2) / denom
    cy = (det1 * (y3 - y4) - (y1 - y2) * det2) / denom
    if cx < -8 or cx > (frame_w + 8) or cy < -8 or cy > (frame_h + 8):
        return clamp_point(avg, frame_w, frame_h)
    return clamp_point((cx, cy), frame_w, frame_h)


# ═══════════════════════════════════════════════════════════════
#  Homography  (4-point → 4-point  perspective transform)
# ═══════════════════════════════════════════════════════════════

def _solve_linear_8x8(matrix, values):
    """Gaussian elimination for 8×8 system. Returns tuple or None."""
    size = len(values)
    a = []
    for row in range(size):
        row_data = [float(matrix[row][col]) for col in range(size)]
        row_data.append(float(values[row]))
        a.append(row_data)

    for col in range(size):
        pivot = col
        best = abs(a[pivot][col])
        for row in range(col + 1, size):
            v = abs(a[row][col])
            if v > best:
                pivot, best = row, v
        if best < 1e-6:
            return None
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]

        pv = a[col][col]
        for idx in range(col, size + 1):
            a[col][idx] /= pv

        for row in range(size):
            if row == col:
                continue
            factor = a[row][col]
            for idx in range(col, size + 1):
                a[row][idx] -= factor * a[col][idx]

    return tuple(a[row][size] for row in range(size))


def compute_homography(src_points, dst_points):
    """Compute 3×3 homography mapping src → dst (4 corner pairs)."""
    m = [];  v = []
    for idx in range(4):
        x, y = float(src_points[idx][0]), float(src_points[idx][1])
        u, k = float(dst_points[idx][0]), float(dst_points[idx][1])
        m.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y]);   v.append(u)
        m.append([0.0, 0.0, 0.0, x, y, 1.0, -k * x, -k * y]);   v.append(k)

    sol = _solve_linear_8x8(m, v)
    if sol is None:
        return None
    return ((sol[0], sol[1], sol[2]),
            (sol[3], sol[4], sol[5]),
            (sol[6], sol[7], 1.0))


def apply_homography(h, x, y):
    if h is None:
        return None
    denom = h[2][0] * x + h[2][1] * y + h[2][2]
    if abs(denom) < 1e-6:
        return None
    return ((h[0][0] * x + h[0][1] * y + h[0][2]) / denom,
            (h[1][0] * x + h[1][1] * y + h[1][2]) / denom)


def normalize_corners(corners):
    """Sort 4 corners to top-left, top-right, bottom-right, bottom-left order."""
    pts = [(float(p[0]), float(p[1])) for p in corners]
    center = (sum(p[0] for p in pts) / 4, sum(p[1] for p in pts) / 4)
    pts.sort(key=lambda p: math.atan2(p[1] - center[1], p[0] - center[0]))
    tl = min(range(4), key=lambda i: pts[i][0] + pts[i][1])
    return [pts[(tl + i) % 4] for i in range(4)]


def plane_size_cm_for_corners(corners, target_aspect, width_cm, height_cm):
    """Determine whether corners represent portrait or landscape target."""
    top    = math.hypot(corners[1][0] - corners[0][0], corners[1][1] - corners[0][1])
    right  = math.hypot(corners[2][0] - corners[1][0], corners[2][1] - corners[1][1])
    bottom = math.hypot(corners[2][0] - corners[3][0], corners[2][1] - corners[3][1])
    left   = math.hypot(corners[3][0] - corners[0][0], corners[3][1] - corners[0][1])
    w_px, h_px = (top + bottom) * 0.5, (left + right) * 0.5
    aspect = w_px / max(h_px, 1e-6)
    if abs(aspect - (1.0 / target_aspect)) < abs(aspect - target_aspect):
        return height_cm, width_cm
    return width_cm, height_cm


# ═══════════════════════════════════════════════════════════════
#  Circle-mode state machine
# ═══════════════════════════════════════════════════════════════

class CircleState:
    """State constants for circle_mode.  Safe to compare via ``==``.

    Usage:  ``if self.state == CircleState.IDLE: …``
    """
    IDLE = 0
    WAITING = 1
    RUNNING = 2

    _names = {0: "IDLE", 1: "WAITING", 2: "RUNNING"}

    @staticmethod
    def name(value):
        return CircleState._names.get(value, "UNKNOWN")
