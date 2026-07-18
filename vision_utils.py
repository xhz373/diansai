import math


class FrameGeometryMixin:
    frame_width = 0
    frame_height = 0

    def _distance_sq(self, p0, p1):
        dx = p0[0] - p1[0]
        dy = p0[1] - p1[1]
        return dx * dx + dy * dy

    def _clamp_rect(self, x, y, w, h):
        x = max(0, min(self.frame_width - 1, int(x)))
        y = max(0, min(self.frame_height - 1, int(y)))
        w = max(1, min(self.frame_width - x, int(w)))
        h = max(1, min(self.frame_height - y, int(h)))
        return (x, y, w, h)

    def _clamp_point(self, point):
        return (
            max(0, min(self.frame_width - 1, int(point[0]))),
            max(0, min(self.frame_height - 1, int(point[1]))),
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

    def _expand_rect(self, rect, margin):
        x, y, w, h = rect
        return self._clamp_rect(x - margin, y - margin, w + margin * 2, h + margin * 2)

    def _push_point_history(self, history, point, max_len=None):
        history.append(point)
        if max_len is not None and len(history) > max_len:
            history.pop(0)
        return history

    def _filter_point_history(self, history):
        if not history:
            return None
        xs = sorted(point[0] for point in history)
        ys = sorted(point[1] for point in history)
        mid = len(history) // 2
        return (xs[mid], ys[mid])

    def _push_scalar_history(self, history, value, max_len=None):
        history.append(value)
        if max_len is not None and len(history) > max_len:
            history.pop(0)
        return history

    def _filter_scalar_history(self, history):
        if not history:
            return 0.0
        values = sorted(history)
        return values[len(values) // 2]


class HomographyMixin(FrameGeometryMixin):
    target_width_cm = 0.0
    target_height_cm = 0.0
    target_aspect = 1.0

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
        normal_error = abs(aspect - self.target_aspect)
        swapped_error = abs(aspect - (1.0 / self.target_aspect))
        if swapped_error < normal_error:
            return self.target_height_cm, self.target_width_cm
        return self.target_width_cm, self.target_height_cm

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


class RectTrackingMixin(FrameGeometryMixin):
    rect_binary_threshold = (0, 0)
    target_aspect = 1.0
    target_aspect_penalty_scale = 0
    target_min_w = 0
    target_min_h = 0
    target_min_area = 0
    target_center_alpha = 0.0
    target_reset_dist_px = 0
    target_sticky_dist_px = 0
    target_lead_gain = 0.0
    target_lead_max_px = 0
    target_max_jump_px = 0
    target_max_size_change_ratio = 0.0
    target_edge_margin_px = 0
    target_edge_comp_min_ratio = 0.0
    target_min_overlap_ratio = 0.0
    target_init_center_bias = 1
    target_near_center_px = 0
    target_border_sample_count = 0
    target_border_hit_ratio_min = 0.0
    target_border_score_scale = 0

    def _rect_aspect_error(self, w, h):
        aspect = w / max(h, 1)
        target_inv = 1.0 / self.target_aspect
        return min(abs(aspect - self.target_aspect), abs(aspect - target_inv))

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
        if (
            center_x < -8
            or center_x > (self.frame_width + 8)
            or center_y < -8
            or center_y > (self.frame_height + 8)
        ):
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
        return dw <= self.target_max_size_change_ratio and dh <= self.target_max_size_change_ratio

    def _compensate_edge_rect(self, rect, last_rect):
        if last_rect is None:
            return rect
        x, y, w, h = rect
        _, _, last_w, last_h = last_rect
        if last_w <= 0 or last_h <= 0:
            return rect

        x2 = x + w
        y2 = y + h
        if x <= self.target_edge_margin_px and w < int(last_w * self.target_edge_comp_min_ratio):
            w = last_w
        elif x2 >= (self.frame_width - self.target_edge_margin_px) and w < int(last_w * self.target_edge_comp_min_ratio):
            x = max(0, x2 - last_w)
            w = self.frame_width - x if x + last_w > self.frame_width else last_w

        if y <= self.target_edge_margin_px and h < int(last_h * self.target_edge_comp_min_ratio):
            h = last_h
        elif y2 >= (self.frame_height - self.target_edge_margin_px) and h < int(last_h * self.target_edge_comp_min_ratio):
            y = max(0, y2 - last_h)
            h = self.frame_height - y if y + last_h > self.frame_height else last_h

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
        steps = max(4, self.target_border_sample_count)
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

    def _accept_center(self, candidate_center, last_center):
        if candidate_center is None or last_center is None:
            return True
        return self._distance_sq(candidate_center, last_center) <= (self.target_max_jump_px * self.target_max_jump_px)

    def _prepare_rect_image(self, img):
        rect_img = img.to_grayscale()
        rect_img.binary([self.rect_binary_threshold])
        return rect_img

    def _select_best_rect(self, rect_img, rects, reference_center, reference_rect):
        best = None
        best_score = None
        image_center = (self.frame_width // 2, self.frame_height // 2)

        for r in rects:
            raw_rect = r.rect()
            corners = r.corners()
            if raw_rect is None or corners is None or len(corners) != 4:
                continue

            rect = self._compensate_edge_rect(raw_rect, reference_rect)
            x, y, w, h = rect
            if w < self.target_min_w or h < self.target_min_h:
                continue
            area = w * h
            if area < self.target_min_area:
                continue
            if not self._rect_size_change_ok(rect, reference_rect):
                continue

            center = self._rect_center_from_corners(corners)
            border_hit_ratio = self._rect_border_hit_ratio(rect_img, rect)
            if border_hit_ratio < self.target_border_hit_ratio_min:
                continue
            if reference_center is not None:
                jump_sq = self._distance_sq(center, reference_center)
                if jump_sq > (self.target_reset_dist_px * self.target_reset_dist_px):
                    continue
            if reference_rect is not None:
                overlap_ratio = self._rect_overlap_ratio(rect, reference_rect)
                if overlap_ratio < self.target_min_overlap_ratio and (
                    reference_center is None
                    or self._distance_sq(center, reference_center) > (self.target_sticky_dist_px * self.target_sticky_dist_px)
                ):
                    continue

            aspect_penalty = int(self._rect_aspect_error(w, h) * self.target_aspect_penalty_scale)
            if reference_center is not None:
                distance_penalty = self._distance_sq(center, reference_center) // 10
            else:
                distance_penalty = self._distance_sq(center, image_center) // self.target_init_center_bias
            edge_penalty = 0
            if x <= 2 or y <= 2 or (x + w) >= (self.frame_width - 2) or (y + h) >= (self.frame_height - 2):
                edge_penalty = 3600
            center_bias_bonus = 0
            if self._distance_sq(center, image_center) <= (self.target_near_center_px * self.target_near_center_px):
                center_bias_bonus = 2000
            border_score_bonus = int(border_hit_ratio * self.target_border_score_scale)

            score = area - aspect_penalty - distance_penalty - edge_penalty + center_bias_bonus + border_score_bonus
            if best_score is None or score > best_score:
                best_score = score
                best = (rect, corners, center)

        return best
