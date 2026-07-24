from libs.PipeLine import PipeLine
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
import gc
import sys
import time
import nncase_runtime as nn
import ulab.numpy as np


KMODEL_PATH = "/sdcard/app/steelball_yolov8n_320_topk_uint8.kmodel"
MODEL_INPUT_SIZE = [320, 320]
RGB888P_SIZE = [640, 360]
DISPLAY_MODE = "virt"
DISPLAY_SIZE = [640, 360]
LABELS = ["steel_ball"]
CONFIDENCE_THRESHOLD = 0.20
NMS_THRESHOLD = 0.45
TOPK_CANDIDATES = 32
MAX_DETECTIONS = 8
PROFILE_FRAMES = 3
BOX_COLOR = (255, 0, 255, 0)
TEXT_COLOR = (255, 255, 255, 0)
CAMERA_FPS = 30
FRAME_RETRY_COUNT = 3


def align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def center_pad_params(input_size, output_size):
    scale = min(
        output_size[0] / input_size[0],
        output_size[1] / input_size[1],
    )
    resized_width = int(input_size[0] * scale)
    resized_height = int(input_size[1] * scale)
    horizontal_padding = (output_size[0] - resized_width) / 2
    vertical_padding = (output_size[1] - resized_height) / 2
    top = int(round(vertical_padding - 0.1))
    bottom = int(round(vertical_padding + 0.1))
    left = int(round(horizontal_padding - 0.1))
    right = int(round(horizontal_padding + 0.1))
    return top, bottom, left, right


def get_frame_with_retry(pipeline):
    last_error = None
    for attempt in range(1, FRAME_RETRY_COUNT + 1):
        try:
            return pipeline.get_frame()
        except RuntimeError as error:
            last_error = error
            print(
                "[CAMERA] snapshot retry %d/%d: %s"
                % (attempt, FRAME_RETRY_COUNT, str(error))
            )
            if attempt < FRAME_RETRY_COUNT:
                time.sleep_ms(100)
    raise last_error


class YoloV8TopKApp(AIBase):
    def __init__(
        self,
        kmodel_path,
        model_input_size,
        rgb888p_size,
        display_size,
        labels,
        confidence_threshold=0.20,
        nms_threshold=0.45,
        debug_mode=0,
    ):
        aligned_rgb_size = [align_up(rgb888p_size[0], 16), rgb888p_size[1]]
        super().__init__(
            kmodel_path, model_input_size, aligned_rgb_size, debug_mode
        )
        self.model_input_size = model_input_size
        self.rgb888p_size = aligned_rgb_size
        self.display_size = [align_up(display_size[0], 16), display_size[1]]
        self.labels = labels
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.profile_frames_remaining = PROFILE_FRAMES
        self.output_shapes_printed = False
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(
            nn.ai2d_format.NCHW_FMT,
            nn.ai2d_format.NCHW_FMT,
            np.uint8,
            np.uint8,
        )

    def config_preprocess(self):
        top, bottom, left, right = center_pad_params(
            self.rgb888p_size, self.model_input_size
        )
        self.ai2d.pad(
            [0, 0, 0, 0, top, bottom, left, right],
            0,
            [114, 114, 114],
        )
        self.ai2d.resize(
            nn.interp_method.tf_bilinear,
            nn.interp_mode.half_pixel,
        )
        self.ai2d.build(
            [1, 3, self.rgb888p_size[1], self.rgb888p_size[0]],
            [1, 3, self.model_input_size[1], self.model_input_size[0]],
        )

    def postprocess(self, results):
        if len(results) != 1:
            raise RuntimeError("TopK YOLO model must have exactly one output")
        output = results[0]
        if not self.output_shapes_printed:
            print("[YOLO] raw output shape:", output.shape)
            self.output_shapes_printed = True

        start = time.ticks_ms()
        if output.shape == (1, 5, TOPK_CANDIDATES):
            output = output[0]
        elif output.shape == (1, TOPK_CANDIDATES, 5):
            output = output[0].transpose()
        elif output.shape == (TOPK_CANDIDATES, 5):
            output = output.transpose()
        elif output.shape != (5, TOPK_CANDIDATES):
            raise RuntimeError("unexpected TopK output shape: %s" % str(output.shape))

        scale = min(
            self.model_input_size[0] / self.rgb888p_size[0],
            self.model_input_size[1] / self.rgb888p_size[1],
        )
        resized_width = int(self.rgb888p_size[0] * scale)
        resized_height = int(self.rgb888p_size[1] * scale)
        pad_x = (self.model_input_size[0] - resized_width) / 2
        pad_y = (self.model_input_size[1] - resized_height) / 2

        candidates = []
        for index in range(TOPK_CANDIDATES):
            score = float(output[4, index])
            if score < self.confidence_threshold:
                break
            center_x = float(output[0, index])
            center_y = float(output[1, index])
            width = float(output[2, index])
            height = float(output[3, index])
            x1 = (center_x - width / 2 - pad_x) / scale
            y1 = (center_y - height / 2 - pad_y) / scale
            x2 = (center_x + width / 2 - pad_x) / scale
            y2 = (center_y + height / 2 - pad_y) / scale
            x1 = max(0.0, min(float(self.rgb888p_size[0] - 1), x1))
            y1 = max(0.0, min(float(self.rgb888p_size[1] - 1), y1))
            x2 = max(0.0, min(float(self.rgb888p_size[0] - 1), x2))
            y2 = max(0.0, min(float(self.rgb888p_size[1] - 1), y2))
            if x2 > x1 and y2 > y1:
                candidates.append([0, score, x1, y1, x2, y2])

        detections = []
        for candidate in candidates:
            keep = True
            for selected in detections:
                if self.box_iou(candidate, selected) > self.nms_threshold:
                    keep = False
                    break
            if keep:
                detections.append(candidate)
                if len(detections) >= MAX_DETECTIONS:
                    break
        if self.profile_frames_remaining > 0:
            print(
                "[PROFILE] topk_postprocess=%d ms"
                % time.ticks_diff(time.ticks_ms(), start)
            )
        return detections

    @staticmethod
    def box_iou(first, second):
        left = max(first[2], second[2])
        top = max(first[3], second[3])
        right = min(first[4], second[4])
        bottom = min(first[5], second[5])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = max(0.0, first[4] - first[2]) * max(
            0.0, first[5] - first[3]
        )
        second_area = max(0.0, second[4] - second[2]) * max(
            0.0, second[5] - second[3]
        )
        union = first_area + second_area - intersection
        if union <= 0.0:
            return 0.0
        return intersection / union

    def draw_result(self, pipeline, detections, fps):
        pipeline.osd_img.clear()
        detection_count = 0 if detections is None else len(detections)
        pipeline.osd_img.draw_string_advanced(
            8,
            8,
            24,
            "balls=%d  fps=%.1f" % (detection_count, fps),
            color=TEXT_COLOR,
        )
        if detections is None:
            return

        for detection in detections:
            class_index = int(detection[0])
            score = float(detection[1])
            x1 = float(detection[2])
            y1 = float(detection[3])
            x2 = float(detection[4])
            y2 = float(detection[5])
            screen_x = int(x1 * self.display_size[0] / self.rgb888p_size[0])
            screen_y = int(y1 * self.display_size[1] / self.rgb888p_size[1])
            screen_width = max(
                2,
                int(
                    (x2 - x1)
                    * self.display_size[0]
                    / self.rgb888p_size[0]
                ),
            )
            screen_height = max(
                2,
                int(
                    (y2 - y1)
                    * self.display_size[1]
                    / self.rgb888p_size[1]
                ),
            )
            pipeline.osd_img.draw_rectangle(
                screen_x,
                screen_y,
                screen_width,
                screen_height,
                color=BOX_COLOR,
                thickness=4,
            )
            label_y = max(34, screen_y) - 34
            pipeline.osd_img.draw_string_advanced(
                screen_x,
                label_y,
                28,
                "%s %.2f" % (self.labels[class_index], score),
                color=BOX_COLOR,
            )


def main():
    pipeline = None
    detector = None
    try:
        print("[YOLO] initializing TopK pipeline without aicube...")
        pipeline = PipeLine(
            rgb888p_size=RGB888P_SIZE,
            display_mode=DISPLAY_MODE,
            display_size=DISPLAY_SIZE,
        )
        pipeline.create(fps=CAMERA_FPS)
        display_size = pipeline.get_display_size()

        detector = YoloV8TopKApp(
            KMODEL_PATH,
            MODEL_INPUT_SIZE,
            RGB888P_SIZE,
            display_size,
            LABELS,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            nms_threshold=NMS_THRESHOLD,
            debug_mode=1,
        )
        detector.config_preprocess()
        print("[YOLO] ready, model:", KMODEL_PATH)

        frame_count = 0
        total_frames = 0
        fps = 0.0
        fps_start = time.ticks_ms()
        while True:
            frame_start = time.ticks_ms()
            capture_start = frame_start
            image_np = get_frame_with_retry(pipeline)
            capture_ms = time.ticks_diff(time.ticks_ms(), capture_start)

            run_start = time.ticks_ms()
            detections = detector.run(image_np)
            run_ms = time.ticks_diff(time.ticks_ms(), run_start)

            draw_start = time.ticks_ms()
            detector.draw_result(pipeline, detections, fps)
            pipeline.show_image()
            draw_ms = time.ticks_diff(time.ticks_ms(), draw_start)
            if total_frames % 10 == 0:
                gc.collect()

            frame_count += 1
            total_frames += 1
            now = time.ticks_ms()
            if total_frames <= PROFILE_FRAMES:
                print(
                    "[PROFILE] capture=%d run=%d draw_show=%d total=%d ms"
                    % (
                        capture_ms,
                        run_ms,
                        draw_ms,
                        time.ticks_diff(now, frame_start),
                    )
                )
                detector.profile_frames_remaining -= 1
                if total_frames == PROFILE_FRAMES:
                    detector.debug_mode = 0

            elapsed = time.ticks_diff(now, fps_start)
            if elapsed >= 1000:
                fps = frame_count * 1000.0 / elapsed
                max_score = 0.0
                if detections is not None and len(detections) > 0:
                    max_score = float(detections[0][1])
                print(
                    "[YOLO] balls=%d max=%.3f fps=%.1f"
                    % (
                        0 if detections is None else len(detections),
                        max_score,
                        fps,
                    )
                )
                frame_count = 0
                fps_start = now
    except KeyboardInterrupt:
        print("[YOLO] stopped")
    except Exception as error:
        print("[YOLO] failed:", error)
        sys.print_exception(error)
    finally:
        if detector is not None:
            detector.deinit()
        if pipeline is not None:
            pipeline.destroy()
        gc.collect()


if __name__ == "__main__":
    main()
