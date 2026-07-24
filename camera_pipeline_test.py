from libs.PipeLine import PipeLine
import gc
import os
import sys
import time


RGB888P_SIZE = [640, 360]
DISPLAY_SIZE = [640, 360]
DISPLAY_MODE = "virt"
CAMERA_FPS = 30
FRAME_COUNT = 10
RETRY_COUNT = 3
TEST_VERSION = "camera-30fps-v1"


def get_frame_with_retry(pipeline, frame_index):
    last_error = None
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            frame = pipeline.get_frame()
            print(
                "[CAMERA TEST] frame %d/%d shape=%s"
                % (frame_index, FRAME_COUNT, str(frame.shape))
            )
            return frame
        except RuntimeError as error:
            last_error = error
            print(
                "[CAMERA TEST] frame %d retry %d/%d: %s"
                % (frame_index, attempt, RETRY_COUNT, str(error))
            )
            if attempt < RETRY_COUNT:
                time.sleep_ms(100)
    raise last_error


def main():
    pipeline = None
    try:
        print("[CAMERA TEST] version:", TEST_VERSION)
        print("[CAMERA TEST] board:", os.uname()[-1])
        print("[CAMERA TEST] creating pipeline at 30 fps...")
        pipeline = PipeLine(
            rgb888p_size=RGB888P_SIZE,
            display_mode=DISPLAY_MODE,
            display_size=DISPLAY_SIZE,
        )
        pipeline.create(fps=CAMERA_FPS)
        print("[CAMERA TEST] pipeline ready")

        for frame_index in range(1, FRAME_COUNT + 1):
            frame = get_frame_with_retry(pipeline, frame_index)
            del frame
            if frame_index % 5 == 0:
                gc.collect()
        print("[CAMERA TEST] PASSED: captured 10 frames")
    except Exception as error:
        print("[CAMERA TEST] FAILED:", error)
        sys.print_exception(error)
    finally:
        if pipeline is not None:
            pipeline.destroy()
        gc.collect()


if __name__ == "__main__":
    main()
