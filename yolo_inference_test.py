import gc
import sys
import nncase_runtime as nn
import ulab.numpy as np


KMODEL_PATH = "/sdcard/app/steelball_yolov8n_320_topk_uint8.kmodel"
TEST_VERSION = "topk-v1-no-aicube"


def print_free_memory(stage):
    gc.collect()
    try:
        print("[MEM] %s free=%d" % (stage, gc.mem_free()))
    except Exception:
        pass


def main():
    kpu = None
    input_tensor = None
    output_tensor = None
    try:
        print("[TEST] version:", TEST_VERSION)
        print("[TEST] 1/5 loading TopK kmodel...")
        kpu = nn.kpu()
        kpu.load_kmodel(KMODEL_PATH)
        print(
            "[TEST] 1/5 load OK, inputs=%d outputs=%d"
            % (kpu.inputs_size(), kpu.outputs_size())
        )
        print_free_memory("after kmodel load")

        print("[TEST] 2/5 creating constant uint8 input...")
        input_data = np.ones((1, 3, 320, 320), dtype=np.uint8) * 114
        input_tensor = nn.from_numpy(input_data)
        kpu.set_input_tensor(0, input_tensor)
        print("[TEST] 2/5 input OK")

        print("[TEST] 3/5 KPU run START")
        kpu.run()
        print("[TEST] 3/5 KPU run OK")

        print("[TEST] 4/5 reading output...")
        output_tensor = kpu.get_output_tensor(0)
        output = output_tensor.to_numpy()
        print("[TEST] output shape=%s" % str(output.shape))
        if output.shape != (1, 5, 32):
            raise RuntimeError("unexpected TopK output shape")
        print("[TEST] max score=%.4f" % float(np.max(output[0, 4])))
        print("[TEST] 4/5 output OK")
        print("[TEST] 5/5 TOPK INFERENCE TEST PASSED")
    except Exception as error:
        print("[TEST] FAILED:", error)
        sys.print_exception(error)
    finally:
        del output_tensor
        del input_tensor
        if kpu is not None:
            kpu.__del__()
        nn.shrink_memory_pool()
        gc.collect()


if __name__ == "__main__":
    main()
