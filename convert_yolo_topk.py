from pathlib import Path

import nncase
import numpy as np
import onnx
import onnxruntime as ort

from convert_yolo import (
    ROOT,
    create_compile_options,
    letterbox_image,
    normalize_output as normalize_standard_output,
    select_calibration_images,
)


ONNX_PATH = ROOT / "steelball_yolov8n_320_topk.onnx"
KMODEL_PATH = ROOT / "steelball_yolov8n_320_topk_uint8.kmodel"
STANDARD_KMODEL_PATH = ROOT / "steelball_yolov8n_320_uint8.kmodel"
CALIBRATION_DIR = Path(r"D:\steelball_project\yolo_steelball_dataset\images")
TOPK_CANDIDATES = 32


def shape_of(value_info):
    return [
        dimension.dim_value
        for dimension in value_info.type.tensor_type.shape.dim
    ]


def normalize_output(output):
    output = np.asarray(output)
    if output.shape == (1, 5, TOPK_CANDIDATES):
        return output[0]
    if output.shape == (5, TOPK_CANDIDATES):
        return output
    if output.shape == (1, TOPK_CANDIDATES, 5):
        return output[0].transpose(1, 0)
    if output.shape == (TOPK_CANDIDATES, 5):
        return output.transpose(1, 0)
    raise RuntimeError("unexpected TopK output shape: %s" % (output.shape,))


def unordered_match_mae(reference, actual):
    remaining = list(range(reference.shape[1]))
    box_errors = []
    score_errors = []
    for actual_index in range(actual.shape[1]):
        best_index = min(
            remaining,
            key=lambda reference_index: float(
                np.mean(
                    np.abs(
                        reference[:, reference_index]
                        - actual[:, actual_index]
                    )
                )
            ),
        )
        box_errors.append(
            float(
                np.mean(
                    np.abs(
                        reference[:4, best_index]
                        - actual[:4, actual_index]
                    )
                )
            )
        )
        score_errors.append(
            abs(
                float(reference[4, best_index])
                - float(actual[4, actual_index])
            )
        )
        remaining.remove(best_index)
    return float(np.mean(box_errors)), float(np.mean(score_errors))


def check_onnx():
    model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(model)
    input_shape = shape_of(model.graph.input[0])
    output_shape = shape_of(model.graph.output[0])
    if input_shape != [1, 3, 320, 320]:
        raise RuntimeError("unexpected input shape: %s" % input_shape)
    if output_shape != [1, 5, TOPK_CANDIDATES]:
        raise RuntimeError("unexpected output shape: %s" % output_shape)
    print("ONNX interface:", input_shape, "->", output_shape)


def compile_kmodel(calibration_paths):
    dump_dir = ROOT / "tmp_yolo_topk"
    dump_dir.mkdir(exist_ok=True)
    compiler = nncase.Compiler(create_compile_options(dump_dir))
    compiler.import_onnx(ONNX_PATH.read_bytes(), nncase.ImportOptions())

    calibration_data = [
        [letterbox_image(image_path)] for image_path in calibration_paths
    ]
    options = nncase.PTQTensorOptions()
    options.samples_count = len(calibration_data)
    options.quant_type = "uint8"
    options.w_quant_type = "uint8"
    options.calibrate_method = "Kld"
    options.set_tensor_data(calibration_data)
    compiler.use_ptq(options)

    print("Compiling with %d PTQ images..." % len(calibration_data))
    compiler.compile()
    KMODEL_PATH.write_bytes(compiler.gencode_tobytes())
    print("Kmodel generated:", KMODEL_PATH)


def validate_kmodel(image_path):
    input_uint8 = letterbox_image(image_path)
    input_float = input_uint8.astype(np.float32) / 255.0
    session = ort.InferenceSession(
        str(ONNX_PATH), providers=["CPUExecutionProvider"]
    )
    onnx_output = normalize_output(
        session.run(
            None, {session.get_inputs()[0].name: input_float}
        )[0]
    )

    simulator = nncase.Simulator()
    simulator.load_model(KMODEL_PATH.read_bytes())
    simulator.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(input_uint8))
    simulator.run()
    kmodel_output = normalize_output(
        simulator.get_output_tensor(0).to_numpy()
    )

    standard_simulator = nncase.Simulator()
    standard_simulator.load_model(STANDARD_KMODEL_PATH.read_bytes())
    standard_simulator.set_input_tensor(
        0, nncase.RuntimeTensor.from_numpy(input_uint8)
    )
    standard_simulator.run()
    standard_output = normalize_standard_output(
        standard_simulator.get_output_tensor(0).to_numpy()
    )
    standard_indices = np.argsort(standard_output[4])[::-1]
    standard_topk = standard_output[:, standard_indices[:TOPK_CANDIDATES]]

    box_mae = float(np.mean(np.abs(onnx_output[:4] - kmodel_output[:4])))
    score_mae = float(np.mean(np.abs(onnx_output[4] - kmodel_output[4])))
    standard_box_mae = float(
        np.mean(np.abs(standard_topk[:4] - kmodel_output[:4]))
    )
    standard_score_mae = float(
        np.mean(np.abs(standard_topk[4] - kmodel_output[4]))
    )
    unordered_box_mae, unordered_score_mae = unordered_match_mae(
        standard_topk, kmodel_output
    )
    print("Simulator input:", simulator.get_input_shape(0))
    print("Simulator output:", simulator.get_output_shape(0))
    print(
        "Top score: ONNX=%.4f KMODEL=%.4f"
        % (float(onnx_output[4, 0]), float(kmodel_output[4, 0]))
    )
    print("Quantization MAE: boxes=%.4f scores=%.6f" % (box_mae, score_mae))
    print(
        "Versus standard kmodel TopK: boxes=%.4f scores=%.6f"
        % (standard_box_mae, standard_score_mae)
    )
    print(
        "Unordered candidate match: boxes=%.4f scores=%.6f"
        % (unordered_box_mae, unordered_score_mae)
    )
    print("ONNX scores:", onnx_output[4, :8])
    print("Standard scores:", standard_topk[4, :8])
    print("TopK kmodel scores:", kmodel_output[4, :8])


def main():
    check_onnx()
    calibration_paths = select_calibration_images(CALIBRATION_DIR, 50)
    compile_kmodel(calibration_paths)
    validate_kmodel(calibration_paths[0])


if __name__ == "__main__":
    main()
