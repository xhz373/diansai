import argparse
import os
from pathlib import Path

import nncase
import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image


ROOT = Path(__file__).resolve().parent
DEFAULT_ONNX_PATH = ROOT / "steelball_yolov8n_320.onnx"
DEFAULT_KMODEL_PATH = ROOT / "steelball_yolov8n_320_uint8.kmodel"
DEFAULT_CALIBRATION_DIR = Path(
    r"D:\steelball_project\yolo_steelball_dataset\images"
)
INPUT_SIZE = 320
PAD_VALUE = 114
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def letterbox_image(image_path):
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
        scale = min(INPUT_SIZE / image.width, INPUT_SIZE / image.height)
        resized_width = max(1, int(image.width * scale))
        resized_height = max(1, int(image.height * scale))
        image = image.resize(
            (resized_width, resized_height), Image.Resampling.BILINEAR
        )

        canvas = Image.new(
            "RGB", (INPUT_SIZE, INPUT_SIZE), (PAD_VALUE,) * 3
        )
        left = (INPUT_SIZE - resized_width) // 2
        top = (INPUT_SIZE - resized_height) // 2
        canvas.paste(image, (left, top))

        image_data = np.asarray(canvas, dtype=np.uint8)
        return np.ascontiguousarray(image_data.transpose(2, 0, 1)[None])


def select_calibration_images(calibration_dir, samples_count):
    image_paths = sorted(
        path
        for path in calibration_dir.rglob("*")
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not image_paths:
        raise RuntimeError(f"校准目录没有图片: {calibration_dir}")

    selected_count = min(samples_count, len(image_paths))
    indices = np.linspace(
        0, len(image_paths) - 1, selected_count, dtype=np.int32
    )
    return [image_paths[int(index)] for index in indices]


def create_compile_options(dump_dir):
    options = nncase.CompileOptions()
    options.target = "k230"
    options.dump_ir = False
    options.dump_asm = False
    options.dump_dir = str(dump_dir)
    options.input_type = "uint8"
    options.input_shape = [1, 3, INPUT_SIZE, INPUT_SIZE]
    options.input_range = [0, 255]
    options.input_layout = "NCHW"
    options.preprocess = True
    options.output_layout = ""
    options.mean = [0, 0, 0]
    options.std = [255, 255, 255]
    return options


def validate_onnx(onnx_path):
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    input_shape = [
        dimension.dim_value
        for dimension in model.graph.input[0].type.tensor_type.shape.dim
    ]
    output_shape = [
        dimension.dim_value
        for dimension in model.graph.output[0].type.tensor_type.shape.dim
    ]
    if input_shape != [1, 3, INPUT_SIZE, INPUT_SIZE]:
        raise RuntimeError(f"ONNX 输入形状错误: {input_shape}")
    expected_candidates = sum(
        (INPUT_SIZE // stride) ** 2 for stride in (8, 16, 32)
    )
    if output_shape != [1, 5, expected_candidates]:
        raise RuntimeError(f"ONNX 输出形状错误: {output_shape}")
    print(f"ONNX 接口检查通过: {input_shape} -> {output_shape}")


def compile_kmodel(onnx_path, kmodel_path, calibration_paths, dump_dir):
    compiler = nncase.Compiler(create_compile_options(dump_dir))
    compiler.import_onnx(onnx_path.read_bytes(), nncase.ImportOptions())

    calibration_data = [
        [letterbox_image(image_path)] for image_path in calibration_paths
    ]
    ptq_options = nncase.PTQTensorOptions()
    ptq_options.samples_count = len(calibration_data)
    ptq_options.quant_type = "uint8"
    ptq_options.w_quant_type = "uint8"
    ptq_options.calibrate_method = "Kld"
    ptq_options.set_tensor_data(calibration_data)
    compiler.use_ptq(ptq_options)

    print(f"开始编译，PTQ 校准图片: {len(calibration_data)} 张")
    compiler.compile()
    kmodel_path.write_bytes(compiler.gencode_tobytes())
    print(f"K230 模型已生成: {kmodel_path}")


def normalize_output(output):
    output = np.asarray(output)
    expected_candidates = sum(
        (INPUT_SIZE // stride) ** 2 for stride in (8, 16, 32)
    )
    if output.shape == (5, expected_candidates):
        return output
    if output.shape == (expected_candidates, 5):
        return output.transpose(1, 0)
    if output.shape == (1, 5, expected_candidates):
        return output[0]
    if output.shape == (1, expected_candidates, 5):
        return output[0].transpose(1, 0)
    raise RuntimeError(f"不支持的输出形状: {output.shape}")


def box_iou(first, second):
    left = max(first[1], second[1])
    top = max(first[2], second[2])
    right = min(first[3], second[3])
    bottom = min(first[4], second[4])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[3] - first[1]) * max(
        0.0, first[4] - first[2]
    )
    second_area = max(0.0, second[3] - second[1]) * max(
        0.0, second[4] - second[2]
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def decode_detections(
    output, confidence_threshold=0.20, nms_threshold=0.45, limit=8
):
    output = normalize_output(output)
    scores = output[4]
    indices = np.flatnonzero(scores >= confidence_threshold)
    candidates = []
    for index in indices:
        center_x, center_y, width, height = output[:4, index]
        candidates.append(
            (
                float(scores[index]),
                float(center_x - width / 2),
                float(center_y - height / 2),
                float(center_x + width / 2),
                float(center_y + height / 2),
            )
        )
    candidates.sort(reverse=True)

    detections = []
    for candidate in candidates:
        if all(
            box_iou(candidate, selected) <= nms_threshold
            for selected in detections
        ):
            detections.append(candidate)
            if len(detections) >= limit:
                break
    return detections


def output_summary(output, confidence_threshold=0.20):
    output = normalize_output(output)
    scores = output[4]
    detections = decode_detections(output, confidence_threshold)
    return {
        "candidates": int(np.sum(scores >= confidence_threshold)),
        "max_score": float(np.max(scores)),
        "detections": len(detections),
        "detection_scores": [round(item[0], 4) for item in detections],
    }


def validate_kmodel(onnx_path, kmodel_path, image_path):
    input_uint8 = letterbox_image(image_path)
    input_float = input_uint8.astype(np.float32) / 255.0

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    onnx_output = session.run(None, {session.get_inputs()[0].name: input_float})[0]

    simulator = nncase.Simulator()
    simulator.load_model(kmodel_path.read_bytes())
    simulator.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(input_uint8))
    simulator.run()
    kmodel_output = simulator.get_output_tensor(0).to_numpy()

    onnx_normalized = normalize_output(onnx_output)
    kmodel_normalized = normalize_output(kmodel_output)
    box_mae = float(
        np.mean(np.abs(onnx_normalized[:4] - kmodel_normalized[:4]))
    )
    score_mae = float(
        np.mean(np.abs(onnx_normalized[4] - kmodel_normalized[4]))
    )

    print(f"模拟器输入形状: {simulator.get_input_shape(0)}")
    print(f"模拟器输出形状: {simulator.get_output_shape(0)}")
    print(f"ONNX 预测摘要: {output_summary(onnx_output)}")
    print(f"KMODEL 预测摘要: {output_summary(kmodel_output)}")
    print(f"量化误差: box_mae={box_mae:.4f}, score_mae={score_mae:.6f}")


def parse_args():
    parser = argparse.ArgumentParser(description="转换 YOLOv8n 到 K230 kmodel")
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_KMODEL_PATH)
    parser.add_argument(
        "--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR
    )
    parser.add_argument("--samples", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    onnx_path = args.onnx.resolve()
    kmodel_path = args.output.resolve()
    calibration_dir = args.calibration_dir.resolve()
    dump_dir = ROOT / "tmp_yolo"
    dump_dir.mkdir(exist_ok=True)

    if not onnx_path.exists():
        raise FileNotFoundError(f"找不到 ONNX: {onnx_path}")
    if not calibration_dir.exists():
        raise FileNotFoundError(f"找不到校准目录: {calibration_dir}")

    validate_onnx(onnx_path)
    calibration_paths = select_calibration_images(
        calibration_dir, args.samples
    )
    compile_kmodel(
        onnx_path, kmodel_path, calibration_paths, dump_dir
    )
    validate_kmodel(onnx_path, kmodel_path, calibration_paths[0])


if __name__ == "__main__":
    main()
