import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "steelball.pt"
DEFAULT_INPUT_SIZE = 320


def parse_args():
    parser = argparse.ArgumentParser(description="导出 K230 使用的 YOLOv8 ONNX")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_INPUT_SIZE)
    return parser.parse_args()


def main():
    args = parse_args()
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"找不到模型: {WEIGHTS_PATH}")

    model = YOLO(str(WEIGHTS_PATH))
    exported_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        batch=1,
        dynamic=False,
        simplify=False,
        opset=11,
        nms=False,
    )
    exported_path = Path(exported_path).resolve()
    target_path = ROOT / f"steelball_yolov8n_{args.imgsz}.onnx"
    if target_path.exists():
        target_path.unlink()
    exported_path.replace(target_path)
    print(f"YOLO ONNX 已导出: {target_path}")


if __name__ == "__main__":
    main()
