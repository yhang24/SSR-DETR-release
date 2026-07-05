import argparse
from pathlib import Path

from ultralytics import RTDETR


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Validate the released FEU-DETR PyTorch checkpoint.")
    parser.add_argument("--weights", default=str(root / "weights" / "best.pt"), help="Path to best.pt.")
    parser.add_argument("--data", default=str(root / "VisDrone.yaml"), help="Path to dataset yaml.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=4, help="Validation batch size.")
    parser.add_argument("--device", default="0", help="CUDA device id, or cpu.")
    parser.add_argument("--project", default=str(root / "runs" / "val"), help="Output project directory.")
    parser.add_argument("--name", default="feudetr-visdrone", help="Validation run name.")
    parser.add_argument("--save-json", action="store_true", help="Save COCO-format JSON predictions.")
    parser.add_argument("--save-txt", action="store_true", help="Save YOLO-format txt predictions.")
    return parser.parse_args()


def main():
    args = parse_args()
    model = RTDETR(args.weights)
    model.val(
        data=args.data,
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        save_txt=args.save_txt,
        save_json=args.save_json,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
