import warnings
from pathlib import Path
import sys

warnings.filterwarnings("ignore")


SCRIPT_DIR = Path(__file__).resolve().parent
RELEASE_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "weights" / "best.pt").exists() else Path(
    "/home/zfh/Wyh/DaiMa/rtdetr/FEU-DETR-release"
)
sys.path.insert(0, str(RELEASE_ROOT))

from ultralytics import RTDETR


WEIGHTS = RELEASE_ROOT / "weights" / "best.pt"
DATA = "/home/zfh/Wyh/DaiMa/rtdetr/FEU-DETR/VisDrone.yaml"


if __name__ == "__main__":
    model = RTDETR(str(WEIGHTS))
    model.val(
        data=DATA,
        split="val",
        imgsz=640,
        batch=4,
        device="0",
        save_txt=False,
        save_json=False,
        project=str(RELEASE_ROOT / "runs" / "val"),
        name="feudetr-visdrone-val1",
    )
