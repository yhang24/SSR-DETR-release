# SSR-DETR Release

This is a minimal PyTorch validation package for the released FEU-DETR checkpoint.

## Contents

- `weights/best.pt`: sanitized PyTorch checkpoint for validation.
- `val.py`: validation entry point.
- `VisDrone.yaml`: dataset configuration template.
- `ultralytics/`: minimal runtime code required to load and validate the checkpoint.

Training code, experiment logs, ablation configurations, and other run artifacts are not included.

## Setup

```bash
pip install -r requirements.txt
```

The original experiment used Python with PyTorch CUDA. A CUDA-enabled PyTorch build is recommended for GPU validation.

## Dataset

Edit `VisDrone.yaml` and set `path` to your local VisDrone dataset root. An absolute path is recommended:

```yaml
path: /path/to/VisDrone
```

The expected directory layout is:

```text
VisDrone2019-DET-train/images
VisDrone2019-DET-val/images
VisDrone2019-DET-test-dev/images
```

## Validation

```bash
python val.py --data VisDrone.yaml --weights weights/best.pt --device 0 --batch 4 --imgsz 640
```

For CPU validation:

```bash
python val.py --device cpu
```
