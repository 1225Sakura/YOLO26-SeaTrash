# Scripts Directory

V11 material classification scripts for YOLO26 SeaTrash project.

## Structure

```
scripts/
├── training/
│   └── train_v11_material_8gpu.py  # V11 training (9-class, 8 GPU)
├── inference/
│   └── infer_video.py              # Video inference
└── data/
    ├── integrate_material_9class.py  # Dataset integration
    └── validate_material_9class.py   # Dataset validation
```

## Quick Start

### Training
```bash
python scripts/training/train_v11_material_8gpu.py
```

### Inference
```bash
python scripts/inference/infer_video.py --video path/to/video.mp4
```

### Dataset Validation
```bash
python scripts/data/validate_material_9class.py
```
