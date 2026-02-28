#!/usr/bin/env python3
"""
Marine Debris Detection - v4.0 Training Script
8× RTX 4090 Optimized Configuration

Performance Target:
- mAP50: ≥ 90%
- Recall: ≥ 75%
- Precision: ≥ 85%

Hardware: 8× NVIDIA RTX 4090 (24GB each, 192GB total)
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Configuration
MODEL_PATH = str(PROJECT_ROOT / "outputs/training_archive/v1.0/models/best.pt")
DATA_YAML = str(PROJECT_ROOT / "data/datasets/marine_debris_unified/data.yaml")
OUTPUT_DIR = str(PROJECT_ROOT / "outputs/runs/train")
PROJECT_NAME = "marine_debris_v4.0_8gpu"

# 8-GPU Optimized Training Configuration
TRAINING_CONFIG = {
    # Hardware Configuration
    "device": "0,1,2,3,4,5,6,7",  # 8× RTX 4090
    "batch": 80,  # 8 GPU × 10 per GPU
    "workers": 16,  # 2 workers per GPU

    # Training Parameters
    "epochs": 1000,
    "imgsz": 640,
    "patience": 200,  # Longer patience for 8-GPU training
    "save_period": 50,  # Save checkpoint every 50 epochs
    "cache": True,  # Cache images for faster training
    "amp": True,  # Automatic Mixed Precision

    # Optimizer Configuration
    "optimizer": "AdamW",
    "lr0": 0.001,  # Initial learning rate
    "lrf": 0.01,  # Final learning rate (lr0 * lrf)
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 5.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,

    # Loss Weights (Optimized for recall)
    "box": 7.5,
    "cls": 0.25,  # Reduced for higher recall
    "dfl": 1.5,

    # Data Augmentation (Water-specific)
    "hsv_h": 0.025,  # Water color hue variation
    "hsv_s": 0.8,    # Saturation for underwater
    "hsv_v": 0.5,    # Value/brightness
    "degrees": 5.0,  # Rotation
    "translate": 0.1,
    "scale": 0.5,
    "shear": 2.0,
    "perspective": 0.0005,  # Water surface angle
    "flipud": 0.2,  # Vertical flip for underwater
    "fliplr": 0.5,  # Horizontal flip
    "mosaic": 1.0,
    "mixup": 0.15,
    "copy_paste": 0.1,

    # Regularization
    "label_smoothing": 0.1,
    "dropout": 0.0,

    # Validation
    "val": True,
    "plots": True,
    "save": True,
    "save_txt": False,
    "save_conf": False,
}

def main():
    """Execute v4.0 training with 8-GPU configuration"""

    print("="*80)
    print("Marine Debris Detection - v4.0 Training")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: YOLO26x")
    print(f"  Base Weights: {MODEL_PATH}")
    print(f"  Dataset: {DATA_YAML}")
    print(f"  GPUs: 8× RTX 4090")
    print(f"  Batch Size: {TRAINING_CONFIG['batch']} (10 per GPU)")
    print(f"  Epochs: {TRAINING_CONFIG['epochs']}")
    print(f"  Workers: {TRAINING_CONFIG['workers']}")
    print(f"  Image Size: {TRAINING_CONFIG['imgsz']}")
    print(f"\nOptimizations:")
    print(f"  - Reduced cls loss: {TRAINING_CONFIG['cls']} (for higher recall)")
    print(f"  - Water-specific augmentation")
    print(f"  - Label smoothing: {TRAINING_CONFIG['label_smoothing']}")
    print(f"  - AMP enabled: {TRAINING_CONFIG['amp']}")
    print(f"  - Image caching: {TRAINING_CONFIG['cache']}")
    print(f"\nTarget Performance:")
    print(f"  - mAP50: ≥ 90%")
    print(f"  - Recall: ≥ 75%")
    print(f"  - Precision: ≥ 85%")
    print("="*80)

    # Verify files exist
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Error: Model file not found: {MODEL_PATH}")
        sys.exit(1)

    if not os.path.exists(DATA_YAML):
        print(f"\n❌ Error: Dataset config not found: {DATA_YAML}")
        sys.exit(1)

    # Import YOLO
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n❌ Error: ultralytics not installed")
        print("Install with: pip install ultralytics")
        sys.exit(1)

    # Load model
    print(f"\n📦 Loading base model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # Start training
    print(f"\n🚀 Starting v4.0 training...")
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n{'='*80}\n")

    try:
        results = model.train(
            data=DATA_YAML,
            project=OUTPUT_DIR,
            name=PROJECT_NAME,
            exist_ok=True,
            **TRAINING_CONFIG
        )

        print(f"\n{'='*80}")
        print("✅ Training completed successfully!")
        print(f"⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n📊 Results saved to: {OUTPUT_DIR}/{PROJECT_NAME}")
        print(f"📈 Best model: {OUTPUT_DIR}/{PROJECT_NAME}/weights/best.pt")
        print("="*80)

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ Training failed: {str(e)}")
        print("="*80)
        sys.exit(1)

if __name__ == "__main__":
    main()

