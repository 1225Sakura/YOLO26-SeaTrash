#!/usr/bin/env python3
"""
Marine Debris Detection - v4.0 Training Script
8× RTX 4090 Optimized Configuration

Based on deep analysis of v1.0, v2.0, v3.0 training data:
- v3.0 achieved 88.31% mAP50 at epoch 78, then degraded to 65.80% by epoch 228
- Key issue: patience=150 too long, model severely overfit
- Solution: Reduce patience, optimize loss weights, improve LR schedule

Performance Target:
- mAP50: ≥ 90% (current best: 88.31%)
- Recall: ≥ 85% (current best: 83.85%)
- Precision: ≥ 88% (current best: 86.94%)

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
MODEL_PATH = str(PROJECT_ROOT / "outputs/training_archive/v3.0/models/best.pt")  # Use v3.0 as base
DATA_YAML = str(PROJECT_ROOT / "data/datasets/marine_debris_unified/data.yaml")
OUTPUT_DIR = str(PROJECT_ROOT / "outputs/runs/train")
PROJECT_NAME = "marine_debris_v4.0_8gpu_optimized"

# 8-GPU Optimized Training Configuration (Data-Driven)
TRAINING_CONFIG = {
    # Hardware Configuration
    "device": "0,1,2,3,4,5,6,7",  # 8× RTX 4090
    "batch": 80,  # 8 GPU × 10 per GPU
    "workers": 16,  # 2 workers per GPU

    # Training Parameters (OPTIMIZED based on v3.0 analysis)
    "epochs": 600,  # Reduced from 1000 to avoid overfitting
    "imgsz": 640,
    "patience": 100,  # CRITICAL: Reduced from 200 (v3.0 overfit after 150 epochs)
    "save_period": 25,  # More frequent checkpoints
    "cache": True,  # Cache images for faster training
    "amp": True,  # Automatic Mixed Precision
    "close_mosaic": 50,  # Disable mosaic last 50 epochs for fine-tuning

    # Optimizer Configuration
    "optimizer": "AdamW",
    "lr0": 0.001,  # Initial learning rate
    "lrf": 0.05,  # OPTIMIZED: Increased from 0.01 (slower decay, v3.0 peaked too early)
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 10.0,  # OPTIMIZED: Increased from 5 for better convergence
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,

    # Loss Weights (DATA-DRIVEN optimization)
    "box": 7.5,
    "cls": 0.2,  # CRITICAL: Reduced from 0.25 (analysis shows lower cls → higher recall)
    "dfl": 1.8,  # OPTIMIZED: Increased from 1.5 for better bbox regression

    # Data Augmentation (Enhanced based on analysis)
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
    "mixup": 0.2,  # OPTIMIZED: Increased from 0.15
    "copy_paste": 0.15,  # OPTIMIZED: Increased from 0.1

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
    "save": True,
    "save_txt": False,
    "save_conf": False,
}

def main():
    """Execute v4.0 training with data-driven 8-GPU configuration"""

    print("="*80)
    print("Marine Debris Detection - v4.0 Training (Data-Driven Optimization)")
    print("="*80)
    print(f"\n📊 Training Data Analysis Summary:")
    print(f"  v1.0: 75.63% mAP50 (baseline)")
    print(f"  v2.0: 87.79% mAP50 @ epoch 31, degraded to 72.06% @ epoch 183")
    print(f"  v3.0: 88.31% mAP50 @ epoch 78, degraded to 65.80% @ epoch 228")
    print(f"\n⚠️  Critical Finding: Severe overfitting after best epoch")
    print(f"  v2.0: +152 epochs after best → -17.96% mAP50")
    print(f"  v3.0: +150 epochs after best → -25.49% mAP50")

    print(f"\n🔧 v4.0 Key Optimizations:")
    print(f"  1. Patience: 150 → 100 (prevent overfitting)")
    print(f"  2. Epochs: 1000 → 600 (realistic target)")
    print(f"  3. cls loss: 0.3 → 0.2 (boost recall)")
    print(f"  4. lrf: 0.01 → 0.05 (slower LR decay)")
    print(f"  5. mixup: 0.15 → 0.2 (stronger augmentation)")
    print(f"  6. close_mosaic: 50 (fine-tune last epochs)")
    print(f"  7. Base model: v3.0 best.pt (transfer learning)")

    print(f"\n⚙️  Configuration:")
    print(f"  Model: YOLO26x (from v3.0 best)")
    print(f"  Base Weights: {MODEL_PATH}")
    print(f"  Dataset: {DATA_YAML}")
    print(f"  GPUs: 8× RTX 4090")
    print(f"  Batch Size: {TRAINING_CONFIG['batch']} (10 per GPU)")
    print(f"  Epochs: {TRAINING_CONFIG['epochs']}")
    print(f"  Patience: {TRAINING_CONFIG['patience']}")
    print(f"  Workers: {TRAINING_CONFIG['workers']}")
    print(f"  Image Size: {TRAINING_CONFIG['imgsz']}")

    print(f"\n📈 Loss Configuration:")
    print(f"  box: {TRAINING_CONFIG['box']}")
    print(f"  cls: {TRAINING_CONFIG['cls']} (↓ from 0.3)")
    print(f"  dfl: {TRAINING_CONFIG['dfl']} (↑ from 1.5)")

    print(f"\n🎨 Augmentation:")
    print(f"  mixup: {TRAINING_CONFIG['mixup']} (↑ from 0.15)")
    print(f"  copy_paste: {TRAINING_CONFIG['copy_paste']} (↑ from 0.1)")
    print(f"  close_mosaic: {TRAINING_CONFIG['close_mosaic']} epochs")

    print(f"\n🎯 Target Performance:")
    print(f"  mAP50: ≥ 90% (gap: 1.69%)")
    print(f"  Recall: ≥ 85% (current: 83.85%)")
    print(f"  Precision: ≥ 88% (current: 86.94%)")
    print(f"\n💡 Expected: 89.5-90.5% mAP50 in 16-20 hours")
    print("="*80)

    # Verify files exist
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Error: Model file not found: {MODEL_PATH}")
        print(f"   Please ensure v3.0 training is complete and archived.")
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
    print(f"\n📦 Loading v3.0 best model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # Start training
    print(f"\n🚀 Starting v4.0 data-driven training...")
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
        print(f"\n💡 Next steps:")
        print(f"  1. Check results.csv for best epoch and performance")
        print(f"  2. Compare with v3.0 (88.31% mAP50)")
        print(f"  3. Run inference on test videos")
        print(f"  4. Archive results to outputs/training_archive/v4.0/")
        print("="*80)

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ Training failed: {str(e)}")
        print(f"\n💡 Troubleshooting:")
        print(f"  - Check GPU memory: nvidia-smi")
        print(f"  - Reduce batch size if OOM: 80 → 64")
        print(f"  - Check dataset paths in {DATA_YAML}")
        print("="*80)
        sys.exit(1)

if __name__ == "__main__":
    main()

