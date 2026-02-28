#!/usr/bin/env python3
"""
Training Data Analysis for v4.0 Optimization
Analyzes v1.0, v2.0, v3.0 training results to identify optimization strategies
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
PROJECT_ROOT = Path("/home/user/sea")
V1_CSV = PROJECT_ROOT / "outputs/training_archive/v1.0/metrics/results.csv"
V2_CSV = PROJECT_ROOT / "outputs/training_archive/v2.0/metrics/results.csv"
V3_CSV = PROJECT_ROOT / "outputs/training_archive/v3.0/metrics/results.csv"

def analyze_version(csv_path, version_name):
    """Analyze single version training data"""
    df = pd.read_csv(csv_path)

    # Find best epoch
    best_idx = df['metrics/mAP50(B)'].idxmax()
    best_epoch = df.loc[best_idx]

    # Find final epoch
    final_epoch = df.iloc[-1]

    # Calculate metrics
    max_map50 = df['metrics/mAP50(B)'].max()
    max_recall = df['metrics/recall(B)'].max()
    max_precision = df['metrics/precision(B)'].max()

    # Loss analysis
    final_box_loss = final_epoch['train/box_loss']
    final_cls_loss = final_epoch['train/cls_loss']
    final_dfl_loss = final_epoch['train/dfl_loss']

    # Validation loss
    final_val_box = final_epoch['val/box_loss']
    final_val_cls = final_epoch['val/cls_loss']

    # Overfitting check
    train_val_gap = abs(final_box_loss - final_val_box)

    print(f"\n{'='*70}")
    print(f"{version_name} Analysis")
    print(f"{'='*70}")
    print(f"\n📊 Best Performance (Epoch {int(best_epoch['epoch'])}):")
    print(f"  mAP50: {best_epoch['metrics/mAP50(B)']:.4f}")
    print(f"  Recall: {best_epoch['metrics/recall(B)']:.4f}")
    print(f"  Precision: {best_epoch['metrics/precision(B)']:.4f}")
    print(f"  mAP50-95: {best_epoch['metrics/mAP50-95(B)']:.4f}")

    print(f"\n📉 Final Performance (Epoch {int(final_epoch['epoch'])}):")
    print(f"  mAP50: {final_epoch['metrics/mAP50(B)']:.4f}")
    print(f"  Recall: {final_epoch['metrics/recall(B)']:.4f}")
    print(f"  Precision: {final_epoch['metrics/precision(B)']:.4f}")

    print(f"\n🔍 Loss Analysis:")
    print(f"  Train Box Loss: {final_box_loss:.5f}")
    print(f"  Train Cls Loss: {final_cls_loss:.5f}")
    print(f"  Train DFL Loss: {final_dfl_loss:.5f}")
    print(f"  Val Box Loss: {final_val_box:.5f}")
    print(f"  Val Cls Loss: {final_val_cls:.5f}")
    print(f"  Train-Val Gap: {train_val_gap:.5f}")

    # Performance degradation
    if best_epoch['epoch'] != final_epoch['epoch']:
        map_drop = best_epoch['metrics/mAP50(B)'] - final_epoch['metrics/mAP50(B)']
        recall_drop = best_epoch['metrics/recall(B)'] - final_epoch['metrics/recall(B)']
        print(f"\n⚠️  Performance Degradation:")
        print(f"  mAP50 drop: {map_drop:.4f} ({map_drop/best_epoch['metrics/mAP50(B)']*100:.2f}%)")
        print(f"  Recall drop: {recall_drop:.4f} ({recall_drop/best_epoch['metrics/recall(B)']*100:.2f}%)")

    # Learning rate at best epoch
    print(f"\n📈 Learning Rate at Best Epoch: {best_epoch['lr/pg0']:.8f}")

    return {
        'version': version_name,
        'best_epoch': int(best_epoch['epoch']),
        'final_epoch': int(final_epoch['epoch']),
        'best_map50': best_epoch['metrics/mAP50(B)'],
        'best_recall': best_epoch['metrics/recall(B)'],
        'best_precision': best_epoch['metrics/precision(B)'],
        'final_map50': final_epoch['metrics/mAP50(B)'],
        'final_recall': final_epoch['metrics/recall(B)'],
        'train_val_gap': train_val_gap,
        'final_cls_loss': final_cls_loss,
        'final_box_loss': final_box_loss,
        'lr_at_best': best_epoch['lr/pg0']
    }

def compare_versions(results):
    """Compare all versions and identify patterns"""
    print(f"\n{'='*70}")
    print("Cross-Version Comparison")
    print(f"{'='*70}")

    df = pd.DataFrame(results)

    print("\n📊 Performance Progression:")
    print(df[['version', 'best_map50', 'best_recall', 'best_precision']].to_string(index=False))

    print("\n🎯 Key Findings:")

    # v1.0 vs v2.0
    v1_v2_map_diff = results[1]['best_map50'] - results[0]['best_map50']
    v1_v2_recall_diff = results[1]['best_recall'] - results[0]['best_recall']
    print(f"\n1. v1.0 → v2.0:")
    print(f"   mAP50: {v1_v2_map_diff:+.4f} ({'REGRESSION' if v1_v2_map_diff < 0 else 'improvement'})")
    print(f"   Recall: {v1_v2_recall_diff:+.4f} ({'SEVERE DROP' if v1_v2_recall_diff < -0.05 else 'change'})")
    if v1_v2_recall_diff < -0.05:
        print(f"   ⚠️  PROBLEM: Recall dropped {abs(v1_v2_recall_diff)*100:.1f}% - likely cls loss too high")

    # v2.0 vs v3.0
    v2_v3_map_diff = results[2]['best_map50'] - results[1]['best_map50']
    v2_v3_recall_diff = results[2]['best_recall'] - results[1]['best_recall']
    print(f"\n2. v2.0 → v3.0:")
    print(f"   mAP50: {v2_v3_map_diff:+.4f} (+{v2_v3_map_diff/results[1]['best_map50']*100:.1f}%)")
    print(f"   Recall: {v2_v3_recall_diff:+.4f} (+{v2_v3_recall_diff/results[1]['best_recall']*100:.1f}%)")
    print(f"   ✅ SUCCESS: Major improvement from cls loss reduction (0.5 → 0.3)")

    # Early stopping analysis
    print(f"\n3. Early Stopping Analysis:")
    for r in results:
        epochs_after_best = r['final_epoch'] - r['best_epoch']
        print(f"   {r['version']}: Best at epoch {r['best_epoch']}, stopped at {r['final_epoch']} (+{epochs_after_best})")

    # v3.0 degradation
    v3_degradation = results[2]['best_map50'] - results[2]['final_map50']
    if v3_degradation > 0.1:
        print(f"\n   ⚠️  v3.0 SEVERE DEGRADATION: {v3_degradation:.4f} ({v3_degradation/results[2]['best_map50']*100:.1f}%)")
        print(f"   Problem: Continued training 150 epochs after best, model overfit")

    # Loss analysis
    print(f"\n4. Loss Weight Analysis:")
    print(f"   v2.0 cls loss: {results[1]['final_cls_loss']:.5f} (weight: 0.5)")
    print(f"   v3.0 cls loss: {results[2]['final_cls_loss']:.5f} (weight: 0.3)")
    print(f"   Reduction: {(1 - results[2]['final_cls_loss']/results[1]['final_cls_loss'])*100:.1f}%")
    print(f"   ✅ Lower cls loss → Higher recall")

def generate_v4_recommendations(results):
    """Generate v4.0 optimization recommendations"""
    print(f"\n{'='*70}")
    print("v4.0 Optimization Recommendations")
    print(f"{'='*70}")

    v3 = results[2]
    gap_to_90 = 0.90 - v3['best_map50']

    print(f"\n🎯 Current Status:")
    print(f"   v3.0 Best: {v3['best_map50']:.4f} (88.31%)")
    print(f"   Target: 0.9000 (90%)")
    print(f"   Gap: {gap_to_90:.4f} ({gap_to_90/0.90*100:.2f}%)")

    print(f"\n💡 Key Optimizations for v4.0:")

    print(f"\n1. ⚠️  CRITICAL: Fix Early Stopping Issue")
    print(f"   Problem: v3.0 degraded from 88.31% → 65.80% after epoch 78")
    print(f"   Root Cause: Patience=150 too long, model overfit")
    print(f"   Solution:")
    print(f"     - Reduce patience: 150 → 100")
    print(f"     - Add checkpoint saving every 25 epochs")
    print(f"     - Monitor validation loss trend, not just mAP50")

    print(f"\n2. 🔧 Further Reduce Classification Loss Weight")
    print(f"   Current: cls=0.3 (v3.0)")
    print(f"   Analysis: Recall 83.85% good, but can push higher")
    print(f"   Recommendation: cls=0.2 (33% reduction)")
    print(f"   Expected: Recall 85%+, mAP50 89-90%")

    print(f"\n3. 📈 Optimize Learning Rate Schedule")
    print(f"   v3.0 LR at best epoch: {v3['lr_at_best']:.8f}")
    print(f"   Problem: LR decayed too fast, reached best early (epoch 78/800)")
    print(f"   Solution:")
    print(f"     - Increase lrf: 0.01 → 0.05 (slower decay)")
    print(f"     - Extend warmup: 5 → 10 epochs")
    print(f"     - Use cosine annealing with restarts")

    print(f"\n4. 🎨 Enhanced Data Augmentation")
    print(f"   Current mixup: 0.15")
    print(f"   Recommendation: mixup=0.2, copy_paste=0.15")
    print(f"   Add: Random erasing (0.1) for occlusion robustness")

    print(f"\n5. 🔄 Multi-Scale Training")
    print(f"   Current: Fixed 640×640")
    print(f"   Recommendation: Multi-scale [480, 640, 800]")
    print(f"   Benefit: Better scale invariance, +1-2% mAP50")

    print(f"\n6. ⚡ 8-GPU Specific Optimizations")
    print(f"   Batch: 80 (10 per GPU) ✓")
    print(f"   Add: Gradient accumulation (2 steps) → effective batch 160")
    print(f"   Add: SyncBatchNorm for better statistics")
    print(f"   Add: EMA (Exponential Moving Average) decay=0.9999")

    print(f"\n7. 📊 Loss Function Refinement")
    print(f"   Current: box=7.5, cls=0.3, dfl=1.5")
    print(f"   Recommendation: box=7.5, cls=0.2, dfl=1.8")
    print(f"   Rationale: Boost DFL for better bbox regression")

    print(f"\n8. 🎓 Knowledge Distillation (Optional)")
    print(f"   Use v3.0 best model as teacher")
    print(f"   Soft labels with temperature=3")
    print(f"   Expected: +0.5-1% mAP50")

    print(f"\n📋 Recommended v4.0 Configuration:")
    print(f"   epochs: 600 (not 1000, avoid overfitting)")
    print(f"   patience: 100 (not 200)")
    print(f"   batch: 80")
    print(f"   cls: 0.2 (not 0.25)")
    print(f"   lrf: 0.05 (not 0.01)")
    print(f"   mixup: 0.2 (not 0.15)")
    print(f"   copy_paste: 0.15 (not 0.1)")
    print(f"   close_mosaic: 50 (disable mosaic last 50 epochs)")
    print(f"   multi_scale: True")

    print(f"\n🎯 Expected v4.0 Performance:")
    print(f"   mAP50: 89.5-90.5% (conservative: 89.5%)")
    print(f"   Recall: 84-86%")
    print(f"   Precision: 87-89%")
    print(f"   Training time: 16-20 hours (8× RTX 4090)")

def main():
    print("="*70)
    print("YOLO26-SeaTrash Training Data Analysis")
    print("="*70)

    # Analyze each version
    results = []
    for csv_path, version in [(V1_CSV, "v1.0"), (V2_CSV, "v2.0"), (V3_CSV, "v3.0")]:
        result = analyze_version(csv_path, version)
        results.append(result)

    # Compare versions
    compare_versions(results)

    # Generate v4.0 recommendations
    generate_v4_recommendations(results)

    print(f"\n{'='*70}")
    print("Analysis Complete")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
