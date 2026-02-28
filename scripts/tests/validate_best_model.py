#!/usr/bin/env python3
"""
验证原始最佳模型并生成详细分析数据
"""
import sys
sys.path.insert(0, '/home/user/sea/yolo26')

from ultralytics import YOLO

# 加载最佳模型
model = YOLO('/home/user/sea/runs/train/marine_debris_yolo26x2/weights/best.pt')

# 验证模型，生成详细分析
print("=" * 70)
print("验证原始最佳模型 (mAP50: 75.63%)")
print("=" * 70)

results = model.val(
    data='/home/user/sea/configs/marine_debris.yaml',
    split='val',
    save_json=True,
    save_hybrid=True,
    conf=0.001,  # 低置信度阈值以捕获所有预测
    iou=0.6,
    max_det=300,
    plots=True,  # 生成所有分析图表
    verbose=True
)

print("\n" + "=" * 70)
print("验证完成！分析文件已生成")
print("=" * 70)
print(f"结果目录: {results.save_dir}")
print(f"mAP50: {results.box.map50:.4f}")
print(f"mAP50-95: {results.box.map:.4f}")
print(f"Precision: {results.box.mp:.4f}")
print(f"Recall: {results.box.mr:.4f}")
