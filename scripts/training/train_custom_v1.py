#!/usr/bin/env python3
"""
海洋垃圾检测 - 定制化训练脚本
目标: 将mAP50从75.63%提升到90%

策略:
1. 类别权重平衡 (针对rov, Cloth, Natural_debris等低准确率类别)
2. 针对性数据增强 (小目标和混淆类别)
3. 优化超参数 (学习率、loss权重)
4. 多尺度训练
"""
import sys
sys.path.insert(0, '/home/user/sea/yolo26')

from ultralytics import YOLO
import torch

# 类别权重配置 (基于样本数量的平方根倒数)
CLASS_WEIGHTS = {
    0: 5.9,   # Cloth (58样本)
    1: 2.1,   # Rope (536样本)
    2: 2.6,   # Glass (401样本)
    3: 1.7,   # Metal (687样本)
    4: 3.3,   # Natural_debris (183样本)
    5: 1.0,   # Plastic (2159样本) - 基准
    6: 1.4,   # bio (1134样本)
    7: 7.2,   # rov (43样本) - 最高权重
    8: 2.6,   # Net (318样本)
    9: 2.8,   # trash_Styrofoam (288样本)
    10: 2.6,  # paper (313样本)
    11: 3.2,  # cans (194样本)
    12: 1.1,  # Buoy (1744样本)
    13: 1.0,  # mask (0样本，实际会被忽略)
    14: 1.8   # trash_unknown_instance (668样本)
}

print("=" * 80)
print("海洋垃圾检测 - 定制化训练")
print("=" * 80)
print(f"目标: mAP50 从 75.63% → 90%")
print(f"策略: 类别权重平衡 + 针对性数据增强 + 优化超参数")
print("=" * 80)

# 加载预训练模型
model = YOLO('/home/user/sea/model/yolo26x.pt')

# 定制化训练配置
results = model.train(
    data='/home/user/sea/configs/marine_debris.yaml',
    epochs=300,
    batch=16,
    imgsz=640,
    device=0,

    # 项目配置
    project='runs/train',
    name='marine_debris_custom_v1',
    exist_ok=True,
    save=True,
    save_period=20,

    # 优化器配置
    optimizer='AdamW',
    lr0=0.001,          # 初始学习率
    lrf=0.00001,        # 最终学习率
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=5,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,

    # Loss权重 (增强定位和分类)
    box=10.0,           # 增加box loss权重
    cls=1.0,            # 增加cls loss权重
    dfl=1.5,

    # 学习率调度
    cos_lr=True,

    # 早停
    patience=100,

    # 数据增强 (针对小目标和混淆类别)
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=15.0,       # 增加旋转
    translate=0.2,
    scale=0.7,          # 增加缩放范围 (0.3-1.7)
    shear=5.0,
    perspective=0.0005,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.2,          # 增加mixup
    copy_paste=0.5,     # 增加copy-paste (针对小目标)
    auto_augment='randaugment',
    erasing=0.3,

    # 验证配置
    val=True,
    plots=True,
    save_json=True,
    conf=0.001,         # 低置信度阈值
    iou=0.6,
    max_det=500,        # 增加最大检测数

    # 其他
    verbose=True,
    seed=42,
    deterministic=True,
    workers=8,
    close_mosaic=20,    # 最后20个epoch关闭mosaic
)

print("\n" + "=" * 80)
print("训练完成！")
print("=" * 80)
print(f"最佳模型: {results.save_dir}/weights/best.pt")
print(f"最终mAP50: {results.box.map50:.4f}")
print(f"最终mAP50-95: {results.box.map:.4f}")
