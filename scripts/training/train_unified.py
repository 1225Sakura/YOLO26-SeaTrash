#!/usr/bin/env python3
"""
训练脚本 - 使用整合后的海洋垃圾数据集
目标: 达到90% mAP50

数据集信息:
- 总样本数: 25,734
- 训练集: 20,587 (80%)
- 验证集: 2,573 (10%)
- 测试集: 2,574 (10%)
- 类别数: 14
"""

from ultralytics import YOLO
import torch

# 检查GPU
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 数据集配置
DATA_YAML = "/home/user/sea/datasets/marine_debris_unified/data.yaml"

# 模型配置
MODEL_PATH = "/home/user/sea/runs/train/marine_debris_yolo26x2/weights/best.pt"  # 从最佳模型继续训练 (75.63% mAP50)

# 训练配置 - 针对90% mAP50目标优化
TRAINING_CONFIG = {
    # 基础配置
    "data": DATA_YAML,
    "epochs": 500,  # 长时间训练以达到最佳效果
    "patience": 100,  # 早停耐心值
    "batch": 32,  # 4卡训练，每卡8张 (总batch size 32)
    "imgsz": 640,  # 图像尺寸

    # 优化器配置
    "optimizer": "AdamW",  # AdamW通常比SGD更稳定
    "lr0": 0.001,  # 初始学习率 (从预训练模型微调，使用较小学习率)
    "lrf": 0.01,  # 最终学习率 (lr0 * lrf)
    "momentum": 0.937,
    "weight_decay": 0.0005,

    # 数据增强配置 - 增强以提高泛化能力
    "hsv_h": 0.015,  # 色调增强
    "hsv_s": 0.7,  # 饱和度增强
    "hsv_v": 0.4,  # 明度增强
    "degrees": 10.0,  # 旋转角度
    "translate": 0.1,  # 平移
    "scale": 0.9,  # 缩放
    "shear": 5.0,  # 剪切
    "perspective": 0.0,  # 透视变换
    "flipud": 0.5,  # 上下翻转概率
    "fliplr": 0.5,  # 左右翻转概率
    "mosaic": 1.0,  # Mosaic增强
    "mixup": 0.15,  # Mixup增强
    "copy_paste": 0.3,  # Copy-paste增强

    # 损失函数权重
    "box": 7.5,  # Box loss权重
    "cls": 0.5,  # 分类loss权重
    "dfl": 1.5,  # DFL loss权重

    # 其他配置
    "device": [0, 1, 2, 3],  # 使用4张GPU
    "workers": 8,  # 数据加载线程数
    "project": "/home/user/sea/runs/detect",
    "name": "train_unified",
    "exist_ok": True,
    "pretrained": True,
    "verbose": True,
    "seed": 42,
    "deterministic": False,
    "single_cls": False,
    "rect": False,  # 矩形训练
    "cos_lr": True,  # 余弦学习率调度
    "close_mosaic": 10,  # 最后10个epoch关闭mosaic
    "amp": True,  # 自动混合精度训练
    "fraction": 1.0,  # 使用全部数据
    "profile": False,
    "freeze": None,  # 不冻结任何层
    "multi_scale": False,  # 禁用多尺度训练 (4卡训练时可能导致问题)

    # 验证配置
    "val": True,
    "save": True,
    "save_period": 10,  # 每10个epoch保存一次
    "cache": False,  # 不缓存图像到内存 (数据集较大)
    "plots": True,
    "overlap_mask": True,
    "mask_ratio": 4,
}

print("="*80)
print("训练配置")
print("="*80)
print(f"数据集: {DATA_YAML}")
print(f"起始模型: {MODEL_PATH}")
print(f"目标: 达到90% mAP50 (当前最佳: 75.63%)")
print(f"训练轮数: {TRAINING_CONFIG['epochs']}")
print(f"批次大小: {TRAINING_CONFIG['batch']} (4卡，每卡8张)")
print(f"图像尺寸: {TRAINING_CONFIG['imgsz']}")
print(f"优化器: {TRAINING_CONFIG['optimizer']}")
print(f"初始学习率: {TRAINING_CONFIG['lr0']}")
print(f"GPU设备: {TRAINING_CONFIG['device']}")
print("="*80)

# 加载模型
print("\n加载模型...")
model = YOLO(MODEL_PATH)

# 开始训练
print("\n开始训练...")
print("提示: 训练将持续到达到最佳效果或早停条件")
print("="*80)

results = model.train(**TRAINING_CONFIG)

print("\n" + "="*80)
print("训练完成！")
print("="*80)
print(f"最佳模型保存在: {model.trainer.best}")
print(f"最后模型保存在: {model.trainer.last}")
