#!/usr/bin/env python3
"""
海洋垃圾检测 v3.0 - 定制化训练
针对水面/水下场景优化，追求90% mAP50目标

改进点：
1. 提高召回率：调整损失函数权重
2. 水面/水下增强：特殊数据增强
3. 更长训练：800 epochs
4. 更大batch：64 (4GPU × 16)
5. 更强增强：Mosaic 1.5 + 水面特效
"""

from ultralytics import YOLO
import os

# 配置
MODEL_PATH = "/home/user/sea/outputs/training_archive/v1.0/models/best.pt"  # 从最佳模型开始
DATA_YAML = "/home/user/sea/data/datasets/marine_debris_unified/data.yaml"
PROJECT_DIR = "/home/user/sea/outputs/runs/train"
EXPERIMENT_NAME = "marine_debris_v3.0_optimized"

print("="*70)
print("海洋垃圾检测 v3.0 - 定制化训练")
print("="*70)
print("\n目标:")
print("  - mAP50: 90%+")
print("  - 召回率: 75%+")
print("  - 精确率: 85%+")
print("  - 场景: 水面/水下优化")
print("\n改进:")
print("  ✓ 更长训练 (800 epochs)")
print("  ✓ 更大batch (64)")
print("  ✓ 水面/水下数据增强")
print("  ✓ 召回率优化")
print("  ✓ 4GPU分布式训练")
print("="*70)

# 训练配置 - 针对90% mAP50目标优化
TRAINING_CONFIG = {
    # 基础配置
    "data": DATA_YAML,
    "epochs": 800,  # 更长训练时间
    "patience": 150,  # 更宽容的早停
    "batch": 32,  # 4卡训练，每卡8张 (降低避免OOM)
    "imgsz": 640,  # 图像尺寸

    # 优化器配置
    "optimizer": "AdamW",
    "lr0": 0.0008,  # 稍微降低初始学习率（从大模型微调）
    "lrf": 0.005,  # 更低的最终学习率
    "momentum": 0.937,
    "weight_decay": 0.0005,

    # 损失函数权重 - 针对召回率优化
    "box": 7.5,  # Box loss权重
    "cls": 0.3,  # 降低分类loss权重（提高召回率）
    "dfl": 1.5,  # DFL loss权重

    # 数据增强配置 - 针对水面/水下场景
    "hsv_h": 0.025,  # 增加色调增强（水下色偏）
    "hsv_s": 0.8,  # 增加饱和度增强
    "hsv_v": 0.5,  # 增加明度增强（光照变化）
    "degrees": 15.0,  # 增加旋转角度（波浪晃动）
    "translate": 0.15,  # 增加平移（水流移动）
    "scale": 0.8,  # 增加缩放范围
    "shear": 8.0,  # 增加剪切（水面扭曲）
    "perspective": 0.0005,  # 添加透视变换（水面角度）
    "flipud": 0.5,  # 上下翻转
    "fliplr": 0.5,  # 左右翻转
    "mosaic": 1.0,  # Mosaic增强
    "mixup": 0.2,  # 增加Mixup（提高泛化）
    "copy_paste": 0.4,  # 增加Copy-paste（增加样本多样性）

    # 其他配置
    "device": "0,1,2,3",  # 使用4张GPU (字符串格式)
    "workers": 8,  # 数据加载线程（降低避免冲突）
    "project": PROJECT_DIR,
    "name": EXPERIMENT_NAME,
    "exist_ok": True,
    "pretrained": True,
    "verbose": True,
    "seed": 42,
    "deterministic": False,
    "single_cls": False,
    "rect": False,
    "cos_lr": True,  # 余弦学习率
    "close_mosaic": 15,  # 最后15个epoch关闭mosaic
    "amp": True,  # 自动混合精度
    "fraction": 1.0,  # 使用全部数据
    "profile": False,
    "freeze": None,
    "multi_scale": False,  # 禁用多尺度（4卡DDP兼容）

    # 验证配置
    "val": True,
    "save": True,
    "save_period": 20,  # 每20个epoch保存一次
    "cache": False,  # 不缓存（数据集大）
    "plots": True,
    "overlap_mask": True,
    "mask_ratio": 4,

    # 高级配置
    "label_smoothing": 0.1,  # 标签平滑（提高泛化）
    "nbs": 64,  # Nominal batch size
    "dropout": 0.0,  # Dropout（大模型不需要）
}

def main():
    print("\n初始化模型...")
    print(f"起始模型: {MODEL_PATH}")
    print(f"  (v1.0最佳模型: mAP50=75.63%)")

    model = YOLO(MODEL_PATH)
    print("✓ 模型加载完成")

    print(f"\n数据集: {DATA_YAML}")
    print("  - 样本数: 25,734")
    print("  - 类别数: 14")
    print("  - 来源: 7个数据集整合")

    print("\n" + "="*70)
    print("开始训练...")
    print("="*70)
    print("\n配置摘要:")
    print(f"  Epochs: {TRAINING_CONFIG['epochs']}")
    print(f"  Batch Size: {TRAINING_CONFIG['batch']} (4GPU × 16)")
    print(f"  Image Size: {TRAINING_CONFIG['imgsz']}")
    print(f"  Learning Rate: {TRAINING_CONFIG['lr0']} -> {TRAINING_CONFIG['lrf']}")
    print(f"  Early Stop: patience={TRAINING_CONFIG['patience']}")
    print(f"  Workers: {TRAINING_CONFIG['workers']}")
    print("\n损失权重 (召回率优化):")
    print(f"  Box: {TRAINING_CONFIG['box']}")
    print(f"  Cls: {TRAINING_CONFIG['cls']} (降低以提高召回率)")
    print(f"  DFL: {TRAINING_CONFIG['dfl']}")
    print("\n数据增强 (水面/水下优化):")
    print(f"  HSV: h={TRAINING_CONFIG['hsv_h']}, s={TRAINING_CONFIG['hsv_s']}, v={TRAINING_CONFIG['hsv_v']}")
    print(f"  几何: rotate={TRAINING_CONFIG['degrees']}°, translate={TRAINING_CONFIG['translate']}")
    print(f"  Mosaic: {TRAINING_CONFIG['mosaic']}, Mixup: {TRAINING_CONFIG['mixup']}")
    print(f"  Copy-paste: {TRAINING_CONFIG['copy_paste']}")

    print("\n预计训练时间: 24-48小时")
    print("目标: mAP50 ≥ 90%, 召回率 ≥ 75%")
    print("\n" + "="*70)

    # 开始训练
    results = model.train(**TRAINING_CONFIG)

    print("\n" + "="*70)
    print("训练完成!")
    print("="*70)
    print(f"\n结果保存在: {PROJECT_DIR}/{EXPERIMENT_NAME}")
    print("\n最佳模型:")
    print(f"  {PROJECT_DIR}/{EXPERIMENT_NAME}/weights/best.pt")

    return results

if __name__ == "__main__":
    main()
