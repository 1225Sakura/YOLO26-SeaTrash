#!/usr/bin/env python3
"""
海洋垃圾检测从头训练脚本 - 备选方案
使用更大的模型和最优超参数从头开始训练
"""
import sys
sys.path.insert(0, '/home/user/sea/yolo26')

from ultralytics import YOLO

def main():
    # 使用预训练的YOLO26x模型
    model = YOLO('/home/user/sea/model/yolo26x.pt')

    # 最优训练配置
    results = model.train(
        data='/home/user/sea/configs/marine_debris.yaml',
        epochs=300,
        batch=20,                      # 平衡batch size
        imgsz=640,
        device=0,

        # 优化器配置
        optimizer='AdamW',
        lr0=0.001,                     # 标准初始学习率
        lrf=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=10,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # 损失函数权重
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # 强数据增强
        hsv_h=0.03,
        hsv_s=0.9,
        hsv_v=0.6,
        degrees=15.0,
        translate=0.2,
        scale=0.9,
        shear=5.0,
        perspective=0.001,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.5,
        erasing=0.4,

        # 训练策略
        patience=80,
        save=True,
        save_period=10,
        cache=False,
        workers=8,
        project='runs/train',
        name='marine_debris_yolo26x_scratch',
        exist_ok=True,
        verbose=True,
        seed=42,
        deterministic=True,

        val=True,
        plots=True,
        close_mosaic=20,
        amp=True,
        fraction=1.0,
        iou=0.7,
        max_det=300,
        cos_lr=True,
    )

    print("\n训练完成！")
    print(f"最佳模型: runs/train/marine_debris_yolo26x_scratch/weights/best.pt")

    # 验证
    metrics = model.val(data='/home/user/sea/configs/marine_debris.yaml')
    print(f"\nmAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

if __name__ == '__main__':
    main()
