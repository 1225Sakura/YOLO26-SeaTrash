#!/usr/bin/env python3
"""
海洋垃圾检测高级训练脚本 - 目标mAP50达到90%
策略：从最佳模型继续训练 + 优化超参数 + 增强数据增强
"""
import sys
sys.path.insert(0, '/home/user/sea/yolo26')

from ultralytics import YOLO

def main():
    # 从最佳模型继续训练
    model = YOLO('/home/user/sea/runs/train/marine_debris_yolo26x2/weights/best.pt')

    # 高级训练配置 - 针对高mAP优化
    results = model.train(
        data='/home/user/sea/configs/marine_debris.yaml',
        epochs=250,                    # 大幅增加训练轮数
        batch=12,                      # 适配单GPU内存
        imgsz=640,
        device=0,

        # 优化器配置
        optimizer='AdamW',
        lr0=0.0005,                    # 降低初始学习率（继续训练）
        lrf=0.01,                      # 最终学习率因子
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # 损失函数权重优化
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # 增强的数据增强策略
        hsv_h=0.02,                    # HSV色调增强
        hsv_s=0.8,                     # HSV饱和度增强
        hsv_v=0.5,                     # HSV明度增强
        degrees=10.0,                  # 旋转角度
        translate=0.2,                 # 平移
        scale=0.9,                     # 缩放范围
        shear=5.0,                     # 剪切
        perspective=0.001,             # 透视变换
        flipud=0.5,                    # 上下翻转概率
        fliplr=0.5,                    # 左右翻转概率
        mosaic=1.0,                    # Mosaic增强
        mixup=0.15,                    # Mixup增强
        copy_paste=0.3,                # Copy-paste增强
        erasing=0.4,                   # 随机擦除
        crop_fraction=1.0,             # 裁剪比例

        # 训练策略
        patience=50,                   # 早停耐心值
        save=True,
        save_period=10,                # 每10个epoch保存
        cache=False,
        workers=8,
        project='runs/train',
        name='marine_debris_yolo26x_v2',
        exist_ok=True,
        pretrained=False,              # 不加载预训练（已经是微调模型）
        verbose=True,
        seed=42,
        deterministic=True,

        # 验证配置
        val=True,
        plots=True,

        # 高级选项
        close_mosaic=15,               # 最后15个epoch关闭mosaic
        amp=True,                      # 自动混合精度
        fraction=1.0,                  # 使用100%数据
        profile=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,

        # NMS配置
        iou=0.7,
        max_det=300,

        # 学习率调度
        cos_lr=True,                   # 余弦学习率
    )

    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"最佳模型: runs/train/marine_debris_yolo26x_v2/weights/best.pt")
    print(f"最终模型: runs/train/marine_debris_yolo26x_v2/weights/last.pt")

    # 在验证集上评估
    print("\n开始最终验证...")
    metrics = model.val(data='/home/user/sea/configs/marine_debris.yaml')
    print(f"\n最终 mAP50: {metrics.box.map50:.4f}")
    print(f"最终 mAP50-95: {metrics.box.map:.4f}")
    print(f"最终 Precision: {metrics.box.mp:.4f}")
    print(f"最终 Recall: {metrics.box.mr:.4f}")

if __name__ == '__main__':
    main()
