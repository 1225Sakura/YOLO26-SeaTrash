#!/usr/bin/env python3
"""
Marine Debris Detection Training Script
使用YOLO26进行海洋垃圾目标检测模型微调
"""
import sys
sys.path.insert(0, '/home/user/sea/yolo26')

from ultralytics import YOLO

def main():
    # 加载预训练模型
    model = YOLO('/home/user/sea/model/yolo26x.pt')

    # 训练参数
    results = model.train(
        data='/home/user/sea/configs/marine_debris.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='marine_debris_yolo26x',
        project='/home/user/sea/runs/train',
        patience=20,
        save=True,
        save_period=10,
        device=0,
        workers=8,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        verbose=True,
        seed=0,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        amp=True,
        fraction=1.0,
        profile=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        plots=True
    )

    print("\n训练完成！")
    print(f"最佳模型保存在: {results.save_dir}/weights/best.pt")
    print(f"最后模型保存在: {results.save_dir}/weights/last.pt")

if __name__ == '__main__':
    main()
