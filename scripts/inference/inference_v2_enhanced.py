#!/usr/bin/env python3
"""
海洋垃圾检测 - v2.0模型增强推理
使用多尺度 + TTA 追求最高精度
"""

from ultralytics import YOLO
import os

# 配置
MODEL_PATH = "/home/user/sea/training_archive/v2.0/models/best.pt"
VIDEO_DIR = "/home/user/sea/vido"
OUTPUT_DIR = "/home/user/sea/inference_v2.0_enhanced"

VIDEOS = ["海洋垃圾1.mp4", "海洋垃圾3.mp4"]
SCALES = [640, 800, 960, 1120, 1280]  # 5个尺度

print("="*70)
print("海洋垃圾检测 - v2.0模型增强推理")
print("="*70)
print(f"\n模型: v2.0 (mAP50=72.06%, 精确率=86.13%)")
print(f"数据集: 25,734样本，7个数据集整合")
print(f"多尺度: {SCALES}")
print(f"TTA: 启用")
print(f"追求最高精度，不在乎时间\n")

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载模型
print("加载模型...")
model = YOLO(MODEL_PATH)
print("✓ 模型加载完成\n")

# 处理每个视频的每个尺度
for video in VIDEOS:
    video_path = os.path.join(VIDEO_DIR, video)
    video_name = video.replace(".mp4", "")

    if not os.path.exists(video_path):
        print(f"警告: 视频不存在 {video_path}")
        continue

    print("="*70)
    print(f"处理视频: {video}")
    print("="*70)

    for i, scale in enumerate(SCALES, 1):
        print(f"\n[{i}/{len(SCALES)}] 尺度 {scale}px + TTA")

        output_name = f"{video_name}_scale{scale}"

        # 使用YOLO predict进行推理
        # augment=True 启用TTA（测试时增强）
        results = model.predict(
            source=video_path,
            imgsz=scale,
            conf=0.20,          # 置信度阈值
            iou=0.35,           # NMS IOU阈值
            augment=True,       # 启用TTA
            device=0,           # GPU 0
            save=True,          # 保存结果视频
            project=OUTPUT_DIR,
            name=output_name,
            exist_ok=True,
            verbose=True,
            stream=True,        # 流式处理，节省内存
        )

        # 处理结果（流式）
        frame_count = 0
        for r in results:
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"  已处理 {frame_count} 帧")

        print(f"  ✓ 完成 {scale}px 推理，共 {frame_count} 帧")

print("\n" + "="*70)
print("所有推理完成!")
print(f"输出目录: {OUTPUT_DIR}")
print("="*70)
print("\n说明:")
print("  - 生成了 {} 个不同尺度的检测结果".format(len(SCALES) * len(VIDEOS)))
print("  - 每个尺度都使用了TTA增强")
print("  - 可以对比不同尺度的检测效果")
print("  - 建议查看最大尺度(1280px)的结果以获得最佳精度")
