#!/usr/bin/env python3
"""
v3.0模型视频推理测试
使用最佳模型 (mAP50=88.31%, 召回率=83.85%)
"""

from ultralytics import YOLO
import os

# 配置
MODEL_PATH = "/home/user/sea/outputs/runs/train/marine_debris_v3.0_optimized/weights/best.pt"
VIDEO_DIR = "/home/user/sea/data/vido"
OUTPUT_DIR = "/home/user/sea/outputs/inference_v3.0"

VIDEOS = ["海洋垃圾1.mp4", "海洋垃圾3.mp4"]

print("="*70)
print("v3.0模型视频推理测试")
print("="*70)
print(f"\n模型性能:")
print(f"  mAP50: 88.31% (epoch 78)")
print(f"  召回率: 83.85%")
print(f"  精确率: 86.94%")
print(f"  mAP50-95: 73.58%")
print("\n对比v1.0 (之前最佳):")
print(f"  mAP50: +12.68%")
print(f"  召回率: +16.66%")
print(f"  精确率: +11.56%")
print("="*70)

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载模型
print("\n加载v3.0最佳模型...")
model = YOLO(MODEL_PATH)
print("✓ 模型加载完成\n")

# 推理配置
inference_config = {
    "conf": 0.25,      # 置信度阈值
    "iou": 0.45,       # NMS IOU阈值
    "imgsz": 640,      # 图像尺寸
    "device": "cpu",   # 使用CPU（GPU内存不足）
    "save": True,      # 保存结果
    "stream": True,    # 流式处理
    "verbose": False,
}

# 处理每个视频
for video_file in VIDEOS:
    video_path = os.path.join(VIDEO_DIR, video_file)
    video_name = video_file.replace(".mp4", "")

    if not os.path.exists(video_path):
        print(f"警告: 视频不存在 {video_path}")
        continue

    print(f"{'='*70}")
    print(f"处理视频: {video_file}")
    print(f"{'='*70}")

    # 推理
    results = model.predict(
        source=video_path,
        project=OUTPUT_DIR,
        name=video_name,
        exist_ok=True,
        **inference_config
    )

    # 统计检测结果
    frame_count = 0
    detection_count = {}

    for r in results:
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"  已处理 {frame_count} 帧", end='\r')

        # 统计检测类别
        if r.boxes is not None and len(r.boxes) > 0:
            for cls in r.boxes.cls:
                cls_name = model.names[int(cls)]
                detection_count[cls_name] = detection_count.get(cls_name, 0) + 1

    print(f"\n✓ 完成 {video_file} - 共 {frame_count} 帧")
    print(f"\n检测统计:")
    for cls_name, count in sorted(detection_count.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls_name}: {count} 次")

print(f"\n{'='*70}")
print("所有视频推理完成!")
print(f"输出目录: {OUTPUT_DIR}")
print("="*70)
print("\n输出文件:")
print(f"  - {OUTPUT_DIR}/海洋垃圾1/海洋垃圾1.avi")
print(f"  - {OUTPUT_DIR}/海洋垃圾3/海洋垃圾3.avi")
print("\n请查看视频评估v3.0模型的检测效果！")
