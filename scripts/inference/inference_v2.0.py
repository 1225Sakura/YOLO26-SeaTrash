#!/usr/bin/env python3
"""
海洋垃圾检测视频推理脚本 - 使用v2.0模型
"""

from ultralytics import YOLO
import os
from pathlib import Path

# 配置
MODEL_PATH = "/home/user/sea/training_archive/v2.0/models/best.pt"
VIDEO_DIR = "/home/user/sea/vido"
OUTPUT_DIR = "/home/user/sea/inference_v2.0"

# 视频文件
VIDEO_FILES = [
    "海洋垃圾1.mp4",
    "海洋垃圾2.mp4",
    "海洋垃圾3.mp4"
]

def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载模型
    print(f"加载模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # 推理配置
    inference_config = {
        "conf": 0.25,      # 置信度阈值
        "iou": 0.45,       # NMS IOU阈值
        "imgsz": 640,      # 图像尺寸
        "device": 0,       # 使用GPU 0
        "verbose": True,   # 显示详细信息
        "save": True,      # 保存结果视频
        "save_txt": True,  # 保存检测结果文本
        "save_conf": True, # 保存置信度
        "project": OUTPUT_DIR,
        "exist_ok": True,
    }

    # 对每个视频进行推理
    for video_file in VIDEO_FILES:
        video_path = os.path.join(VIDEO_DIR, video_file)

        if not os.path.exists(video_path):
            print(f"⚠️  视频文件不存在: {video_path}")
            continue

        print(f"\n{'='*70}")
        print(f"开始推理: {video_file}")
        print(f"{'='*70}")

        # 为每个视频创建独立的输出目录
        video_name = Path(video_file).stem
        inference_config["name"] = video_name

        try:
            # 运行推理
            results = model(video_path, **inference_config)
            print(f"✓ 完成推理: {video_file}")
            print(f"  输出目录: {OUTPUT_DIR}/{video_name}")

        except Exception as e:
            print(f"✗ 推理失败 {video_file}: {e}")

    print(f"\n{'='*70}")
    print(f"所有视频推理完成!")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
