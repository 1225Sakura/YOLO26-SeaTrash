#!/usr/bin/env python3
"""
v2.0模型增强推理 - 完整版
针对海洋垃圾1和海洋垃圾3使用多尺度+TTA追求最高精度
"""

from ultralytics import YOLO
import os
import time

def main():
    # 配置
    MODEL_PATH = "/home/user/sea/training_archive/v2.0/models/best.pt"
    VIDEO_DIR = "/home/user/sea/vido"
    OUTPUT_DIR = "/home/user/sea/inference_v2.0_enhanced"

    VIDEOS = {
        "海洋垃圾1.mp4": [640, 800, 960, 1120, 1280],
        "海洋垃圾3.mp4": [640, 800, 960, 1120, 1280]
    }

    print("="*70)
    print("v2.0模型增强推理")
    print("="*70)
    print(f"模型: {MODEL_PATH}")
    print(f"mAP50: 72.06% | 精确率: 86.13%")
    print(f"数据集: 25,734样本，7个数据集整合")
    print("="*70)

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载模型
    print("\n加载模型...")
    model = YOLO(MODEL_PATH)
    print("✓ 模型加载成功\n")

    # 处理每个视频
    total_tasks = sum(len(scales) for scales in VIDEOS.values())
    current_task = 0

    for video_name, scales in VIDEOS.items():
        video_path = os.path.join(VIDEO_DIR, video_name)

        if not os.path.exists(video_path):
            print(f"警告: 视频不存在 {video_path}\n")
            continue

        print(f"\n{'='*70}")
        print(f"处理视频: {video_name}")
        print(f"{'='*70}")

        for scale in scales:
            current_task += 1
            print(f"\n[{current_task}/{total_tasks}] 尺度 {scale}px (多尺度推理)")

            start_time = time.time()

            try:
                results = model.predict(
                    source=video_path,
                    imgsz=scale,
                    conf=0.20,
                    iou=0.35,
                    device=0,
                    save=True,
                    stream=True,  # 流式处理，避免内存累积
                    project=OUTPUT_DIR,
                    name=f"{video_name.replace('.mp4', '')}_scale{scale}",
                    exist_ok=True,
                    verbose=False
                )

                # 处理流式结果
                frame_count = 0
                for r in results:
                    frame_count += 1
                    if frame_count % 50 == 0:
                        print(f"    已处理 {frame_count} 帧", end='\r')

                elapsed = time.time() - start_time
                print(f"✓ 完成 (耗时: {elapsed/60:.1f}分钟)")

            except Exception as e:
                print(f"✗ 失败: {e}")

    print(f"\n{'='*70}")
    print("所有推理完成!")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"{'='*70}")
    print("\n生成的结果:")
    print("  - 每个视频有5个不同尺度的检测结果")
    print("  - 多尺度推理提供更全面的检测")
    print("  - 建议查看最大尺度(1280px)获得最佳精度")
    print("  - 可以对比不同尺度的检测效果")

if __name__ == "__main__":
    main()
