#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/user/sea/yolo26')
from ultralytics import YOLO

model = YOLO('/home/user/sea/runs/train/marine_debris_yolo26x2/weights/best.pt')

videos = [
    '/home/user/sea/vido/海洋垃圾1.mp4',
    '/home/user/sea/vido/海洋垃圾2.mp4',
    '/home/user/sea/vido/海洋垃圾3.mp4'
]

for video in videos:
    print(f"\n处理: {video}")
    results = model.predict(
        source=video,
        save=True,
        conf=0.25,
        iou=0.6,
        max_det=300,
        project='runs/detect',
        name='video_detection',
        exist_ok=True
    )
    print(f"完成: {video}")

print("\n所有视频检测完成！")
print("结果保存在: runs/detect/video_detection/")
