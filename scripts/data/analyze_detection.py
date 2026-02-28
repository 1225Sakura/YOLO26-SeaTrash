#!/usr/bin/env python3
"""分析视频检测结果"""
import re
from collections import defaultdict

# 读取检测日志
with open('/tmp/claude/-home-user-sea/tasks/be4ebf2.output', 'r') as f:
    content = f.read()

# 为每个视频统计检测结果
videos = ['海洋垃圾1.mp4', '海洋垃圾2.mp4', '海洋垃圾3.mp4']

for video in videos:
    print(f"\n{'='*60}")
    print(f"视频: {video}")
    print(f"{'='*60}")

    # 提取该视频的所有帧
    pattern = rf"video 1/1 \(frame (\d+)/(\d+)\) .*{re.escape(video)}: \d+x\d+ (.+)"
    matches = re.findall(pattern, content)

    if not matches:
        print("未找到检测记录")
        continue

    total_frames = int(matches[-1][1])
    print(f"总帧数: {total_frames}")

    # 统计检测对象
    detections = defaultdict(int)
    frames_with_detections = 0
    frames_no_detections = 0

    for frame_num, total, detection_str in matches:
        if 'no detections' in detection_str:
            frames_no_detections += 1
        else:
            frames_with_detections += 1
            # 解析检测对象 (例如: "1 Rope, 2 Buoys, 18.1ms")
            parts = detection_str.split(',')
            for part in parts[:-1]:  # 最后一部分是时间，跳过
                part = part.strip()
                match = re.match(r'(\d+)\s+(.+?)s?$', part)
                if match:
                    count = int(match.group(1))
                    obj_name = match.group(2).rstrip('s')  # 移除可能的复数s
                    detections[obj_name] += count

    print(f"\n帧检测统计:")
    print(f"  有检测的帧数: {frames_with_detections} ({frames_with_detections/total_frames*100:.1f}%)")
    print(f"  无检测的帧数: {frames_no_detections} ({frames_no_detections/total_frames*100:.1f}%)")

    if detections:
        print(f"\n检测对象统计 (总数):")
        for obj, count in sorted(detections.items(), key=lambda x: x[1], reverse=True):
            avg_per_frame = count / frames_with_detections if frames_with_detections > 0 else 0
            print(f"  {obj}: {count} 次检测 (平均每帧 {avg_per_frame:.2f})")
    else:
        print("\n未检测到任何对象")

print(f"\n{'='*60}")
print("检测结果视频已保存至:")
print("runs/detect/runs/detect/video_detection/")
print(f"{'='*60}")
