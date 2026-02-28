from ultralytics import YOLO
import os

print("测试模型加载...")
model_path = "/home/user/sea/training_archive/v1.0/models/best.pt"
print(f"模型路径: {model_path}")
print(f"文件存在: {os.path.exists(model_path)}")

try:
    model = YOLO(model_path)
    print("✓ 模型加载成功")
except Exception as e:
    print(f"✗ 模型加载失败: {e}")

video_path = "/home/user/sea/vido/海洋垃圾1.mp4"
print(f"\n视频路径: {video_path}")
print(f"文件存在: {os.path.exists(video_path)}")
