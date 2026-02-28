from ultralytics import YOLO

model = YOLO("/home/user/sea/training_archive/v2.0/models/best.pt")

# 海洋垃圾1 - 尺度640
print("开始推理: 海洋垃圾1 @ 640px")
model.predict(
    source="/home/user/sea/vido/海洋垃圾1.mp4",
    imgsz=640,
    conf=0.20,
    iou=0.35,
    augment=True,
    device=0,
    save=True,
    project="/home/user/sea/inference_v2.0_enhanced",
    name="海洋垃圾1_640",
    exist_ok=True
)
print("完成!")
