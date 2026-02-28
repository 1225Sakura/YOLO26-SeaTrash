#!/usr/bin/env python3
"""
海洋垃圾检测 - 增强推理脚本 (简化版)
使用YOLO内置的增强功能 + 模型集成
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import os

# 配置
MODELS = [
    "/home/user/sea/training_archive/v1.0/models/best.pt",  # mAP50 75.63%
    "/home/user/sea/training_archive/v2.0/models/best.pt",  # mAP50 72.06%
]

VIDEO_DIR = "/home/user/sea/vido"
OUTPUT_DIR = "/home/user/sea/inference_enhanced"
VIDEO_FILES = ["海洋垃圾1.mp4", "海洋垃圾3.mp4"]

# 多尺度配置
SCALES = [640, 800, 960, 1120]

# 推理配置
CONF_THRESHOLD = 0.20
IOU_THRESHOLD = 0.35


def ensemble_predict(models, frame, scales, conf, iou):
    """
    模型集成 + 多尺度推理
    """
    all_boxes = []
    all_scores = []
    all_classes = []

    for model in models:
        for scale in scales:
            # 多尺度推理
            results = model.predict(
                frame,
                imgsz=scale,
                conf=conf,
                iou=iou,
                augment=True,  # 启用TTA
                device=0,
                verbose=False
            )

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                if len(boxes) > 0:
                    all_boxes.append(boxes.xyxy.cpu().numpy())
                    all_scores.append(boxes.conf.cpu().numpy())
                    all_classes.append(boxes.cls.cpu().numpy())

    if len(all_boxes) == 0:
        return None, None, None

    # 合并所有检测
    all_boxes = np.vstack(all_boxes)
    all_scores = np.concatenate(all_scores)
    all_classes = np.concatenate(all_classes)

    # 加权NMS
    keep_indices = weighted_nms(all_boxes, all_scores, all_classes, iou_threshold=0.4)

    return all_boxes[keep_indices], all_scores[keep_indices], all_classes[keep_indices]


def weighted_nms(boxes, scores, classes, iou_threshold=0.4):
    """加权NMS"""
    if len(boxes) == 0:
        return np.array([], dtype=int)

    keep = []
    indices = np.argsort(scores)[::-1]

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)

        if len(indices) == 1:
            break

        # 计算IOU
        current_box = boxes[current]
        other_boxes = boxes[indices[1:]]

        ious = compute_iou_batch(current_box, other_boxes)

        # 保留IOU小于阈值的
        mask = ious < iou_threshold
        indices = indices[1:][mask]

    return np.array(keep)


def compute_iou_batch(box, boxes):
    """批量计算IOU"""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    iou = inter_area / (box_area + boxes_area - inter_area + 1e-6)
    return iou


def process_video(video_path, models, output_path):
    """处理视频"""
    print(f"\n{'='*70}")
    print(f"处理视频: {Path(video_path).name}")
    print(f"{'='*70}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频信息: {width}x{height} @ {fps}fps, 总帧数: {total_frames}")
    print(f"配置: {len(models)}模型 × {len(SCALES)}尺度 × TTA = {len(models)*len(SCALES)*2}次推理/帧")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    class_names = [
        'Plastic', 'Plastic_Bottle', 'Plastic_Buoy', 'Metal', 'Glass',
        'Paper', 'Cloth', 'Net', 'Rope', 'Styrofoam', 'Buoy', 'Rubber',
        'Natural_debris', 'Trash_Other'
    ]

    frame_idx = 0
    detection_count = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 增强推理
        boxes, scores, classes = ensemble_predict(
            models, frame, SCALES, CONF_THRESHOLD, IOU_THRESHOLD
        )

        # 绘制结果
        if boxes is not None:
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i].astype(int)
                conf = scores[i]
                cls = int(classes[i])
                cls_name = class_names[cls]

                detection_count[cls_name] = detection_count.get(cls_name, 0) + 1

                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"{cls_name}: {conf:.2f}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1-lh-10), (x1+lw, y1), color, -1)
                cv2.putText(frame, label, (x1, y1-5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        out.write(frame)
        frame_idx += 1

        if frame_idx % 10 == 0:
            print(f"  处理进度: {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")

    cap.release()
    out.release()

    print(f"\n检测统计:")
    for cls_name, count in sorted(detection_count.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls_name}: {count} 次")
    print(f"\n输出: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*70)
    print("海洋垃圾检测 - 增强推理模式")
    print("="*70)
    print(f"\n配置:")
    print(f"  模型: {len(MODELS)} 个")
    print(f"  多尺度: {SCALES}")
    print(f"  TTA: 启用")
    print(f"  置信度: {CONF_THRESHOLD}")
    print(f"  IOU: {IOU_THRESHOLD}")

    print(f"\n加载模型...")
    models = [YOLO(path) for path in MODELS]
    print(f"  ✓ 加载完成")

    for video_file in VIDEO_FILES:
        video_path = os.path.join(VIDEO_DIR, video_file)
        output_path = os.path.join(OUTPUT_DIR, f"{Path(video_file).stem}_enhanced.avi")

        if os.path.exists(video_path):
            process_video(video_path, models, output_path)
        else:
            print(f"警告: 视频不存在 {video_path}")

    print(f"\n{'='*70}")
    print("完成!")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
