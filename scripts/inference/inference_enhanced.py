#!/usr/bin/env python3
"""
海洋垃圾检测 - 增强推理脚本
使用多尺度、TTA、模型集成等技术追求最高精度
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os
from tqdm import tqdm
from collections import defaultdict

# 配置
MODELS = [
    "/home/user/sea/training_archive/v1.0/models/best.pt",  # mAP50 75.63%
    "/home/user/sea/training_archive/v2.0/models/best.pt",  # mAP50 72.06%
]

VIDEO_DIR = "/home/user/sea/vido"
OUTPUT_DIR = "/home/user/sea/inference_enhanced"
VIDEO_FILES = [
    "海洋垃圾1.mp4",
    "海洋垃圾3.mp4"
]

# 增强推理配置
MULTI_SCALES = [480, 640, 800, 960]  # 多尺度推理
TTA_AUGMENTATIONS = [
    {'fliplr': 0.0, 'flipud': 0.0},  # 原始
    {'fliplr': 1.0, 'flipud': 0.0},  # 左右翻转
    {'fliplr': 0.0, 'flipud': 1.0},  # 上下翻转
]

# NMS配置
CONF_THRESHOLD = 0.15  # 降低置信度阈值以获取更多候选
IOU_THRESHOLD = 0.3    # 降低IOU阈值以保留更多检测


class EnhancedDetector:
    """增强检测器，支持多尺度、TTA、模型集成"""

    def __init__(self, model_paths, device=0):
        self.models = [YOLO(path) for path in model_paths]
        self.device = device
        print(f"加载了 {len(self.models)} 个模型用于集成")

    def detect_single_frame(self, frame, conf=0.15, iou=0.3):
        """
        对单帧进行增强检测
        使用多尺度 + TTA + 模型集成
        """
        all_detections = []

        # 遍历每个模型
        for model_idx, model in enumerate(self.models):
            # 多尺度推理
            for scale in MULTI_SCALES:
                # TTA增强
                for aug in TTA_AUGMENTATIONS:
                    # 应用增强
                    aug_frame = self._apply_augmentation(frame, aug)

                    # 调整尺寸
                    h, w = aug_frame.shape[:2]
                    scale_factor = scale / max(h, w)
                    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                    resized = cv2.resize(aug_frame, (new_w, new_h))

                    # 推理
                    results = model(resized, conf=conf, iou=iou,
                                   device=self.device, verbose=False)

                    # 提取检测结果
                    if len(results) > 0 and results[0].boxes is not None:
                        boxes = results[0].boxes
                        for i in range(len(boxes)):
                            # 获取坐标和置信度
                            xyxy = boxes.xyxy[i].cpu().numpy()
                            conf_score = float(boxes.conf[i].cpu().numpy())
                            cls = int(boxes.cls[i].cpu().numpy())

                            # 反向缩放到原始尺寸
                            xyxy = xyxy / scale_factor

                            # 反向增强变换
                            xyxy = self._reverse_augmentation(xyxy, aug, frame.shape)

                            all_detections.append({
                                'bbox': xyxy,
                                'conf': conf_score,
                                'cls': cls,
                                'model_idx': model_idx,
                                'scale': scale
                            })

        # 加权NMS融合所有检测结果
        final_detections = self._weighted_nms(all_detections, iou_threshold=iou)

        return final_detections

    def _apply_augmentation(self, frame, aug):
        """应用数据增强"""
        aug_frame = frame.copy()
        if aug['fliplr'] > 0.5:
            aug_frame = cv2.flip(aug_frame, 1)  # 左右翻转
        if aug['flipud'] > 0.5:
            aug_frame = cv2.flip(aug_frame, 0)  # 上下翻转
        return aug_frame

    def _reverse_augmentation(self, bbox, aug, shape):
        """反向增强变换"""
        h, w = shape[:2]
        x1, y1, x2, y2 = bbox

        if aug['flipud'] > 0.5:
            y1, y2 = h - y2, h - y1
        if aug['fliplr'] > 0.5:
            x1, x2 = w - x2, w - x1

        return np.array([x1, y1, x2, y2])

    def _weighted_nms(self, detections, iou_threshold=0.3):
        """
        加权NMS：融合多个检测结果
        使用置信度加权平均坐标
        """
        if len(detections) == 0:
            return []

        # 按类别分组
        class_detections = defaultdict(list)
        for det in detections:
            class_detections[det['cls']].append(det)

        final_results = []

        # 对每个类别单独处理
        for cls, dets in class_detections.items():
            # 转换为numpy数组
            boxes = np.array([d['bbox'] for d in dets])
            scores = np.array([d['conf'] for d in dets])

            # NMS
            keep_indices = self._nms(boxes, scores, iou_threshold)

            # 对保留的检测进行加权融合
            for idx in keep_indices:
                # 找到与该检测重叠的所有检测
                overlapping = []
                for i, det in enumerate(dets):
                    iou = self._compute_iou(boxes[idx], det['bbox'])
                    if iou > iou_threshold:
                        overlapping.append((det, iou))

                # 加权平均
                if len(overlapping) > 0:
                    total_weight = sum(d['conf'] * iou for d, iou in overlapping)
                    weighted_bbox = np.zeros(4)
                    weighted_conf = 0

                    for det, iou in overlapping:
                        weight = det['conf'] * iou / total_weight
                        weighted_bbox += det['bbox'] * weight
                        weighted_conf += det['conf'] * weight

                    final_results.append({
                        'bbox': weighted_bbox,
                        'conf': weighted_conf,
                        'cls': cls
                    })

        return final_results

    def _nms(self, boxes, scores, iou_threshold):
        """标准NMS"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def _compute_iou(self, box1, box2):
        """计算两个框的IOU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
        return iou


def process_video(video_path, detector, output_path):
    """处理视频"""
    print(f"\n{'='*70}")
    print(f"处理视频: {Path(video_path).name}")
    print(f"{'='*70}")

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        return

    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频信息: {width}x{height} @ {fps}fps, 总帧数: {total_frames}")

    # 创建输出视频
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 类别名称
    class_names = [
        'Plastic', 'Plastic_Bottle', 'Plastic_Buoy', 'Metal', 'Glass',
        'Paper', 'Cloth', 'Net', 'Rope', 'Styrofoam', 'Buoy', 'Rubber',
        'Natural_debris', 'Trash_Other'
    ]

    # 处理每一帧
    frame_idx = 0
    detection_stats = defaultdict(int)

    with tqdm(total=total_frames, desc="处理进度") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 增强检测
            detections = detector.detect_single_frame(
                frame,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD
            )

            # 绘制检测结果
            for det in detections:
                x1, y1, x2, y2 = det['bbox'].astype(int)
                conf = det['conf']
                cls = det['cls']

                # 统计
                detection_stats[class_names[cls]] += 1

                # 绘制边界框
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # 绘制标签
                label = f"{class_names[cls]}: {conf:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(frame, (x1, y1 - label_h - 10),
                            (x1 + label_w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # 写入输出视频
            out.write(frame)

            frame_idx += 1
            pbar.update(1)

    cap.release()
    out.release()

    print(f"\n检测统计:")
    for cls_name, count in sorted(detection_stats.items(),
                                   key=lambda x: x[1], reverse=True):
        print(f"  {cls_name}: {count} 次检测")

    print(f"\n输出保存到: {output_path}")


def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*70)
    print("海洋垃圾检测 - 增强推理模式")
    print("="*70)
    print(f"\n配置:")
    print(f"  模型数量: {len(MODELS)}")
    print(f"  多尺度: {MULTI_SCALES}")
    print(f"  TTA增强: {len(TTA_AUGMENTATIONS)} 种")
    print(f"  置信度阈值: {CONF_THRESHOLD}")
    print(f"  IOU阈值: {IOU_THRESHOLD}")
    print(f"  总推理次数/帧: {len(MODELS) * len(MULTI_SCALES) * len(TTA_AUGMENTATIONS)}")

    # 初始化检测器
    print(f"\n初始化增强检测器...")
    detector = EnhancedDetector(MODELS, device=0)

    # 处理每个视频
    for video_file in VIDEO_FILES:
        video_path = os.path.join(VIDEO_DIR, video_file)
        output_path = os.path.join(OUTPUT_DIR,
                                   f"{Path(video_file).stem}_enhanced.avi")

        if not os.path.exists(video_path):
            print(f"警告: 视频文件不存在 {video_path}")
            continue

        process_video(video_path, detector, output_path)

    print(f"\n{'='*70}")
    print("所有视频处理完成!")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
