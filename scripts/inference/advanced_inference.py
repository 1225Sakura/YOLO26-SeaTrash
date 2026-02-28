#!/usr/bin/env python3
"""
高级推理脚本 - 使用多种方法提升检测性能
- 模型集成（Ensemble）
- 测试时增强（TTA）
- 多尺度推理
- 加权框融合（WBF）
"""
import sys
sys.path.insert(0, '/home/user/sea/yolo26')

import cv2
import numpy as np
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from pathlib import Path
import torch
from tqdm import tqdm

class AdvancedInference:
    def __init__(self, model_paths, conf_threshold=0.1, iou_threshold=0.5, device='cuda:0'):
        """
        初始化高级推理器
        Args:
            model_paths: 模型权重路径列表
            conf_threshold: 置信度阈值（降低以提高召回率）
            iou_threshold: IOU阈值
            device: 设备
        """
        print(f"加载 {len(model_paths)} 个模型...")
        self.models = [YOLO(path) for path in model_paths]
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        # 获取类别名称
        self.class_names = self.models[0].names
        print(f"类别数: {len(self.class_names)}")
        print(f"类别: {self.class_names}")

    def tta_transforms(self, image):
        """
        测试时增强变换
        返回: [(变换后的图像, 反变换函数), ...]
        """
        h, w = image.shape[:2]
        transforms = []

        # 原始图像
        transforms.append((image.copy(), lambda boxes: boxes))

        # 水平翻转
        flipped_h = cv2.flip(image, 1)
        def unflip_h(boxes):
            boxes[:, 0] = w - boxes[:, 0]  # x1
            boxes[:, 2] = w - boxes[:, 2]  # x2
            boxes[:, [0, 2]] = boxes[:, [2, 0]]  # swap
            return boxes
        transforms.append((flipped_h, unflip_h))

        # 垂直翻转
        flipped_v = cv2.flip(image, 0)
        def unflip_v(boxes):
            boxes[:, 1] = h - boxes[:, 1]  # y1
            boxes[:, 3] = h - boxes[:, 3]  # y2
            boxes[:, [1, 3]] = boxes[:, [3, 1]]  # swap
            return boxes
        transforms.append((flipped_v, unflip_v))

        # 轻微旋转调整（小角度）
        # 90度旋转
        rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        def unrotate_90(boxes):
            # x' = h - y, y' = x
            new_boxes = boxes.copy()
            new_boxes[:, 0] = h - boxes[:, 1]  # x1
            new_boxes[:, 1] = boxes[:, 0]      # y1
            new_boxes[:, 2] = h - boxes[:, 3]  # x2
            new_boxes[:, 3] = boxes[:, 2]      # y2
            new_boxes[:, [0, 2]] = np.sort(new_boxes[:, [0, 2]], axis=1)
            new_boxes[:, [1, 3]] = np.sort(new_boxes[:, [1, 3]], axis=1)
            return new_boxes
        transforms.append((rotated_90, unrotate_90))

        return transforms

    def multi_scale_inference(self, image, model, scales=[0.8, 1.0, 1.2]):
        """
        多尺度推理
        """
        h, w = image.shape[:2]
        all_boxes = []
        all_scores = []
        all_labels = []

        for scale in scales:
            new_h, new_w = int(h * scale), int(w * scale)
            resized = cv2.resize(image, (new_w, new_h))

            results = model.predict(
                resized,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
                device=self.device
            )[0]

            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                # 缩放回原始尺寸
                boxes[:, [0, 2]] = boxes[:, [0, 2]] / scale
                boxes[:, [1, 3]] = boxes[:, [1, 3]] / scale

                scores = results.boxes.conf.cpu().numpy()
                labels = results.boxes.cls.cpu().numpy().astype(int)

                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.append(labels)

        if len(all_boxes) == 0:
            return np.array([]), np.array([]), np.array([])

        return np.vstack(all_boxes), np.concatenate(all_scores), np.concatenate(all_labels)

    def predict_frame(self, image, use_tta=True, use_multiscale=True):
        """
        对单帧进行集成推理
        """
        h, w = image.shape[:2]

        # 收集所有模型的所有预测
        all_boxes_list = []
        all_scores_list = []
        all_labels_list = []

        for model_idx, model in enumerate(self.models):
            if use_tta:
                # 使用TTA
                transforms = self.tta_transforms(image)
                for trans_img, reverse_fn in transforms:
                    if use_multiscale:
                        boxes, scores, labels = self.multi_scale_inference(trans_img, model)
                    else:
                        results = model.predict(
                            trans_img,
                            conf=self.conf_threshold,
                            iou=self.iou_threshold,
                            verbose=False,
                            device=self.device
                        )[0]

                        if results.boxes is not None and len(results.boxes) > 0:
                            boxes = results.boxes.xyxy.cpu().numpy()
                            scores = results.boxes.conf.cpu().numpy()
                            labels = results.boxes.cls.cpu().numpy().astype(int)
                        else:
                            boxes, scores, labels = np.array([]), np.array([]), np.array([])

                    if len(boxes) > 0:
                        # 反向变换boxes
                        boxes = reverse_fn(boxes)
                        all_boxes_list.append(boxes)
                        all_scores_list.append(scores)
                        all_labels_list.append(labels)
            else:
                # 不使用TTA
                if use_multiscale:
                    boxes, scores, labels = self.multi_scale_inference(image, model)
                else:
                    results = model.predict(
                        image,
                        conf=self.conf_threshold,
                        iou=self.iou_threshold,
                        verbose=False,
                        device=self.device
                    )[0]

                    if results.boxes is not None and len(results.boxes) > 0:
                        boxes = results.boxes.xyxy.cpu().numpy()
                        scores = results.boxes.conf.cpu().numpy()
                        labels = results.boxes.cls.cpu().numpy().astype(int)
                    else:
                        boxes, scores, labels = np.array([]), np.array([]), np.array([])

                if len(boxes) > 0:
                    all_boxes_list.append(boxes)
                    all_scores_list.append(scores)
                    all_labels_list.append(labels)

        if len(all_boxes_list) == 0:
            return [], [], []

        # 使用WBF融合所有检测框
        return self.weighted_boxes_fusion(
            all_boxes_list, all_scores_list, all_labels_list,
            image_size=(w, h)
        )

    def weighted_boxes_fusion(self, boxes_list, scores_list, labels_list, image_size):
        """
        加权框融合
        """
        w, h = image_size

        # 将boxes归一化到[0, 1]
        normalized_boxes_list = []
        for boxes in boxes_list:
            if len(boxes) > 0:
                norm_boxes = boxes.copy()
                norm_boxes[:, [0, 2]] /= w
                norm_boxes[:, [1, 3]] /= h
                normalized_boxes_list.append(norm_boxes.tolist())
            else:
                normalized_boxes_list.append([])

        scores_list = [s.tolist() if len(s) > 0 else [] for s in scores_list]
        labels_list = [l.tolist() if len(l) > 0 else [] for l in labels_list]

        # WBF融合
        try:
            fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                normalized_boxes_list,
                scores_list,
                labels_list,
                weights=None,  # 平均权重
                iou_thr=self.iou_threshold,
                skip_box_thr=self.conf_threshold * 0.5  # 更低的阈值
            )

            # 反归一化
            if len(fused_boxes) > 0:
                fused_boxes[:, [0, 2]] *= w
                fused_boxes[:, [1, 3]] *= h

            return fused_boxes, fused_scores, fused_labels
        except Exception as e:
            print(f"WBF融合失败: {e}")
            # 如果WBF失败，返回第一个模型的结果
            if len(boxes_list) > 0 and len(boxes_list[0]) > 0:
                return boxes_list[0], scores_list[0], np.array(labels_list[0])
            return np.array([]), np.array([]), np.array([])

    def draw_detections(self, image, boxes, scores, labels):
        """
        在图像上绘制检测框
        """
        img_draw = image.copy()

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box)
            class_name = self.class_names[int(label)]

            # 绘制框
            color = (0, 255, 0)
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            label_text = f"{class_name}: {score:.2f}"
            label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img_draw, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(img_draw, label_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return img_draw

    def process_video(self, video_path, output_path, use_tta=True, use_multiscale=True):
        """
        处理视频
        """
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        detection_stats = {
            'total_frames': total_frames,
            'frames_with_detection': 0,
            'total_detections': 0,
            'detections_per_class': {}
        }

        print(f"\n处理视频: {video_path}")
        print(f"总帧数: {total_frames}, 分辨率: {width}x{height}, FPS: {fps}")
        print(f"使用TTA: {use_tta}, 使用多尺度: {use_multiscale}")

        frame_idx = 0
        with tqdm(total=total_frames, desc="处理进度") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 推理
                boxes, scores, labels = self.predict_frame(
                    frame, use_tta=use_tta, use_multiscale=use_multiscale
                )

                # 统计
                if len(boxes) > 0:
                    detection_stats['frames_with_detection'] += 1
                    detection_stats['total_detections'] += len(boxes)

                    for label in labels:
                        class_name = self.class_names[int(label)]
                        detection_stats['detections_per_class'][class_name] = \
                            detection_stats['detections_per_class'].get(class_name, 0) + 1

                # 绘制
                frame_drawn = self.draw_detections(frame, boxes, scores, labels)
                out.write(frame_drawn)

                frame_idx += 1
                pbar.update(1)

        cap.release()
        out.release()

        return detection_stats


def main():
    # 配置
    model_paths = [
        '/home/user/sea/runs/train/marine_debris_yolo26x2/weights/best.pt',
        '/home/user/sea/runs/train/marine_debris_yolo26x2/weights/last.pt',
        '/home/user/sea/runs/train/marine_debris_yolo26x2/weights/epoch80.pt',
    ]

    videos = [
        '/home/user/sea/vido/海洋垃圾1.mp4',
        '/home/user/sea/vido/海洋垃圾2.mp4',
        '/home/user/sea/vido/海洋垃圾3.mp4'
    ]

    output_dir = Path('runs/detect/advanced_inference')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化推理器
    print("="*60)
    print("高级集成推理系统")
    print("="*60)
    print(f"模型数量: {len(model_paths)}")
    print(f"置信度阈值: 0.1 (降低以提高召回率)")
    print(f"IOU阈值: 0.5")
    print(f"使用方法: 模型集成 + TTA + 多尺度 + WBF")
    print("="*60)

    inferencer = AdvancedInference(
        model_paths=model_paths,
        conf_threshold=0.1,
        iou_threshold=0.5,
        device='cuda:0'
    )

    # 处理每个视频
    all_stats = {}
    for video_path in videos:
        video_name = Path(video_path).stem
        output_path = str(output_dir / f"{video_name}_advanced.avi")

        stats = inferencer.process_video(
            video_path,
            output_path,
            use_tta=True,
            use_multiscale=True
        )

        all_stats[video_name] = stats

        # 打印统计
        print(f"\n{'='*60}")
        print(f"视频: {video_name}")
        print(f"{'='*60}")
        print(f"总帧数: {stats['total_frames']}")
        print(f"有检测的帧: {stats['frames_with_detection']} "
              f"({stats['frames_with_detection']/stats['total_frames']*100:.1f}%)")
        print(f"总检测数: {stats['total_detections']}")
        print(f"每类检测数:")
        for cls, count in sorted(stats['detections_per_class'].items(),
                                 key=lambda x: x[1], reverse=True):
            print(f"  {cls}: {count}")
        print(f"输出: {output_path}")

    print(f"\n{'='*60}")
    print("所有视频处理完成！")
    print(f"结果保存在: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
