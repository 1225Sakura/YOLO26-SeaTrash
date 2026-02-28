#!/usr/bin/env python3
"""
海洋垃圾检测推理脚本
使用训练好的YOLO26模型进行目标检测
"""
import sys
sys.path.insert(0, '/home/user/sea/yolo26')

from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser(description='海洋垃圾目标检测推理')
    parser.add_argument('--source', type=str, required=True, help='输入图片/视频/目录路径')
    parser.add_argument('--model', type=str, default='runs/train/marine_debris_yolo26x2/weights/best.pt', help='模型路径')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IOU阈值')
    parser.add_argument('--imgsz', type=int, default=640, help='推理图像尺寸')
    parser.add_argument('--save', action='store_true', help='保存检测结果')
    parser.add_argument('--show', action='store_true', help='显示检测结果')

    args = parser.parse_args()

    # 加载模型
    print(f"加载模型: {args.model}")
    model = YOLO(args.model)

    # 运行推理
    print(f"开始推理: {args.source}")
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        save=args.save,
        show=args.show,
        device=0
    )

    print(f"推理完成！结果保存在: runs/detect/")

if __name__ == '__main__':
    main()
