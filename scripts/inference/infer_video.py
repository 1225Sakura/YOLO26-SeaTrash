#!/usr/bin/env python3
"""
Video Inference Script for YOLO26x Marine Debris Detection
Processes videos and generates annotated output with detection results
"""

import sys
from pathlib import Path
import cv2
from ultralytics import YOLO
import argparse
from tqdm import tqdm

def infer_video(model_path, video_path, output_dir, conf_threshold=0.25, iou_threshold=0.45):
    """
    Run inference on a video file

    Args:
        model_path: Path to YOLO model weights
        video_path: Path to input video
        output_dir: Directory to save output video
        conf_threshold: Confidence threshold for detections
        iou_threshold: IOU threshold for NMS
    """
    # Load model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    # Open video
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo: {video_path.name}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup output video
    output_path = output_dir / f"{video_path.stem}_detected.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Process video
    print(f"\nProcessing video...")
    frame_count = 0
    detection_stats = {}

    with tqdm(total=total_frames, desc="Processing") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference
            results = model(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)

            # Count detections
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = model.names[cls_id]
                        detection_stats[cls_name] = detection_stats.get(cls_name, 0) + 1

            # Draw results on frame
            annotated_frame = results[0].plot()

            # Write frame
            out.write(annotated_frame)
            frame_count += 1
            pbar.update(1)

    # Release resources
    cap.release()
    out.release()

    # Print statistics
    print(f"\n✅ Processing complete!")
    print(f"Output saved to: {output_path}")
    print(f"Processed {frame_count} frames")

    if detection_stats:
        print(f"\n📊 Detection Statistics:")
        for cls_name, count in sorted(detection_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cls_name}: {count} detections")
    else:
        print("\n⚠️  No objects detected in video")

    return output_path

def main():
    parser = argparse.ArgumentParser(description='YOLO26x Video Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights')
    parser.add_argument('--video', type=str, required=True, help='Path to video file or directory')
    parser.add_argument('--output', type=str, default='runs/inference/videos', help='Output directory')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold for NMS')

    args = parser.parse_args()

    video_path = Path(args.video)

    # Process single video or directory
    if video_path.is_file():
        infer_video(args.model, video_path, args.output, args.conf, args.iou)
    elif video_path.is_dir():
        video_files = list(video_path.glob('*.mp4')) + list(video_path.glob('*.avi'))
        print(f"Found {len(video_files)} videos in {video_path}")

        for video_file in video_files:
            print(f"\n{'='*60}")
            infer_video(args.model, video_file, args.output, args.conf, args.iou)
    else:
        print(f"Error: {video_path} is not a valid file or directory")
        sys.exit(1)

if __name__ == '__main__':
    main()
