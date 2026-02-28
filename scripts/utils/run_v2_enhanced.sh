#!/bin/bash
# v2.0模型增强推理脚本

MODEL="/home/user/sea/training_archive/v2.0/models/best.pt"
VIDEO_DIR="/home/user/sea/vido"
OUTPUT_DIR="/home/user/sea/inference_v2.0_enhanced"

mkdir -p "$OUTPUT_DIR"

echo "======================================================================"
echo "海洋垃圾检测 - v2.0模型增强推理"
echo "======================================================================"
echo ""
echo "模型: v2.0 (mAP50=72.06%, 精确率=86.13%)"
echo "多尺度: 640, 800, 960, 1120, 1280"
echo "TTA: 启用"
echo ""

# 海洋垃圾1
echo "======================================================================"
echo "处理: 海洋垃圾1.mp4"
echo "======================================================================"

for scale in 640 800 960 1120 1280; do
    echo ""
    echo "尺度 ${scale}px + TTA..."
    yolo predict \
        model="$MODEL" \
        source="$VIDEO_DIR/海洋垃圾1.mp4" \
        imgsz=$scale \
        conf=0.20 \
        iou=0.35 \
        augment=True \
        device=0 \
        save=True \
        project="$OUTPUT_DIR" \
        name="海洋垃圾1_scale${scale}" \
        exist_ok=True
    echo "✓ 完成 ${scale}px"
done

# 海洋垃圾3
echo ""
echo "======================================================================"
echo "处理: 海洋垃圾3.mp4"
echo "======================================================================"

for scale in 640 800 960 1120 1280; do
    echo ""
    echo "尺度 ${scale}px + TTA..."
    yolo predict \
        model="$MODEL" \
        source="$VIDEO_DIR/海洋垃圾3.mp4" \
        imgsz=$scale \
        conf=0.20 \
        iou=0.35 \
        augment=True \
        device=0 \
        save=True \
        project="$OUTPUT_DIR" \
        name="海洋垃圾3_scale${scale}" \
        exist_ok=True
    echo "✓ 完成 ${scale}px"
done

echo ""
echo "======================================================================"
echo "所有推理完成!"
echo "输出目录: $OUTPUT_DIR"
echo "======================================================================"
