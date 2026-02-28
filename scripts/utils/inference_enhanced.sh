#!/bin/bash
# 海洋垃圾检测 - 增强推理脚本

echo "======================================================================"
echo "         海洋垃圾检测 - 增强推理模式"
echo "======================================================================"
echo ""

# 配置
MODEL1="/home/user/sea/training_archive/v1.0/models/best.pt"
MODEL2="/home/user/sea/training_archive/v2.0/models/best.pt"
VIDEO_DIR="/home/user/sea/vido"
OUTPUT_DIR="/home/user/sea/inference_enhanced"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 视频列表
VIDEOS=("海洋垃圾1.mp4" "海洋垃圾3.mp4")

# 多尺度配置
SCALES=(640 800 960 1120)

echo "配置:"
echo "  模型1: $MODEL1 (mAP50 75.63%)"
echo "  模型2: $MODEL2 (mAP50 72.06%)"
echo "  多尺度: ${SCALES[@]}"
echo "  TTA: 启用"
echo ""

# 处理每个视频
for video in "${VIDEOS[@]}"; do
    echo "======================================================================"
    echo "处理视频: $video"
    echo "======================================================================"

    video_path="$VIDEO_DIR/$video"
    video_name="${video%.mp4}"

    if [ ! -f "$video_path" ]; then
        echo "警告: 视频文件不存在 $video_path"
        continue
    fi

    # 对每个模型和尺度进行推理
    temp_outputs=()
    idx=0

    for model in "$MODEL1" "$MODEL2"; do
        for scale in "${SCALES[@]}"; do
            echo ""
            echo "推理: 模型$(basename $model) @ ${scale}px (TTA启用)"

            temp_output="$OUTPUT_DIR/temp_${video_name}_${idx}.avi"
            temp_outputs+=("$temp_output")

            yolo predict \
                model="$model" \
                source="$video_path" \
                imgsz=$scale \
                conf=0.20 \
                iou=0.35 \
                augment=True \
                device=0 \
                save=True \
                project="$OUTPUT_DIR" \
                name="temp_${video_name}_${idx}" \
                exist_ok=True \
                verbose=False

            idx=$((idx + 1))
        done
    done

    echo ""
    echo "✓ 完成 $video 的所有推理"
    echo "  生成了 ${#temp_outputs[@]} 个临时结果"

done

echo ""
echo "======================================================================"
echo "所有视频处理完成!"
echo "输出目录: $OUTPUT_DIR"
echo "======================================================================"
