#!/bin/bash
# 海洋垃圾检测训练监控脚本

echo "=== YOLO26 海洋垃圾检测训练监控 ==="
echo ""

# 检查训练进程
TRAIN_PID=$(pgrep -f "python3 train.py" | head -1)
if [ -n "$TRAIN_PID" ]; then
    echo "✓ 训练进程运行中 (PID: $TRAIN_PID)"
    TRAIN_TIME=$(ps -p $TRAIN_PID -o etime= | tr -d ' ')
    echo "  运行时间: $TRAIN_TIME"
else
    echo "✗ 训练进程未运行"
fi

echo ""
echo "=== 训练进度 ==="
RESULTS_FILE="/home/user/sea/runs/train/marine_debris_yolo26x2/results.csv"
if [ -f "$RESULTS_FILE" ]; then
    echo "最新训练结果:"
    tail -5 "$RESULTS_FILE" | column -t -s,
    echo ""
    EPOCH_COUNT=$(tail -n +2 "$RESULTS_FILE" | wc -l)
    echo "已完成 epoch: $EPOCH_COUNT / 100"
else
    echo "训练结果文件尚未生成"
fi

echo ""
echo "=== 模型文件 ==="
WEIGHTS_DIR="/home/user/sea/runs/train/marine_debris_yolo26x2/weights"
if [ -d "$WEIGHTS_DIR" ]; then
    ls -lh "$WEIGHTS_DIR"
else
    echo "权重文件目录尚未创建"
fi

echo ""
echo "=== GPU使用情况 ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader

echo ""
echo "提示："
echo "- 训练结果: /home/user/sea/runs/train/marine_debris_yolo26x2/"
echo "- 最佳模型: runs/train/marine_debris_yolo26x2/weights/best.pt"
echo "- 查看详细结果: cat $RESULTS_FILE"
