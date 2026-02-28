#!/bin/bash
# v4.0 Training Launcher for 8× RTX 4090
# Usage: ./start_v4_training.sh

set -e

echo "========================================"
echo "YOLO26-SeaTrash v4.0 Training Launcher"
echo "========================================"
echo ""
echo "Hardware: 8× NVIDIA RTX 4090"
echo "Batch Size: 80 (10 per GPU)"
echo "Target: mAP50 ≥ 90%"
echo ""

# Check GPU availability
echo "Checking GPU status..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

echo ""
echo "Starting training in 5 seconds..."
echo "Press Ctrl+C to cancel"
sleep 5

# Create log directory
mkdir -p /home/user/sea/logs

# Start training with nohup
LOG_FILE="/home/user/sea/logs/train_v4_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "Training started!"
echo "Log file: $LOG_FILE"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""

cd /home/user/sea
nohup python3 scripts/training/train_v4_8gpu.py > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "Training PID: $TRAIN_PID"
echo ""
echo "To stop training:"
echo "  kill $TRAIN_PID"
echo ""
echo "========================================"
