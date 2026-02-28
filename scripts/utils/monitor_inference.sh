#!/bin/bash
# 实时监控高级推理进度

echo "=== 高级推理进度监控 ==="
echo ""

# 检查进程
if pgrep -f "python advanced_inference.py" > /dev/null; then
    echo "✅ 推理进程正在运行"
    PID=$(pgrep -f "python advanced_inference.py")
    echo "   PID: $PID"
    echo "   运行时间: $(ps -p $PID -o etime= | tr -d ' ')"
    echo ""
else
    echo "❌ 推理进程未运行"
    echo ""
fi

# 显示最新进度
echo "📊 最新进度:"
tail -20 /tmp/claude/-home-user-sea/tasks/ba1f50e.output 2>/dev/null | grep -E "(处理视频|总帧数|处理进度|视频:|有检测的帧|总检测数|输出:)" | tail -15

echo ""
echo "💾 输出文件:"
ls -lh runs/detect/advanced_inference/*.avi 2>/dev/null | awk '{print "   " $9 " - " $5}'

echo ""
echo "🔄 实时查看: tail -f /tmp/claude/-home-user-sea/tasks/ba1f50e.output"
