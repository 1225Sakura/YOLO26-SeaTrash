# V11 Model Location

## Best Model (推荐使用)

**路径**：
```
/home/roots/YOLO26-SeaTrash/runs/train/v11_material_9class_20260306_115502/weights/best.pt
```

**模型信息**：
- 大小：113MB
- mAP50：82.54%
- Precision：85.03%
- Recall：76.85%
- 训练时间：18小时33分钟
- 训练轮数：272 epochs
- 硬件：8x RTX 4090

## 使用方法

### 视频推理
```bash
python scripts/inference/infer_video.py \
    --weights runs/train/v11_material_9class_20260306_115502/weights/best.pt \
    --video data/vido/海洋垃圾1.mp4 \
    --output runs/inference/v11_videos/
```

### 图片推理
```bash
from ultralytics import YOLO

model = YOLO('runs/train/v11_material_9class_20260306_115502/weights/best.pt')
results = model.predict('image.jpg', conf=0.25)
```

## 其他检查点

**Last Checkpoint**：
```
runs/train/v11_material_9class_20260306_115502/weights/last.pt
```
- 用于恢复训练

**Epoch Checkpoints**：
```
runs/train/v11_material_9class_20260306_115502/weights/epoch*.pt
```
- 每10个epoch保存一次
- 用于分析训练过程

## 推理结果

V11推理输出位置：
```
runs/inference/v11_videos/
├── 海洋垃圾1_detected.mp4  (30MB)
├── 海洋垃圾2_detected.mp4  (69MB)
└── 海洋垃圾3_detected.mp4  (36MB)
```

## 9类材质分类

1. **Plastic** (塑料) - 35%
2. **Glass** (玻璃) - 6%
3. **Metal** (金属) - 8%
4. **Fiber** (纤维) - 5%
5. **Paper** (纸张) - 6%
6. **Foam** (泡沫) - 3%
7. **Rubber** (橡胶) - 2%
8. **Electronics** (电子产品) - 1%
9. **Other** (其他) - 25%
