# 海洋垃圾目标检测 - 训练总结

## 训练完成情况

✓ 训练已成功完成
- 完成时间: 2026-01-20 10:09
- 总训练时长: 约18小时
- 完成Epochs: 90/100（可能触发早停）

## 最终性能指标

### 最佳模型 (Epoch 44)
- **mAP50: 75.63%** - 在IoU=0.5时的平均精度
- **mAP50-95: 64.12%** - 在IoU=0.5-0.95的平均精度
- **Precision: 75.97%** - 精确率
- **Recall: 69.40%** - 召回率

### 最终模型 (Epoch 90)
- mAP50: 75.32%
- mAP50-95: 63.92%
- Precision: 76.52%
- Recall: 69.51%

### 性能提升
- mAP50: 37.16% → 75.63% (+38.47%)
- mAP50-95: 24.96% → 64.12% (+39.16%)
- Recall: 36.79% → 69.40% (+32.61%)

## 模型文件

```
runs/train/marine_debris_yolo26x2/weights/
├── best.pt (338MB) - 最佳模型 (推荐使用)
├── last.pt (338MB) - 最终模型
└── epoch*.pt - 每10个epoch的检查点
```

## 数据集信息

- 总图片数: 44,637张
- 总标注数: 97,599个
- 类别数: 15类海洋垃圾
- 训练集/验证集/测试集划分已完成

## 使用方法

### 1. 推理（检测）

```bash
# 单张图片检测
python3 inference.py --source /path/to/image.jpg --save

# 批量检测（目录）
python3 inference.py --source /path/to/images/ --save

# 视频检测
python3 inference.py --source /path/to/video.mp4 --save

# 调整置信度阈值
python3 inference.py --source image.jpg --conf 0.5 --save
```

### 2. 模型验证

```bash
cd yolo26
python3 -c "from ultralytics import YOLO; model = YOLO('../runs/train/marine_debris_yolo26x2/weights/best.pt'); model.val(data='../configs/marine_debris.yaml')"
```

### 3. 导出模型

```bash
# 导出为ONNX格式
cd yolo26
python3 -c "from ultralytics import YOLO; model = YOLO('../runs/train/marine_debris_yolo26x2/weights/best.pt'); model.export(format='onnx')"

# 导出为TensorRT格式（需要TensorRT）
python3 -c "from ultralytics import YOLO; model = YOLO('../runs/train/marine_debris_yolo26x2/weights/best.pt'); model.export(format='engine')"
```

## 项目结构

```
/home/user/sea/
├── yolo26/                    # YOLO26源代码
├── datasets/
│   └── marine_debris/         # 数据集（44,637张图片）
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
├── configs/
│   └── marine_debris.yaml     # 数据集配置
├── model/
│   └── yolo26x.pt            # 预训练模型
├── runs/
│   └── train/
│       └── marine_debris_yolo26x2/  # 训练结果
│           ├── weights/
│           │   └── best.pt   # 最佳模型 ⭐
│           ├── results.csv   # 训练指标
│           └── *.jpg         # 训练可视化
├── train.py                   # 训练脚本
├── inference.py               # 推理脚本 ⭐
└── monitor_training.sh        # 监控脚本
```

## 15类海洋垃圾类别

根据 classes.txt:
1. Plastic bottle (塑料瓶)
2. Plastic bag (塑料袋)
3. Can (罐头)
4. Glass (玻璃)
5. Paper (纸张)
6. Cardboard (纸板)
7. Metal (金属)
8. Cloth (布料)
9. Wood (木材)
10. Rubber (橡胶)
11. Foam (泡沫)
12. Fishing net (渔网)
13. Rope (绳索)
14. Cigarette (烟蒂)
15. Other (其他)

## 训练配置

- 模型: YOLO26x (58.8M参数)
- GPU: NVIDIA RTX 4090
- Batch size: 16
- Image size: 640x640
- Optimizer: AdamW
- 数据增强: 翻转、HSV、平移、缩放、Mosaic等

## 下一步建议

1. **模型部署**: 使用 best.pt 进行实际应用
2. **性能优化**: 如需更快推理，可导出为ONNX或TensorRT格式
3. **进一步微调**: 如果特定类别性能不佳，可针对性增加数据
4. **模型集成**: 可与其他模型集成提升性能

## 训练曲线

查看训练过程可视化:
```bash
# 查看详细训练指标
cat runs/train/marine_debris_yolo26x2/results.csv

# 查看训练批次样本
ls runs/train/marine_debris_yolo26x2/train_batch*.jpg
```
