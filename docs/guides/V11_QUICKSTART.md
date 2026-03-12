# V11 Material Classification - Quick Start Guide

**版本**: V11.0
**数据集**: material_trash_9class_v1
**类别**: 9类材质分类
**更新时间**: 2026-03-06

---

## 🎯 核心特性

### 9类材质分类体系

更符合垃圾回收实际需求的材质导向分类：

| ID | 材质 | 实例数 | 占比 | 特点 |
|----|------|--------|------|------|
| 0 | Plastic | 186,376 | 60.60% | 主导类别 |
| 1 | Glass | 13,000 | 4.23% | 中等类别 |
| 2 | Metal | 22,666 | 7.37% | 中等类别 |
| 3 | Paper | 12,655 | 4.12% | 中等类别 |
| 4 | Fiber | 12,214 | 3.97% | 中等类别 |
| 5 | Foam | 7,084 | 2.30% | 少数类别 |
| 6 | Rubber | 3,249 | 1.06% | 极少数类 |
| 7 | Electronics | 581 | 0.19% | 极少数类 |
| 8 | Other | 49,705 | 16.16% | 次主导 |

### 优势对比（vs V8-V10 11类体系）

- ✅ 更合理的分类：按材质分类，符合回收实际需求
- ✅ 更好的平衡：主导类别占比76.77% (vs 84%)
- ✅ 更高质量：排除50,417个可疑/无效实例
- ✅ 更清晰语义：材质类别更易理解和标注

---

## 📊 数据集信息

### 规模统计

- **总图片**: 75,156张
- **总实例**: 307,530个
- **数据来源**: 12个数据集（8海洋+4陆地）
- **场景构成**: 海洋75.44% | 陆地24.56%

### 划分统计

| 划分 | 图片数 | 实例数 | 占比 |
|------|--------|--------|------|
| Train | 52,613 | 215,585 | 70.0% |
| Valid | 15,028 | 61,147 | 20.0% |
| Test | 7,515 | 30,798 | 10.0% |

### 质量保证

- ✅ MD5重复: 0
- ✅ 无效标注: 0
- ✅ 图片-标注匹配: 100%
- ✅ 类别ID范围: 0-8
- ✅ 边界框坐标: 0-1范围

详细报告：`docs/reports/MATERIAL_9CLASS_INTEGRATION_REPORT.md`

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 激活环境
conda activate seatrash

# 验证CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

### 2. 验证数据集

```bash
# 运行验证脚本
python scripts/data/validate_material_9class.py data/material_trash_9class_v1

# 查看验证报告
cat data/material_trash_9class_v1/validation_report_v11.txt
```

### 3. 标准训练

```bash
# 基础训练（无优化）
python scripts/training/train_v11_material_8gpu.py

# 使用类别权重
python scripts/training/train_v11_material_8gpu.py --weighted

# 使用重采样
python scripts/training/train_v11_material_8gpu.py --resample

# 完整优化（推荐）
python scripts/training/train_v11_material_8gpu.py --weighted --resample
```

### 4. 监控训练

```bash
# 实时监控
watch -n 30 'tail -n 50 runs/train/v11_material_9class_*/results.csv'

# 查看TensorBoard
tensorboard --logdir runs/train/v11_material_9class_*
```

### 5. 验证模型

```bash
# 验证最佳模型
python scripts/training/train_v11_material_8gpu.py \
  --mode validate \
  --weights runs/train/v11_material_9class_*/weights/best.pt
```

---

## ⚙️  训练配置

### 硬件配置

- **GPU**: 8× NVIDIA RTX 4090 (24GB each)
- **总显存**: 192GB
- **Batch Size**: 64 (8 per GPU)
- **Workers**: 16
- **混合精度**: 启用（AMP）

### 训练参数

```python
TRAIN_CONFIG = {
    'epochs': 300,
    'img_size': 640,
    'batch_size': 64,
    'lr0': 0.001,
    'optimizer': 'AdamW',
    'patience': 50,
    'save_period': 10,
}
```

### 类别权重策略

针对类别不平衡问题，采用逆频率权重（上限5.0）：

```python
CLASS_WEIGHTS = {
    0: 1.0,     # Plastic (主导)
    1: 2.86,    # Glass
    2: 1.64,    # Metal
    3: 2.94,    # Paper
    4: 3.05,    # Fiber
    5: 5.0,     # Foam (限制)
    6: 5.0,     # Rubber (限制)
    7: 5.0,     # Electronics (限制)
    8: 1.0      # Other (次主导)
}
```

### 重采样策略

针对极少数类别，提升到中等水平（~12000实例）：

```python
RESAMPLE_CONFIG = {
    7: 20.0,    # Electronics: 581 → 11,620
    6: 3.7,     # Rubber: 3,249 → 12,021
    5: 1.7,     # Foam: 7,084 → 12,043
}
```

### 数据增强

中等强度增强，平衡泛化和过拟合：

- **色彩**: hsv_h=0.015, hsv_s=0.7, hsv_v=0.4
- **几何**: degrees=10°, translate=0.1, scale=0.7
- **混合**: mosaic=0.8, mixup=0.1, copy_paste=0.1
- **翻转**: fliplr=0.5 (水平), flipud=0.0 (无垂直)

---

## 📈 优化策略

### 1. 类别不平衡处理

**问题**:
- Plastic占60.60%（主导）
- Electronics仅0.19%（极少数）
- 类别分布严重不均

**解决方案**:
- ✅ 类别加权Loss（上限5.0）
- ✅ 少数类别重采样（20×）
- ✅ 分层采样确保比例
- ✅ 针对性数据增强

### 2. 少数类别增强

**目标类别**: Electronics (0.19%), Rubber (1.06%), Foam (2.30%)

**策略**:
- 重采样提升实例数
- 更高的Loss权重
- Copy-paste增强
- Mixup混合增强

### 3. 海洋场景优化

**海洋数据占比**: 75.44%

**增强策略**:
- 水下模糊模拟
- 色调偏移（蓝绿色）
- 亮度衰减（水深效果）
- 对比度降低（散射效果）

### 4. 训练稳定性

- **Warmup**: 5 epochs渐进学习率
- **Patience**: 50 epochs早停
- **Close Mosaic**: 最后10 epochs关闭mosaic
- **Save Period**: 每10 epochs保存检查点

---

## 📊 预期性能

### 目标指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| mAP50 | 70%+ | 整体检测精度 |
| mAP50-95 | 50%+ | 严格IoU精度 |
| Precision | 70%+ | 精确率 |
| Recall | 65%+ | 召回率 |

### 类别性能预期

- **主导类别** (Plastic, Other): mAP50 > 80%
- **中等类别** (Metal, Glass, Paper, Fiber): mAP50 > 65%
- **少数类别** (Foam, Rubber): mAP50 > 50%
- **极少数类** (Electronics): mAP50 > 30%

---

## 🔧 故障排除

### 1. OOM (Out of Memory)

```bash
# 减小batch size
python scripts/training/train_v11_material_8gpu.py --batch 32

# 或修改脚本中的TRAIN_CONFIG['batch_size']
```

### 2. 训练不收敛

- 检查学习率（可能过大）
- 增加warmup epochs
- 检查数据增强强度
- 验证数据集质量

### 3. 少数类别性能差

- 启用类别权重: `--weighted`
- 启用重采样: `--resample`
- 增加训练epochs
- 调整Loss权重

### 4. 过拟合

- 增强数据增强强度
- 增加weight_decay
- 启用dropout（需修改模型）
- 减少训练epochs

---

## 📝 训练日志

### 关键指标监控

训练过程中关注以下指标：

1. **Loss曲线**:
   - train/box_loss: 边界框回归损失
   - train/cls_loss: 分类损失
   - val/box_loss, val/cls_loss: 验证损失

2. **性能指标**:
   - metrics/mAP50(B): 主要评估指标
   - metrics/precision(B): 精确率
   - metrics/recall(B): 召回率

3. **学习率**:
   - lr/pg0, lr/pg1, lr/pg2: 各参数组学习率

### 输出文件

```
runs/train/v11_material_9class_YYYYMMDD_HHMMSS/
├── weights/
│   ├── best.pt          # 最佳模型
│   ├── last.pt          # 最后模型
│   └── epoch*.pt        # 周期检查点
├── results.csv          # 训练结果
├── results.png          # 结果图表
├── confusion_matrix.png # 混淆矩阵
├── F1_curve.png         # F1曲线
├── PR_curve.png         # PR曲线
└── args.yaml            # 训练参数
```

---

## 🎓 最佳实践

### 1. 训练前

- ✅ 验证数据集完整性
- ✅ 检查GPU状态和显存
- ✅ 备份重要模型
- ✅ 设置合理的patience

### 2. 训练中

- ✅ 定期检查训练日志
- ✅ 监控Loss曲线趋势
- ✅ 观察验证集性能
- ✅ 及时调整超参数

### 3. 训练后

- ✅ 在测试集上验证
- ✅ 分析混淆矩阵
- ✅ 检查少数类别性能
- ✅ 保存最佳模型和配置

---

## 📚 相关文档

- [数据集整合报告](../reports/MATERIAL_9CLASS_INTEGRATION_REPORT.md)
- [数据集验证报告](../../data/material_trash_9class_v1/validation_report_v11.txt)
- [项目README](../../README.md)
- [开发规范](../../RULES.md)

---

## 🤝 支持

遇到问题？

1. 检查本文档的故障排除部分
2. 查看训练日志和错误信息
3. 验证数据集和环境配置
4. 参考V8训练经验

---

**最后更新**: 2026-03-06
**脚本**: `scripts/training/train_v11_material_8gpu.py`
**数据集**: `data/material_trash_9class_v1`
