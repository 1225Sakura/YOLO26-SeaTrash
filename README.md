# YOLO26-SeaTrash: V11 Marine Debris Detection

🌊 基于YOLO26x的海洋垃圾智能检测系统 - V11材质分类版本

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-red.svg)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/YOLO-26x-green.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 项目简介

V11版本采用9类材质分类体系，整合12个海洋和陆地垃圾数据集，共75,173张图片、307,530个实例，实现高精度的材质导向垃圾检测。

### 🎯 性能指标

| 指标 | 数值 |
|------|------|
| **mAP50** | **82.54%** |
| **Precision** | **85.03%** |
| **Recall** | **76.85%** |
| **训练时长** | 18小时33分钟 |
| **训练轮数** | 272 epochs |
| **硬件配置** | 8× RTX 4090 |

### 🗂️ 9类材质分类体系

```
0: Plastic       (塑料)      - 60.61%
1: Glass         (玻璃)      - 4.23%
2: Metal         (金属)      - 7.37%
3: Paper         (纸张)      - 3.97%
4: Fiber         (纤维/布料) - 4.11%
5: Foam          (泡沫)      - 2.30%
6: Rubber        (橡胶)      - 1.06%
7: Electronics   (电子产品)  - 0.19%
8: Other         (其他)      - 16.16%
```

**优势**：
- ✅ 符合垃圾回收实际需求（按材质分类）
- ✅ 更好的类别平衡性
- ✅ 更高的数据质量（排除50,417个可疑实例）
- ✅ 更清晰的语义（材质类别更易理解）

### 📊 数据集统计

- **总图片数**: 75,173
  - 训练集: 62,282 (82.85%)
  - 验证集: 7,762 (10.33%)
  - 测试集: 5,129 (6.82%)
- **总实例数**: 307,530
- **场景构成**:
  - 海洋场景: 75.44% (56,712张)
  - 陆地场景: 24.56% (18,461张)
- **数据来源**: 整合12个海洋和陆地垃圾数据集
- **质量保证**: MD5去重，排除可疑实例

## 🚀 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.6+
- CUDA 12.4+ (GPU训练)
- 8GB+ GPU显存 (推理)
- 24GB+ GPU显存 (训练)

### 安装依赖

```bash
# 创建conda环境
conda create -n seatrash python=3.10 -y
conda activate seatrash

# 安装PyTorch (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安装依赖
pip install -r requirements.txt
```

### 训练模型

```bash
# 标准训练
python scripts/training/train_v11_material_8gpu.py

# 使用类别权重（推荐）
python scripts/training/train_v11_material_8gpu.py --weighted

# 使用重采样
python scripts/training/train_v11_material_8gpu.py --resample

# 完整优化（权重+重采样）
python scripts/training/train_v11_material_8gpu.py --weighted --resample

# 验证模型
python scripts/training/train_v11_material_8gpu.py \
  --mode validate \
  --weights runs/train/v11_material_9class_*/weights/best.pt
```

**训练优化**：
- ✅ 9类材质分类体系
- ✅ 类别加权Loss（上限5.0）
- ✅ 少数类别重采样（Electronics 20×, Rubber 3.7×）
- ✅ 海洋场景增强（75.44%海洋数据）
- ✅ 完整训练监控与可视化

### 视频推理

```bash
python scripts/inference/infer_video.py \
  --weights runs/train/v11_material_9class_*/weights/best.pt \
  --video path/to/video.mp4 \
  --output runs/inference/v11_videos/
```

**推理性能**：
- 处理速度: 20 FPS
- 检测精度: mAP50 82.54%
- 支持格式: MP4, AVI, MOV

## 📁 项目结构

```
YOLO26-SeaTrash/
├── README.md                 # 项目说明
├── MODEL_LOCATION.md         # 模型位置说明
├── LICENSE                   # MIT许可证
├── requirements.txt          # Python依赖
├── .gitignore                # Git忽略配置
├── scripts/                  # 脚本目录
│   ├── training/
│   │   └── train_v11_material_8gpu.py    # V11训练脚本
│   ├── inference/
│   │   └── infer_video.py                # 视频推理
│   └── data/
│       ├── integrate_material_9class.py  # 数据集整合
│       └── validate_material_9class.py   # 数据集验证
├── docs/                     # 文档目录
│   ├── guides/
│   │   └── V11_QUICKSTART.md             # 快速开始指南
│   ├── reports/
│   │   ├── V11_TRAINING_REPORT.md        # 训练报告
│   │   ├── V11_INFERENCE_REPORT.md       # 推理报告
│   │   └── MATERIAL_9CLASS_INTEGRATION_REPORT.md  # 数据集报告
│   ├── setup/
│   │   └── DIRECTORY_STRUCTURE.md        # 目录结构说明
│   ├── status/
│   │   └── V11_IMPLEMENTATION_SUMMARY.md # 实施总结
│   └── MATERIAL_MAPPING_FINAL.md         # 材质映射文档
├── data/                     # 数据目录 (被.gitignore排除)
│   ├── README.md
│   ├── datasets/             # 原始数据集
│   ├── material_trash_9class_v1/  # V11整合数据集
│   └── vido/                 # 测试视频
└── runs/                     # 训练输出 (被.gitignore排除)
    ├── README.md
    ├── train/
    │   └── v11_material_9class_*/  # V11训练结果
    └── inference/
        └── v11_videos/       # V11推理结果
```

## 📖 文档

- **[快速开始](docs/guides/V11_QUICKSTART.md)** - V11训练和推理指南
- **[训练报告](docs/reports/V11_TRAINING_REPORT.md)** - 完整训练结果分析
- **[推理报告](docs/reports/V11_INFERENCE_REPORT.md)** - 视频推理性能分析
- **[数据集报告](docs/reports/MATERIAL_9CLASS_INTEGRATION_REPORT.md)** - 数据集整合详情
- **[模型位置](MODEL_LOCATION.md)** - 模型文件位置和使用方法

## 🎯 使用场景

- 🌊 海洋环境监测
- 🚢 船载垃圾识别系统
- 🤖 水下机器人视觉
- 📹 海滩监控分析
- 🔬 海洋污染研究
- ♻️ 智能垃圾分类回收

## 📝 开发规范

本项目遵循 [RULES.md](RULES.md) 中定义的开发规范。

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)

## 👥 作者

1225Sakura

---

**版本**: V11.0
**最后更新**: 2026-03-12
**数据集**: material_trash_9class_v1 (75,173 images)
**模型**: runs/train/v11_material_9class_20260306_115502/weights/best.pt
