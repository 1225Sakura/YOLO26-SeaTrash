# 项目目录结构说明

本文档说明YOLO26-SeaTrash项目的目录组织结构（遵循RULES.md规范）。

## 📁 根目录结构

```
YOLO26-SeaTrash/
├── README.md                 # 项目介绍和快速开始
├── PROGRESS.md              # 任务进度跟踪
├── RULES.md                 # 项目规则（死规则）
├── configs/                 # 配置文件
├── data/                    # 数据集目录
│   ├── datasets/           # 原始数据集
│   └── material_trash_9class_v1/  # 9类材质整合数据集
├── docs/                    # 文档目录（详见下文）
├── runs/                    # 训练运行记录
├── scripts/                 # 脚本目录（详见下文）
├── src/                     # 源代码
└── public/                  # 前端文件
```

## 📚 docs/ 目录结构

```
docs/
├── MATERIAL_MAPPING_FINAL.md  # 9类材质映射规则（当前使用）
├── api/                       # API接口文档
├── guides/                    # 使用指南和教程
├── setup/                     # 配置和部署指南
│   └── DIRECTORY_STRUCTURE.md # 本文档
├── reports/                   # 项目报告
├── status/                    # 状态和验证文档
└── archive/                   # 历史文档归档
    ├── analysis/             # 分析文档（R1-R6等）
    ├── training/             # 训练文档（V5-V8等）
    ├── integration/          # 整合文档
    └── mapping/              # 映射文档
```

### 文档管理规则

- **根目录最多3个MD文件**: README.md, PROGRESS.md, RULES.md
- **优先更新现有文档**，而不是创建新文档
- **过时文档移至archive/**，按类型分类存放

## 🔧 scripts/ 目录结构

```
scripts/
├── data/                    # 数据处理脚本
│   ├── integrate_material_9class.py  # 9类材质数据集整合
│   ├── integrate_marine_land_v1.py   # 11类数据集整合（旧）
│   ├── clean_dataset.py
│   └── ...
├── training/                # 训练脚本
│   ├── train_v4_8gpu.py
│   └── ...
├── inference/               # 推理脚本
│   └── infer_video.py
├── augmentation/            # 数据增强脚本
├── utils/                   # 工具和辅助脚本
│   ├── check_training_progress.sh
│   ├── monitor_v8_training.sh
│   └── ...
├── startup/                 # 启动和停止脚本
└── tests/                   # 测试脚本
```

### 脚本分类标准

- **data/**: 数据集处理、清洗、整合
- **training/**: 模型训练相关
- **inference/**: 模型推理和预测
- **augmentation/**: 数据增强
- **utils/**: 通用工具脚本
- **startup/**: 服务启动/停止脚本
- **tests/**: 测试脚本

## 🗄️ data/ 目录结构

```
data/
├── datasets/                # 原始数据集
│   └── 未/                 # 未整合数据集
│       ├── ocean waste.v2i.yolo26/
│       ├── TrashCan to Yolo.v2-test-version-full-.yolo26/
│       └── ...
└── material_trash_9class_v1/  # 9类材质整合数据集（当前使用）
    ├── images/
    │   ├── train/
    │   ├── valid/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── valid/
    │   └── test/
    ├── data.yaml
    ├── statistics.json
    └── integration_errors.log
```

## 📊 9类材质体系

当前项目使用9类材质分类体系（替代原11类体系）：

```
0. Plastic      - 塑料
1. Glass        - 玻璃
2. Metal        - 金属
3. Fiber        - 纤维
4. Paper        - 纸张
5. Foam         - 泡沫
6. Rubber       - 橡胶
7. Electronics  - 电子产品
8. Other        - 其他
```

详细映射规则见：`docs/MATERIAL_MAPPING_FINAL.md`

## 🔄 版本历史

- **V1-V8**: 11类体系训练版本（已归档）
- **V9+**: 9类材质体系（当前版本）

---

**最后更新**: 2026-03-06
**维护者**: YOLO26-SeaTrash Team
