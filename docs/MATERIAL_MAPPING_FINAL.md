# 材质分类映射方案 - 9类体系（最终版）

## 📋 方案概述

将所有数据集的87个原始类别统一映射到9个材质类别，基于垃圾的实际材质进行分类。

**重要更新**: 排除7个高优先级可疑类别（共35,455实例）

### 最终类别体系（9类）

```
0. Plastic      - 塑料（瓶子、袋子、容器、口罩等）
1. Glass        - 玻璃（瓶子等）
2. Metal        - 金属（罐头、金属碎片）
3. Fiber        - 纤维（布料、渔网、绳索、手套）
4. Paper        - 纸张（纸板、纸张）
5. Foam         - 泡沫（泡沫塑料、泡沫浮标）
6. Rubber       - 橡胶（轮胎等）
7. Electronics  - 电子产品（手机、电子设备）
8. Other        - 其他（材质不明、混合材质、自然物）
```

---

## 📊 统计摘要

| 材质类别 | 原始类别数 | 占比 |
|---------|-----------|------|
| Plastic | 19 | 21.8% |
| Glass | 3 | 3.4% |
| Metal | 5 | 5.7% |
| Fiber | 9 | 10.3% |
| Paper | 5 | 5.7% |
| Foam | 4 | 4.6% |
| Rubber | 2 | 2.3% |
| Electronics | 2 | 2.3% |
| Other | 13 | 14.9% |
| **排除（非垃圾）** | 8 | 9.2% |
| **排除（可疑类别）** | 7 | 8.0% |
| **无效（错误）** | 11 | 12.6% |
| **总计** | **87** | **100%** |

**有效类别**: 61个（70.1%）
**排除类别**: 26个（29.9%）

---

## 🗂️ 完整映射表

### 0️⃣ Plastic（塑料）- 19个类别

- plastic, Plastic, PLASTIC
- plastic bottle, pbottle, PET_Bottle
- plastic bag, pbag
- trash_bag, trash_bottle, trash_container, trash_cup, trash_snack_wrapper
- Plastic_Buoy, Plastic_Buoy_China, Plastic_ETC
- Mask（口罩）
- trash_tarp（防水布）

### 1️⃣ Glass（玻璃）- 3个类别

- Glass, GLASS, gbottle

### 2️⃣ Metal（金属）- 5个类别

- Metal, METAL, metal
- can, trash_can

### 3️⃣ Fiber（纤维）- 9个类别

- Cloth, trash_clothing, glove
- Net, net, trash_net
- Rope, trash_rope, Fishing_net_Rope

### 4️⃣ Paper（纸张）- 5个类别

- Paper, PAPER, paper
- Cardboard, CARDBOARD

### 5️⃣ Foam（泡沫）- 4个类别

- foam, Styrofoam_Box, Styrofoam_Buoy, Styrofoam_Piece

### 6️⃣ Rubber（橡胶）- 2个类别

- Rubber, tire

### 7️⃣ Electronics（电子产品）- 2个类别

- electronics, cellphone

### 8️⃣ Other（其他）- 13个类别

- misc（杂项）
- Natural_debris（自然碎屑）
- trash_branch（树枝）
- trash_pipe（管道）
- trash_wreckage（残骸）
- rod（杆子）
- sunglasses（太阳镜）
- garbage_bag（垃圾袋）
- sampah-detection（印尼语垃圾）
- BIODEGRADABLE（可生物降解）
- 0（垃圾堆）
- c（垃圾堆）
- trash（垃圾）

---

## 🚫 排除类别（26个）

### 非垃圾类别（8个）

- animal_crab, animal_eel, animal_etc, animal_fish
- animal_shells, animal_starfish
- plant, rov

### 高优先级可疑类别（7个，共35,455实例）

| 类别 | 数据集 | 实例数 | 原因 |
|------|--------|--------|------|
| unknow | marine-debris-yolo | 10,099 | 用户要求排除 |
| trash_unknown_instance | TrashCan/Ocean Debris | 9,371 | 用户要求排除 |
| garbage | garbage_best | 6,175 | 用户要求排除 |
| litter | Litter Street Images | 4,970 | 用户要求排除 |
| other-unknown | Litter Street Images | 3,920 | 用户要求排除 |
| Waste | Trash_Detection | 3,804 | 用户要求排除 |
| Buoy | trash sea | 2,116 | 用户要求排除 |

### 无效类别（11个）

ocean debris detection数据集的错误类别（描述文本）

---

## 🎯 关键决策

### 1. garbage_best的"0"和"c"
- 实际内容: 垃圾堆（混合材质）
- 决策: 映射到Other ✅

### 2. 高优先级可疑类别
- 影响: 排除35,455个实例
- 原因: 用户要求排除所有高优先级可疑类别
- 包括: unknow, trash_unknown_instance, garbage, litter, other-unknown, Waste, Buoy

### 3. Mask（口罩）
- 材质: 聚丙烯（PP）无纺布
- 决策: 映射到Plastic ✅

### 4. glove（手套）
- 材质: 布料
- 决策: 映射到Fiber ✅

### 5. tire（轮胎）
- 材质: 橡胶
- 决策: 映射到Rubber ✅

---

## 📁 预期数据集规模

### 整合后预期

- **总图片数**: ~76,000张（排除可疑类别后）
- **总实例数**: 预计300,000+个
- **数据划分**: 70/20/10 (Train/Valid/Test)
- **类别数**: 9类材质

### 排除影响

- **排除数据集**: ocean debris detection (5,133张)
- **排除实例**: 35,455个（高优先级可疑类别）
- **保留数据集**: 12个（8海洋 + 4陆地）

---

**生成时间**: 2025-03-03
**版本**: v3.0 (排除高优先级可疑类别)
**总类别数**: 87 → 9
**有效类别**: 61个
**排除类别**: 26个
