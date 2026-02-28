#!/usr/bin/env python3
"""
整合所有海洋垃圾数据集
- 统一类别命名
- 删除非垃圾类（动物/植物/rov）
- 清理重复标签
- 划分train/val/test集
"""
import os
import shutil
import yaml
from pathlib import Path
from collections import defaultdict
import random
from tqdm import tqdm

# 设置随机种子
random.seed(42)

# 输出目录
OUTPUT_DIR = "/home/user/sea/datasets/marine_debris_unified"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 类别映射配置
# ============================================================================

# 统一类别映射（目标类别）
UNIFIED_CLASSES = [
    "Plastic",           # 0
    "Plastic_Bottle",    # 1
    "Plastic_Buoy",      # 2
    "Metal",             # 3
    "Glass",             # 4
    "Paper",             # 5
    "Cloth",             # 6
    "Net",               # 7
    "Rope",              # 8
    "Styrofoam",         # 9
    "Buoy",              # 10
    "Rubber",            # 11
    "Natural_debris",    # 12
    "Trash_Other"        # 13
]

# 每个数据集的类别映射到统一类别
DATASET_MAPPINGS = {
    "AquaTrash_YOLO": {
        "classes": ["plastic", "metal", "paper", "glass"],
        "mapping": {
            "plastic": "Plastic",
            "metal": "Metal",
            "paper": "Paper",
            "glass": "Glass"
        }
    },
    "Deep-sea Debris Detection Dataset_YOLO": {
        "classes": ["Cloth", "Fishing_net_Rope", "Glass", "Metal", "Natural_debris", "Plastic", "Rubber"],
        "mapping": {
            "Cloth": "Cloth",
            "Fishing_net_Rope": "Net",
            "Glass": "Glass",
            "Metal": "Metal",
            "Natural_debris": "Natural_debris",
            "Plastic": "Plastic",
            "Rubber": "Rubber"
        }
    },
    "Sea Trash 1_YOLO": {
        "classes": ["Glass", "Metal", "Net", "PET_Bottle", "Plastic_Buoy", "Plastic_Buoy_China",
                   "Plastic_ETC", "Rope", "Styrofoam_Box", "Styrofoam_Buoy", "Styrofoam_Piece"],
        "mapping": {
            "Glass": "Glass",
            "Metal": "Metal",
            "Net": "Net",
            "PET_Bottle": "Plastic_Bottle",
            "Plastic_Buoy": "Plastic_Buoy",
            "Plastic_Buoy_China": "Plastic_Buoy",
            "Plastic_ETC": "Plastic",
            "Rope": "Rope",
            "Styrofoam_Box": "Styrofoam",
            "Styrofoam_Buoy": "Styrofoam",
            "Styrofoam_Piece": "Styrofoam"
        }
    },
    "Test trash_YOLO": {
        "classes": ["Glass", "Metal", "Paper", "Plastic", "PlasticBag"],
        "mapping": {
            "Glass": "Glass",
            "Metal": "Metal",
            "Paper": "Paper",
            "Plastic": "Plastic",
            "PlasticBag": "Plastic"
        }
    },
    "TrashCan 1.0_YOLO": {
        "classes": ["rov", "plant", "animal_fish", "animal_starfish", "animal_shells", "animal_crab",
                   "animal_eel", "animal_etc", "trash_clothing", "trash_pipe", "trash_bottle",
                   "trash_bag", "trash_snack_wrapper", "trash_can", "trash_cup", "trash_container",
                   "trash_unknown_instance", "trash_branch", "trash_wreckage", "trash_tarp",
                   "trash_rope", "trash_net"],
        "mapping": {
            # 跳过动物和植物类
            "rov": None,  # 删除
            "plant": None,  # 删除
            "animal_fish": None,
            "animal_starfish": None,
            "animal_shells": None,
            "animal_crab": None,
            "animal_eel": None,
            "animal_etc": None,
            # 垃圾类
            "trash_clothing": "Cloth",
            "trash_pipe": "Trash_Other",
            "trash_bottle": "Plastic_Bottle",
            "trash_bag": "Plastic",
            "trash_snack_wrapper": "Plastic",
            "trash_can": "Metal",
            "trash_cup": "Plastic",
            "trash_container": "Trash_Other",
            "trash_unknown_instance": "Trash_Other",
            "trash_branch": "Natural_debris",
            "trash_wreckage": "Trash_Other",
            "trash_tarp": "Plastic",
            "trash_rope": "Rope",
            "trash_net": "Net"
        }
    },
    "trash sea_YOLO": {
        "classes": ["Buoy", "can", "paper", "plastic bag", "plastic bottle"],
        "mapping": {
            "Buoy": "Buoy",
            "can": "Metal",
            "paper": "Paper",
            "plastic bag": "Plastic",
            "plastic bottle": "Plastic_Bottle"
        }
    },
    "trash_ICRA19_YOLO": {
        "classes": ["plastic"],  # 已经过滤掉bio和rov
        "mapping": {
            "plastic": "Plastic"
        }
    }
}

print("="*80)
print("海洋垃圾数据集整合")
print("="*80)
print(f"统一类别数: {len(UNIFIED_CLASSES)}")
print(f"统一类别: {UNIFIED_CLASSES}")
print("="*80)

# ============================================================================
# 数据集路径配置
# ============================================================================

BASE_PATH = "/mnt/home/数据集/海洋垃圾所有数据集"

DATASETS = {
    "AquaTrash_YOLO": {
        "path": f"{BASE_PATH}/AquaTrash_YOLO",
        "structure": "flat"  # Images/ 和 labels/ 在同一层
    },
    "Deep-sea Debris Detection Dataset_YOLO": {
        "path": f"{BASE_PATH}/Deep-sea Debris Detection Dataset_YOLO",
        "structure": "split"  # train/val/test分开
    },
    "Sea Trash 1_YOLO": {
        "path": f"{BASE_PATH}/Sea Trash 1_YOLO",
        "structure": "split"
    },
    "Test trash_YOLO": {
        "path": f"{BASE_PATH}/Test trash_YOLO",
        "structure": "split"
    },
    "TrashCan 1.0_YOLO": {
        "path": f"{BASE_PATH}/TrashCan 1.0_YOLO",
        "structure": "flat"
    },
    "trash sea_YOLO": {
        "path": f"{BASE_PATH}/trash sea_YOLO",
        "structure": "split"
    },
    "trash_ICRA19_YOLO": {
        "path": "/home/user/sea/datasets/trash_ICRA19_YOLO",
        "structure": "split"
    }
}

# ============================================================================
# 辅助函数
# ============================================================================

def get_unified_class_id(dataset_name, original_class_id):
    """将原始类别ID映射到统一类别ID"""
    mapping_info = DATASET_MAPPINGS[dataset_name]
    original_classes = mapping_info["classes"]
    class_mapping = mapping_info["mapping"]

    if original_class_id >= len(original_classes):
        return None  # 无效类别

    original_class_name = original_classes[original_class_id]
    unified_class_name = class_mapping.get(original_class_name)

    if unified_class_name is None:
        return None  # 需要删除的类别

    if unified_class_name not in UNIFIED_CLASSES:
        print(f"警告: 未知的统一类别 {unified_class_name}")
        return None

    return UNIFIED_CLASSES.index(unified_class_name)

def process_label_file(label_path, dataset_name):
    """处理标签文件，转换类别ID并过滤"""
    new_lines = []

    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                original_class_id = int(parts[0])
                new_class_id = get_unified_class_id(dataset_name, original_class_id)

                if new_class_id is None:
                    continue  # 跳过需要删除的类别

                # 替换类别ID
                parts[0] = str(new_class_id)
                new_lines.append(' '.join(parts) + '\n')

    except Exception as e:
        print(f"错误: 处理标签文件 {label_path} 时出错: {e}")
        return []

    return new_lines

def remove_duplicate_labels(lines):
    """移除重复的标签"""
    seen = set()
    unique_lines = []

    for line in lines:
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)

    return unique_lines

# ============================================================================
# 整合数据集
# ============================================================================

all_samples = []  # (image_path, label_lines, dataset_name)

print("\n开始整合数据集...")

for dataset_name, dataset_info in DATASETS.items():
    print(f"\n处理: {dataset_name}")
    dataset_path = dataset_info['path']

    if not os.path.exists(dataset_path):
        print(f"  警告: 数据集路径不存在，跳过")
        continue

    sample_count = 0

    if dataset_info['structure'] == 'flat':
        # 扁平结构
        if dataset_name == "AquaTrash_YOLO":
            images_dir = os.path.join(dataset_path, 'Images')
            labels_dir = os.path.join(dataset_path, 'labels')
        else:
            images_dir = os.path.join(dataset_path, 'images')
            labels_dir = os.path.join(dataset_path, 'labels')

        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"  警告: 图像或标签目录不存在")
            continue

        for label_file in Path(labels_dir).glob('*.txt'):
            img_name = label_file.stem + '.jpg'
            img_path = os.path.join(images_dir, img_name)

            # 尝试不同的扩展名
            if not os.path.exists(img_path):
                img_path = os.path.join(images_dir, label_file.stem + '.JPG')
            if not os.path.exists(img_path):
                img_path = os.path.join(images_dir, label_file.stem + '.png')

            if not os.path.exists(img_path):
                continue

            label_lines = process_label_file(str(label_file), dataset_name)
            if label_lines:
                label_lines = remove_duplicate_labels(label_lines)
                all_samples.append((img_path, label_lines, dataset_name))
                sample_count += 1

    else:  # split structure
        for split in ['train', 'val', 'valid', 'test']:
            images_dir = os.path.join(dataset_path, split, 'images')
            labels_dir = os.path.join(dataset_path, split, 'labels')

            if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
                continue

            for label_file in Path(labels_dir).glob('*.txt'):
                img_name = label_file.stem + '.jpg'
                img_path = os.path.join(images_dir, img_name)

                if not os.path.exists(img_path):
                    img_path = os.path.join(images_dir, label_file.stem + '.png')

                if not os.path.exists(img_path):
                    continue

                label_lines = process_label_file(str(label_file), dataset_name)
                if label_lines:
                    label_lines = remove_duplicate_labels(label_lines)
                    all_samples.append((img_path, label_lines, dataset_name))
                    sample_count += 1

    print(f"  ✓ 收集到 {sample_count} 个样本")

print(f"\n总样本数: {len(all_samples)}")

# ============================================================================
# 划分数据集
# ============================================================================

print("\n划分数据集...")

# 打乱样本
random.shuffle(all_samples)

# 划分比例: train 80%, val 10%, test 10%
total = len(all_samples)
train_size = int(total * 0.8)
val_size = int(total * 0.1)

train_samples = all_samples[:train_size]
val_samples = all_samples[train_size:train_size + val_size]
test_samples = all_samples[train_size + val_size:]

print(f"训练集: {len(train_samples)}")
print(f"验证集: {len(val_samples)}")
print(f"测试集: {len(test_samples)}")

# ============================================================================
# 保存整合后的数据集
# ============================================================================

print("\n保存整合后的数据集...")

for split_name, samples in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
    images_dir = os.path.join(OUTPUT_DIR, 'images', split_name)
    labels_dir = os.path.join(OUTPUT_DIR, 'labels', split_name)

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    print(f"\n保存 {split_name} 集...")

    for idx, (img_path, label_lines, dataset_name) in enumerate(tqdm(samples)):
        # 生成新的文件名（避免冲突）
        new_name = f"{split_name}_{idx:06d}"
        img_ext = Path(img_path).suffix

        dest_img = os.path.join(images_dir, new_name + img_ext)
        dest_label = os.path.join(labels_dir, new_name + '.txt')

        # 复制图像
        try:
            shutil.copy(img_path, dest_img)
        except Exception as e:
            print(f"错误: 复制图像失败 {img_path}: {e}")
            continue

        # 保存标签
        with open(dest_label, 'w') as f:
            f.writelines(label_lines)

# ============================================================================
# 创建data.yaml配置文件
# ============================================================================

print("\n创建data.yaml配置文件...")

data_yaml = {
    'path': OUTPUT_DIR,
    'train': 'images/train',
    'val': 'images/val',
    'test': 'images/test',
    'nc': len(UNIFIED_CLASSES),
    'names': UNIFIED_CLASSES
}

with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

print(f"✓ data.yaml 已创建")

# ============================================================================
# 统计信息
# ============================================================================

print("\n" + "="*80)
print("整合完成！")
print("="*80)
print(f"输出目录: {OUTPUT_DIR}")
print(f"总样本数: {len(all_samples)}")
print(f"  训练集: {len(train_samples)} ({len(train_samples)/len(all_samples)*100:.1f}%)")
print(f"  验证集: {len(val_samples)} ({len(val_samples)/len(all_samples)*100:.1f}%)")
print(f"  测试集: {len(test_samples)} ({len(test_samples)/len(all_samples)*100:.1f}%)")
print(f"统一类别数: {len(UNIFIED_CLASSES)}")
print(f"类别列表: {UNIFIED_CLASSES}")
print("="*80)
