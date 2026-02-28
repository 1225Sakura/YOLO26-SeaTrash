#!/usr/bin/env python3
"""
分析所有海洋垃圾数据集的类别和样本分布
"""
import os
from collections import defaultdict
from pathlib import Path
import yaml

# 数据集路径
BASE_PATH = "/mnt/home/数据集/海洋垃圾所有数据集"

datasets = {
    "AquaTrash_YOLO": {
        "path": f"{BASE_PATH}/AquaTrash_YOLO",
        "has_yaml": False,
        "structure": "flat"  # Images/ 和 labels/ 在同一层
    },
    "Deep-sea Debris Detection Dataset_YOLO": {
        "path": f"{BASE_PATH}/Deep-sea Debris Detection Dataset_YOLO",
        "has_yaml": True,
        "structure": "split"  # train/val/test分开
    },
    "Sea Trash 1_YOLO": {
        "path": f"{BASE_PATH}/Sea Trash 1_YOLO",
        "has_yaml": True,
        "structure": "split"
    },
    "Test trash_YOLO": {
        "path": f"{BASE_PATH}/Test trash_YOLO",
        "has_yaml": True,
        "structure": "split"
    },
    "TrashCan 1.0_YOLO": {
        "path": f"{BASE_PATH}/TrashCan 1.0_YOLO",
        "has_yaml": False,
        "has_classes_txt": True,
        "structure": "flat"
    },
    "trash sea_YOLO": {
        "path": f"{BASE_PATH}/trash sea_YOLO",
        "has_yaml": True,
        "structure": "split"
    },
    "trash_ICRA19_YOLO": {
        "path": f"{BASE_PATH}/trash_ICRA19_YOLO",
        "has_yaml": False,
        "structure": "split"  # images/ 和 labels/ 下有train/val/test
    }
}

def read_yaml_classes(yaml_path):
    """读取data.yaml中的类别"""
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return data.get('names', [])
    except:
        return []

def read_classes_txt(txt_path):
    """读取classes.txt中的类别"""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except:
        return []

def count_labels_in_dir(label_dir):
    """统计标签目录中每个类别的数量"""
    class_counts = defaultdict(int)
    total_files = 0

    if not os.path.exists(label_dir):
        return class_counts, total_files

    for label_file in Path(label_dir).rglob('*.txt'):
        total_files += 1
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
        except:
            continue

    return class_counts, total_files

def analyze_dataset(name, info):
    """分析单个数据集"""
    print(f"\n{'='*80}")
    print(f"数据集: {name}")
    print(f"{'='*80}")

    dataset_path = info['path']

    # 读取类别名称
    classes = []
    if info.get('has_yaml'):
        yaml_path = os.path.join(dataset_path, 'data.yaml')
        classes = read_yaml_classes(yaml_path)
    elif info.get('has_classes_txt'):
        txt_path = os.path.join(dataset_path, 'classes.txt')
        classes = read_classes_txt(txt_path)

    print(f"类别数: {len(classes)}")
    if classes:
        print(f"类别列表:")
        for i, cls in enumerate(classes):
            print(f"  {i}: {cls}")

    # 统计样本数量
    total_class_counts = defaultdict(int)
    total_images = 0

    if info['structure'] == 'flat':
        # 扁平结构：Images/ 和 labels/ 在同一层
        if 'Images' in os.listdir(dataset_path):
            label_dir = os.path.join(dataset_path, 'labels')
        else:
            label_dir = os.path.join(dataset_path, 'labels')

        class_counts, num_files = count_labels_in_dir(label_dir)
        total_images += num_files
        for cls_id, count in class_counts.items():
            total_class_counts[cls_id] += count

    else:  # split structure
        # 分割结构：train/val/test
        for split in ['train', 'val', 'valid', 'test']:
            label_dir = os.path.join(dataset_path, split, 'labels')
            if not os.path.exists(label_dir):
                # 尝试其他可能的路径
                label_dir = os.path.join(dataset_path, 'labels', split)

            if os.path.exists(label_dir):
                class_counts, num_files = count_labels_in_dir(label_dir)
                total_images += num_files
                for cls_id, count in class_counts.items():
                    total_class_counts[cls_id] += count
                print(f"\n{split}集: {num_files} 张图像")

    print(f"\n总图像数: {total_images}")
    print(f"\n每类样本数:")
    for cls_id in sorted(total_class_counts.keys()):
        cls_name = classes[cls_id] if cls_id < len(classes) else f"class_{cls_id}"
        count = total_class_counts[cls_id]
        print(f"  {cls_name}: {count}")

    return classes, total_class_counts, total_images

# 主程序
print("="*80)
print("海洋垃圾数据集分析")
print("="*80)

all_classes = set()
dataset_stats = {}

for name, info in datasets.items():
    try:
        classes, class_counts, total_images = analyze_dataset(name, info)
        dataset_stats[name] = {
            'classes': classes,
            'class_counts': class_counts,
            'total_images': total_images
        }
        all_classes.update(classes)
    except Exception as e:
        print(f"\n错误: 分析 {name} 时出错: {e}")

# 汇总所有类别
print(f"\n\n{'='*80}")
print("所有数据集的类别汇总")
print(f"{'='*80}")
print(f"\n总共发现 {len(all_classes)} 个不同的类别:")
for i, cls in enumerate(sorted(all_classes), 1):
    print(f"{i:3d}. {cls}")

# 统计总样本数
total_samples = sum(sum(stats['class_counts'].values())
                   for stats in dataset_stats.values())
total_images_all = sum(stats['total_images']
                      for stats in dataset_stats.values())

print(f"\n总图像数: {total_images_all}")
print(f"总标注框数: {total_samples}")

print(f"\n{'='*80}")
print("分析完成！")
print(f"{'='*80}")
