#!/usr/bin/env python3
"""
处理未命名数据集：AquaTrash 和 trash_ICRA19
"""
import os
import csv
import shutil
from pathlib import Path
from PIL import Image

# 输出目录
OUTPUT_BASE = "/home/user/sea/datasets/marine_debris_unified"
os.makedirs(OUTPUT_BASE, exist_ok=True)

# ============================================================================
# 1. 处理 AquaTrash (CSV格式 -> YOLO格式)
# ============================================================================

print("="*80)
print("处理 AquaTrash 数据集")
print("="*80)

aquatrash_base = "/mnt/home/数据集/海洋垃圾所有数据集/AquaTrash"
aquatrash_yolo = "/mnt/home/数据集/海洋垃圾所有数据集/AquaTrash_YOLO"

# AquaTrash类别映射（只保留垃圾类）
aquatrash_classes = {
    'plastic': 0,
    'metal': 1,
    'paper': 2,
    'glass': 3
}

# 读取CSV标注
annotations_file = os.path.join(aquatrash_base, 'annotations.csv')
annotations = {}

with open(annotations_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        img_name = row['image_name']
        class_name = row['class_name'].lower()

        if class_name not in aquatrash_classes:
            continue  # 跳过非垃圾类

        if img_name not in annotations:
            annotations[img_name] = []

        annotations[img_name].append({
            'class': aquatrash_classes[class_name],
            'x_min': int(row['x_min']),
            'y_min': int(row['y_min']),
            'x_max': int(row['x_max']),
            'y_max': int(row['y_max'])
        })

print(f"读取到 {len(annotations)} 张图像的标注")

# 转换为YOLO格式并保存
aquatrash_images_dir = os.path.join(aquatrash_yolo, 'Images')
aquatrash_labels_dir = os.path.join(aquatrash_yolo, 'labels')

converted_count = 0
for img_name, boxes in annotations.items():
    # 读取图像获取尺寸
    img_path = os.path.join(aquatrash_images_dir, img_name)
    if not os.path.exists(img_path):
        print(f"警告: 图像不存在 {img_path}")
        continue

    try:
        img = Image.open(img_path)
        img_width, img_height = img.size

        # 转换为YOLO格式
        label_path = os.path.join(aquatrash_labels_dir,
                                  img_name.replace('.jpg', '.txt').replace('.JPG', '.txt'))

        with open(label_path, 'w') as f:
            for box in boxes:
                # 转换为YOLO格式 (class x_center y_center width height)
                x_center = ((box['x_min'] + box['x_max']) / 2) / img_width
                y_center = ((box['y_min'] + box['y_max']) / 2) / img_height
                width = (box['x_max'] - box['x_min']) / img_width
                height = (box['y_max'] - box['y_min']) / img_height

                f.write(f"{box['class']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        converted_count += 1
    except Exception as e:
        print(f"错误: 处理 {img_name} 时出错: {e}")

print(f"✓ AquaTrash: 成功转换 {converted_count} 张图像")
print(f"  类别: {list(aquatrash_classes.keys())}")

# ============================================================================
# 2. 处理 trash_ICRA19 (已是YOLO格式，但需要过滤类别)
# ============================================================================

print("\n" + "="*80)
print("处理 trash_ICRA19 数据集")
print("="*80)

trash_icra19_base = "/mnt/home/数据集/海洋垃圾所有数据集/trash_ICRA19"
trash_icra19_yolo = "/home/user/sea/datasets/trash_ICRA19_YOLO"

# trash_ICRA19类别映射
# 原始: 0=plastic, 1=bio, 2=rov
# 只保留plastic (class 0)
# 需要过滤掉bio和rov

os.makedirs(os.path.join(trash_icra19_yolo, 'images', 'train'), exist_ok=True)
os.makedirs(os.path.join(trash_icra19_yolo, 'images', 'val'), exist_ok=True)
os.makedirs(os.path.join(trash_icra19_yolo, 'images', 'test'), exist_ok=True)
os.makedirs(os.path.join(trash_icra19_yolo, 'labels', 'train'), exist_ok=True)
os.makedirs(os.path.join(trash_icra19_yolo, 'labels', 'val'), exist_ok=True)
os.makedirs(os.path.join(trash_icra19_yolo, 'labels', 'test'), exist_ok=True)

total_images = 0
total_kept = 0
total_filtered = 0

for split in ['train', 'val', 'test']:
    source_dir = os.path.join(trash_icra19_base, 'dataset', split)
    if not os.path.exists(source_dir):
        continue

    print(f"\n处理 {split} 集...")

    # 获取所有图像文件
    image_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]

    for img_file in image_files:
        total_images += 1
        label_file = img_file.replace('.jpg', '.txt')

        source_img = os.path.join(source_dir, img_file)
        source_label = os.path.join(source_dir, label_file)

        if not os.path.exists(source_label):
            continue

        # 读取标签并过滤
        filtered_lines = []
        with open(source_label, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    if class_id == 0:  # 只保留plastic (class 0)
                        filtered_lines.append(line)
                    else:
                        total_filtered += 1

        # 如果过滤后还有标注，则保存
        if filtered_lines:
            dest_img = os.path.join(trash_icra19_yolo, 'images', split, img_file)
            dest_label = os.path.join(trash_icra19_yolo, 'labels', split, label_file)

            shutil.copy(source_img, dest_img)
            with open(dest_label, 'w') as f:
                f.writelines(filtered_lines)

            total_kept += 1

print(f"\n✓ trash_ICRA19: 处理完成")
print(f"  总图像数: {total_images}")
print(f"  保留图像数: {total_kept}")
print(f"  过滤的标注框数: {total_filtered} (bio和rov)")
print(f"  只保留类别: plastic")

# ============================================================================
# 创建类别映射文件
# ============================================================================

print("\n" + "="*80)
print("创建类别映射文件")
print("="*80)

# AquaTrash类别映射
with open(os.path.join(aquatrash_yolo, 'classes.txt'), 'w') as f:
    for cls_name in sorted(aquatrash_classes.keys(), key=lambda x: aquatrash_classes[x]):
        f.write(f"{cls_name}\n")

print(f"✓ AquaTrash classes.txt 已创建")

# trash_ICRA19类别映射
with open(os.path.join(trash_icra19_yolo, 'classes.txt'), 'w') as f:
    f.write("plastic\n")

print(f"✓ trash_ICRA19 classes.txt 已创建")

print("\n" + "="*80)
print("处理完成！")
print("="*80)
print(f"\nAquaTrash_YOLO: {aquatrash_yolo}")
print(f"trash_ICRA19_YOLO: {trash_icra19_yolo}")
