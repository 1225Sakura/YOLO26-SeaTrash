#!/usr/bin/env python3
"""
Material-Based 9-Class Trash Dataset Integration Script - v1.0

整合12个数据集（8海洋+4陆地）到统一的9类材质体系
- 海洋数据: 8个数据集
- 陆地数据: 4个数据集
- 排除: ocean debris detection (类别错误), garbage_best (99.99%映射到Other)

9类材质体系:
0: Plastic, 1: Glass, 2: Metal, 3: Fiber, 4: Paper,
5: Foam, 6: Rubber, 7: Electronics, 8: Other

排除26个类别:
- 8个非垃圾类别 (animal_*, plant, rov)
- 7个高优先级可疑类别 (unknow, garbage, litter等, 共35,455实例)
- 11个无效类别 (ocean debris detection的错误类别)
"""

import os
import shutil
import yaml
import json
from pathlib import Path
from collections import defaultdict
import hashlib
from tqdm import tqdm
import random

# 项目根目录
PROJECT_ROOT = Path("/home/roots/YOLO26-SeaTrash")
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets" / "未"
OUTPUT_DIR = PROJECT_ROOT / "data" / "material_trash_9class_v1"

# 9类材质体系
UNIFIED_CLASSES = {
    0: "Plastic",
    1: "Glass",
    2: "Metal",
    3: "Fiber",
    4: "Paper",
    5: "Foam",
    6: "Rubber",
    7: "Electronics",
    8: "Other"
}

# 完整类别映射规则（基于MATERIAL_MAPPING_FINAL.md）
CLASS_MAPPING = {
    # === 0. Plastic（塑料）- 19个类别 ===
    "plastic": 0, "Plastic": 0, "PLASTIC": 0,
    "plastic bottle": 0, "pbottle": 0, "PET_Bottle": 0,
    "plastic bag": 0, "pbag": 0,
    "trash_bag": 0, "trash_bottle": 0, "trash_container": 0,
    "trash_cup": 0, "trash_snack_wrapper": 0,
    "Plastic_Buoy": 0, "Plastic_Buoy_China": 0, "Plastic_ETC": 0,
    "Mask": 0,  # 口罩（聚丙烯无纺布）
    "trash_tarp": 0,  # 防水布

    # === 1. Glass（玻璃）- 3个类别 ===
    "Glass": 1, "GLASS": 1, "gbottle": 1,

    # === 2. Metal（金属）- 5个类别 ===
    "Metal": 2, "METAL": 2, "metal": 2,
    "can": 2, "trash_can": 2,

    # === 3. Fiber（纤维）- 9个类别 ===
    "Cloth": 3, "trash_clothing": 3, "glove": 3,
    "Net": 3, "net": 3, "trash_net": 3,
    "Rope": 3, "trash_rope": 3, "Fishing_net_Rope": 3,

    # === 4. Paper（纸张）- 5个类别 ===
    "Paper": 4, "PAPER": 4, "paper": 4,
    "Cardboard": 4, "CARDBOARD": 4,

    # === 5. Foam（泡沫）- 4个类别 ===
    "foam": 5, "Styrofoam_Box": 5, "Styrofoam_Buoy": 5, "Styrofoam_Piece": 5,

    # === 6. Rubber（橡胶）- 2个类别 ===
    "Rubber": 6, "tire": 6,

    # === 7. Electronics（电子产品）- 2个类别 ===
    "electronics": 7, "cellphone": 7,

    # === 8. Other（其他）- 13个类别 ===
    "misc": 8, "Natural_debris": 8, "trash_branch": 8,
    "trash_pipe": 8, "trash_wreckage": 8, "rod": 8,
    "sunglasses": 8, "garbage_bag": 8, "sampah-detection": 8,
    "BIODEGRADABLE": 8, "0": 8, "c": 8, "trash": 8,

    # === 排除类别（-1表示跳过）===
    # 非垃圾类别（8个）
    "animal_crab": -1, "animal_eel": -1, "animal_etc": -1,
    "animal_fish": -1, "animal_shells": -1, "animal_starfish": -1,
    "plant": -1, "rov": -1,

    # 高优先级可疑类别（7个，共35,455实例）
    "unknow": -1, "trash_unknown_instance": -1, "garbage": -1,
    "litter": -1, "other-unknown": -1, "Waste": -1, "Buoy": -1,

    # 无效类别（11个 - ocean debris detection的错误类别）
    "A plastic bottle floating in the ocean": -1,
    "A plastic bag caught on coral": -1,
    "A discarded fishing net tangled in rocks": -1,
    "A glass bottle on the seabed": -1,
    "A metal can resting on sand": -1,
    "A piece of foam floating near the surface": -1,
    "A rubber tire half-buried in sand": -1,
    "A cloth item drifting in the water": -1,
    "A paper wrapper stuck to seaweed": -1,
    "An electronic device on the ocean floor": -1,
    "Unidentified debris in murky water": -1,
}

# 数据集配置（12个数据集）
MARINE_DATASETS = [
    {
        "name": "ocean_waste_v2",
        "path": DATASETS_DIR / "ocean waste.v2i.yolo26",
        "classes": None,
        "type": "bbox"
    },
    {
        "name": "TrashCan to Yolo.v2",
        "path": DATASETS_DIR / "TrashCan to Yolo.v2-test-version-full-.yolo26",
        "classes": None,
        "type": "segmentation"
    },
    {
        "name": "trash sea.v10i",
        "path": DATASETS_DIR / "trash sea.v10i.yolo26",
        "classes": None,
        "type": "bbox"
    },
    {
        "name": "Sea Trash 1-.v1i",
        "path": DATASETS_DIR / "Sea Trash 1-.v1i.yolo26",
        "classes": None,
        "type": "bbox"
    },
    {
        "name": "Neural_Ocean.v3i",
        "path": DATASETS_DIR / "Neural_Ocean.v3i.yolo26",
        "classes": None,
        "type": "bbox"
    },
    {
        "name": "marine-debris-yolo.v1i",
        "path": DATASETS_DIR / "marine-debris-yolo.v1i.yolo26",
        "classes": None,
        "type": "bbox"
    },
    {
        "name": "Deep-sea Debris.v1i",
        "path": DATASETS_DIR / "Deep-sea Debris Detection Dataset.v1i.yolo26",
        "classes": None,
        "type": "bbox"
    },
    {
        "name": "Trash_dataset_ICRA19",
        "path": DATASETS_DIR / "Trash_dataset_ICRA19" / "trash_ICRA19" / "dataset",
        "classes": ["plastic", "bio", "rov"],
        "type": "bbox",
        "structure": "flat"  # 图片和标注在同一目录
    }
]

LAND_DATASETS = [
    {
        "name": "GARBAGE_CLASSIFICATION",
        "path": DATASETS_DIR / "未整合垃圾识别数据集" / "GARBAGE_CLASSIFICATION",
        "classes": ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"],
        "type": "bbox"
    },
    {
        "name": "Trash_Detection",
        "path": DATASETS_DIR / "未整合垃圾识别数据集" / "Trash_Detection",
        "classes": ["Glass", "Metal", "Paper", "Plastic", "Waste"],
        "type": "bbox"
    },
    {
        "name": "EE297_Project",
        "path": DATASETS_DIR / "未整合垃圾识别数据集" / "EE297_Project",
        "classes": ["Cardboard", "Glass", "Metal", "Paper", "Plastic"],
        "type": "bbox"
    },
    {
        "name": "Litter Street Images",
        "path": DATASETS_DIR / "未整合垃圾识别数据集" / "Litter Street Images",
        "classes": ["litter", "other-unknown"],
        "type": "bbox"
    }
]


def load_dataset_classes(dataset_path):
    """从data.yaml加载数据集类别"""
    yaml_path = dataset_path / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            return data.get('names', [])
    return None


def calculate_md5(file_path):
    """计算文件MD5（用于去重）"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def convert_segmentation_to_bbox(coords):
    """
    将分割标注转换为边界框（取外接矩形）
    coords: [x1, y1, x2, y2, x3, y3, ...]
    返回: [x_center, y_center, width, height] (YOLO格式)
    """
    x_coords = coords[0::2]
    y_coords = coords[1::2]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return [x_center, y_center, width, height]


def validate_bbox(bbox, min_area=100, max_aspect_ratio=20):
    """
    验证边界框有效性
    bbox: [x_center, y_center, width, height] (归一化坐标 0-1)
    返回: (is_valid, reason)
    """
    x, y, w, h = bbox

    # 检查坐标范围
    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
        return False, "coordinates out of range"

    # 检查最小面积（假设图片尺寸为640x640）
    area_pixels = w * h * 640 * 640
    if area_pixels < min_area:
        return False, f"area too small ({area_pixels:.0f} < {min_area})"

    # 检查宽高比
    if w > 0 and h > 0:
        aspect_ratio = max(w/h, h/w)
        if aspect_ratio > max_aspect_ratio:
            return False, f"aspect ratio too extreme ({aspect_ratio:.1f} > {max_aspect_ratio})"

    return True, "valid"



def map_label(label_path, class_names, annotation_type, error_log):
    """
    映射标注文件到新类别体系
    返回: [(new_class_id, x_center, y_center, width, height), ...], exclusion_stats
    """
    mapped_labels = []
    exclusion_stats = {
        "non_trash": 0,  # 非垃圾类别
        "suspicious": 0,  # 可疑类别
        "invalid": 0,  # 无效类别
        "invalid_bbox": 0  # 无效边界框
    }

    with open(label_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) < 5:
                error_log.append(f"{label_path}:{line_num} - Invalid format (< 5 values)")
                continue

            old_class_id = int(parts[0])
            if old_class_id >= len(class_names):
                error_log.append(f"{label_path}:{line_num} - Class ID {old_class_id} out of range")
                continue

            old_class_name = class_names[old_class_id]

            # 映射到新类别
            if old_class_name not in CLASS_MAPPING:
                error_log.append(f"{label_path}:{line_num} - Unknown class '{old_class_name}', mapping to Other")
                new_class_id = 8  # Other
            else:
                new_class_id = CLASS_MAPPING[old_class_name]

            # 跳过排除的类别
            if new_class_id == -1:
                # 统计排除原因
                if old_class_name in ["animal_crab", "animal_eel", "animal_etc", "animal_fish",
                                       "animal_shells", "animal_starfish", "plant", "rov"]:
                    exclusion_stats["non_trash"] += 1
                elif old_class_name in ["unknow", "trash_unknown_instance", "garbage",
                                         "litter", "other-unknown", "Waste", "Buoy"]:
                    exclusion_stats["suspicious"] += 1
                else:
                    exclusion_stats["invalid"] += 1
                continue

            # 处理坐标
            if annotation_type == "segmentation" and len(parts) > 5:
                # 分割标注转边界框
                coords = [float(x) for x in parts[1:]]
                bbox = convert_segmentation_to_bbox(coords)
            else:
                # 边界框标注
                bbox = [float(x) for x in parts[1:5]]

            # 验证边界框
            is_valid, reason = validate_bbox(bbox)
            if not is_valid:
                error_log.append(f"{label_path}:{line_num} - Invalid bbox: {reason}")
                exclusion_stats["invalid_bbox"] += 1
                continue

            mapped_labels.append((new_class_id, *bbox))

    return mapped_labels, exclusion_stats


def find_images_and_labels(dataset_path):
    """
    查找数据集中的图片和标注文件（支持多种目录结构）
    返回: [(image_path, label_path, split), ...]
    """
    pairs = []
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp'}

    # 结构1: images/train, images/valid, images/test
    for split in ['train', 'valid', 'test']:
        img_dir = dataset_path / "images" / split
        label_dir = dataset_path / "labels" / split

        if img_dir.exists() and label_dir.exists():
            for img_path in img_dir.iterdir():
                if img_path.suffix.lower() in img_exts:
                    label_path = label_dir / f"{img_path.stem}.txt"
                    if label_path.exists():
                        pairs.append((img_path, label_path, split))

    # 结构2: train/images, train/labels
    if not pairs:
        for split in ['train', 'valid', 'test']:
            img_dir = dataset_path / split / "images"
            label_dir = dataset_path / split / "labels"

            if img_dir.exists() and label_dir.exists():
                for img_path in img_dir.iterdir():
                    if img_path.suffix.lower() in img_exts:
                        label_path = label_dir / f"{img_path.stem}.txt"
                        if label_path.exists():
                            pairs.append((img_path, label_path, split))

    # 结构3: flat结构（图片和标注在同一目录）
    if not pairs:
        for split in ['train', 'valid', 'test', 'val']:
            split_dir = dataset_path / split

            if split_dir.exists() and split_dir.is_dir():
                for img_path in split_dir.iterdir():
                    if img_path.suffix.lower() in img_exts:
                        label_path = split_dir / f"{img_path.stem}.txt"
                        if label_path.exists():
                            split_name = 'valid' if split == 'val' else split
                            pairs.append((img_path, label_path, split_name))

    # 结构4: 完全flat（所有文件在images/labels目录）
    if not pairs:
        img_dir = dataset_path / "images"
        label_dir = dataset_path / "labels"

        if img_dir.exists() and label_dir.exists():
            for img_path in img_dir.iterdir():
                if img_path.suffix.lower() in img_exts:
                    label_path = label_dir / f"{img_path.stem}.txt"
                    if label_path.exists():
                        pairs.append((img_path, label_path, "train"))

    return pairs


def integrate_dataset(dataset_config, output_dir, stats, md5_set, error_log):
    """整合单个数据集"""
    dataset_name = dataset_config["name"]
    dataset_path = dataset_config["path"]
    annotation_type = dataset_config["type"]

    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}")
    print(f"Path: {dataset_path}")
    print(f"Type: {annotation_type}")

    # 加载类别
    class_names = dataset_config["classes"]
    if class_names is None:
        class_names = load_dataset_classes(dataset_path)
        if class_names is None:
            error_log.append(f"ERROR: Cannot load classes for {dataset_name}")
            print(f"Error: Cannot load classes for {dataset_name}")
            return

    print(f"Classes: {len(class_names)} - {class_names[:5]}...")

    # 查找图片和标注
    pairs = find_images_and_labels(dataset_path)
    print(f"Found: {len(pairs)} image-label pairs")

    if not pairs:
        error_log.append(f"WARNING: No valid pairs found in {dataset_name}")
        print(f"Warning: No valid pairs found in {dataset_name}")
        return

    # 处理每个图片-标注对
    processed = 0
    skipped_duplicate = 0
    skipped_empty = 0
    dataset_exclusion_stats = defaultdict(int)

    for img_path, label_path, split in tqdm(pairs, desc=f"Integrating {dataset_name}"):
        # 检查重复（基于MD5）
        img_md5 = calculate_md5(img_path)
        if img_md5 in md5_set:
            skipped_duplicate += 1
            continue
        md5_set.add(img_md5)

        # 映射标注
        mapped_labels, exclusion_stats = map_label(label_path, class_names, annotation_type, error_log)

        # 累计排除统计
        for key, value in exclusion_stats.items():
            dataset_exclusion_stats[key] += value
            stats["exclusion_stats"][key] += value

        if not mapped_labels:
            skipped_empty += 1
            continue

        # 生成新文件名（避免冲突）
        new_name = f"{dataset_name}_{img_path.stem}{img_path.suffix}"

        # 复制图片
        output_img_dir = output_dir / "images" / split
        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_img_path = output_img_dir / new_name
        shutil.copy2(img_path, output_img_path)

        # 写入新标注
        output_label_dir = output_dir / "labels" / split
        output_label_dir.mkdir(parents=True, exist_ok=True)
        output_label_path = output_label_dir / f"{Path(new_name).stem}.txt"

        with open(output_label_path, 'w') as f:
            for label in mapped_labels:
                class_id, x, y, w, h = label
                f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

                # 统计
                stats["class_distribution"][class_id] += 1

        processed += 1
        stats["dataset_sources"][dataset_name] += 1
        stats["split_distribution"][split] += 1

    print(f"Processed: {processed}")
    print(f"Skipped (duplicate): {skipped_duplicate}")
    print(f"Skipped (empty): {skipped_empty}")
    print(f"Excluded (non-trash): {dataset_exclusion_stats['non_trash']}")
    print(f"Excluded (suspicious): {dataset_exclusion_stats['suspicious']}")
    print(f"Excluded (invalid): {dataset_exclusion_stats['invalid']}")
    print(f"Excluded (invalid_bbox): {dataset_exclusion_stats['invalid_bbox']}")

    stats["total_images"] += processed


def rebalance_splits(output_dir, target_ratios=(0.7, 0.2, 0.1)):
    """
    重新平衡train/valid/test划分（完全随机）
    target_ratios: (train, valid, test)
    """
    print(f"\n{'='*60}")
    print("Rebalancing splits...")
    print(f"Target ratios: Train={target_ratios[0]:.0%}, Valid={target_ratios[1]:.0%}, Test={target_ratios[2]:.0%}")

    # 收集所有图片
    all_images = []
    for split in ['train', 'valid', 'test']:
        img_dir = output_dir / "images" / split
        if img_dir.exists():
            for img_path in img_dir.iterdir():
                if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                    label_path = output_dir / "labels" / split / f"{img_path.stem}.txt"
                    if label_path.exists():
                        all_images.append((img_path, label_path))

    print(f"Total images: {len(all_images)}")

    # 完全随机打乱
    random.shuffle(all_images)

    # 计算划分点
    n_total = len(all_images)
    n_train = int(n_total * target_ratios[0])
    n_valid = int(n_total * target_ratios[1])

    splits = {
        'train': all_images[:n_train],
        'valid': all_images[n_train:n_train+n_valid],
        'test': all_images[n_train+n_valid:]
    }

    # 创建临时目录
    temp_dir = output_dir / "temp_rebalance"
    temp_dir.mkdir(exist_ok=True)

    # 移动所有文件到临时目录
    print("Moving files to temporary directory...")
    temp_images = []
    for img_path, label_path in tqdm(all_images, desc="Moving to temp"):
        if not img_path.exists() or not label_path.exists():
            continue

        temp_img = temp_dir / img_path.name
        temp_label = temp_dir / label_path.name

        try:
            shutil.move(str(img_path), str(temp_img))
            shutil.move(str(label_path), str(temp_label))
            temp_images.append((temp_img, temp_label))
        except Exception as e:
            print(f"Warning: Failed to move {img_path.name}: {e}")
            continue

    # 清空原目录
    for split in ['train', 'valid', 'test']:
        for subdir in ['images', 'labels']:
            split_dir = output_dir / subdir / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
            split_dir.mkdir(parents=True, exist_ok=True)

    # 重新分配
    idx = 0
    for split, pairs in splits.items():
        print(f"{split}: {len(pairs)} images")
        for _ in range(len(pairs)):
            if idx >= len(temp_images):
                break

            temp_img, temp_label = temp_images[idx]
            idx += 1

            if not temp_img.exists() or not temp_label.exists():
                continue

            # 移动图片
            new_img_path = output_dir / "images" / split / temp_img.name
            try:
                shutil.move(str(temp_img), str(new_img_path))
            except Exception as e:
                print(f"Warning: Failed to move image {temp_img.name}: {e}")
                continue

            # 移动标注
            new_label_path = output_dir / "labels" / split / temp_label.name
            try:
                shutil.move(str(temp_label), str(new_label_path))
            except Exception as e:
                print(f"Warning: Failed to move label {temp_label.name}: {e}")
                continue

    # 删除临时目录
    shutil.rmtree(temp_dir)


def generate_data_yaml(output_dir, stats):
    """生成data.yaml配置文件"""
    data_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/valid',
        'test': 'images/test',
        'nc': len(UNIFIED_CLASSES),
        'names': list(UNIFIED_CLASSES.values())
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"\nGenerated: {yaml_path}")


def save_statistics(output_dir, stats):
    """保存统计信息"""
    stats_path = output_dir / "statistics.json"

    # 转换为可序列化格式
    stats_serializable = {
        "total_images": stats["total_images"],
        "class_distribution": {UNIFIED_CLASSES[k]: v for k, v in stats["class_distribution"].items()},
        "dataset_sources": dict(stats["dataset_sources"]),
        "split_distribution": dict(stats["split_distribution"]),
        "exclusion_stats": dict(stats["exclusion_stats"])
    }

    with open(stats_path, 'w') as f:
        json.dump(stats_serializable, f, indent=2)

    print(f"Saved statistics: {stats_path}")

    # 打印统计摘要
    print(f"\n{'='*60}")
    print("INTEGRATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total images: {stats['total_images']}")

    print(f"\nClass distribution:")
    total_instances = sum(stats["class_distribution"].values())
    for class_id, count in sorted(stats["class_distribution"].items()):
        class_name = UNIFIED_CLASSES[class_id]
        percentage = count / total_instances * 100 if total_instances > 0 else 0
        print(f"  {class_id} {class_name:15s}: {count:8d} ({percentage:5.2f}%)")

    print(f"\nDataset sources:")
    for dataset, count in sorted(stats["dataset_sources"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {dataset:30s}: {count:6d}")

    print(f"\nSplit distribution:")
    for split, count in stats["split_distribution"].items():
        percentage = count / stats["total_images"] * 100 if stats["total_images"] > 0 else 0
        print(f"  {split:6s}: {count:6d} ({percentage:5.2f}%)")

    print(f"\nExclusion statistics:")
    for reason, count in stats["exclusion_stats"].items():
        print(f"  {reason:15s}: {count:6d}")


def main():
    """主函数"""
    print("="*60)
    print("Material-Based 9-Class Trash Dataset Integration - v1.0")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR}")

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 初始化统计
    stats = {
        "total_images": 0,
        "class_distribution": defaultdict(int),
        "dataset_sources": defaultdict(int),
        "split_distribution": defaultdict(int),
        "exclusion_stats": defaultdict(int)
    }

    # MD5去重集合
    md5_set = set()

    # 错误日志
    error_log = []

    # 整合海洋数据集
    print(f"\n{'#'*60}")
    print("# MARINE DATASETS (8)")
    print(f"{'#'*60}")
    for dataset in MARINE_DATASETS:
        integrate_dataset(dataset, OUTPUT_DIR, stats, md5_set, error_log)

    # 整合陆地数据集
    print(f"\n{'#'*60}")
    print("# LAND DATASETS (4)")
    print(f"{'#'*60}")
    for dataset in LAND_DATASETS:
        integrate_dataset(dataset, OUTPUT_DIR, stats, md5_set, error_log)

    # 重新平衡划分（70/20/10）
    rebalance_splits(OUTPUT_DIR, target_ratios=(0.7, 0.2, 0.1))

    # 生成配置文件
    generate_data_yaml(OUTPUT_DIR, stats)

    # 保存统计信息
    save_statistics(OUTPUT_DIR, stats)

    # 保存错误日志
    if error_log:
        error_log_path = OUTPUT_DIR / "integration_errors.log"
        with open(error_log_path, 'w') as f:
            f.write("\n".join(error_log))
        print(f"\nError log saved: {error_log_path} ({len(error_log)} entries)")

    print(f"\n{'='*60}")
    print("INTEGRATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Total images: {stats['total_images']}")
    print(f"Classes: {len(UNIFIED_CLASSES)}")
    print(f"Excluded instances: {sum(stats['exclusion_stats'].values())}")


if __name__ == "__main__":
    main()
