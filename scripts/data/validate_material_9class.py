#!/usr/bin/env python3
"""
9类材质数据集验证工具
适配 images/{split}/ 目录结构
"""

import os
import hashlib
from pathlib import Path
from collections import defaultdict
import json

class Material9ClassValidator:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.num_classes = 9  # 0-8

        # 统计信息
        self.stats = {
            'total_images': 0,
            'total_labels': 0,
            'total_instances': 0,
            'duplicates': [],
            'invalid_format': [],
            'invalid_class_id': [],
            'invalid_bbox': [],
            'missing_labels': [],
            'missing_images': [],
            'class_distribution': defaultdict(int),
            'split_stats': {}
        }

    def calculate_md5(self, file_path):
        """计算文件MD5"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def find_duplicates(self, split='train'):
        """检测重复图片"""
        print(f"\n{'='*60}")
        print(f"检测 {split} 集重复图片...")
        print(f"{'='*60}")

        images_dir = self.dataset_path / 'images' / split
        if not images_dir.exists():
            print(f"⚠️  目录不存在: {images_dir}")
            return

        hash_to_files = defaultdict(list)
        image_files = list(images_dir.glob('*.jpg')) + \
                     list(images_dir.glob('*.png')) + \
                     list(images_dir.glob('*.jpeg'))

        print(f"正在计算 {len(image_files)} 张图片的MD5...")

        for i, img_path in enumerate(image_files, 1):
            if i % 5000 == 0:
                print(f"  进度: {i}/{len(image_files)}")

            md5_hash = self.calculate_md5(img_path)
            hash_to_files[md5_hash].append(img_path)

        # 找出重复项
        duplicates = {h: files for h, files in hash_to_files.items() if len(files) > 1}

        if duplicates:
            print(f"\n❌ 发现 {len(duplicates)} 组重复图片:")
            for hash_val, files in list(duplicates.items())[:5]:  # 只显示前5组
                print(f"  MD5: {hash_val}")
                for f in files:
                    print(f"    - {f.name}")
                self.stats['duplicates'].extend([str(f) for f in files])
            if len(duplicates) > 5:
                print(f"  ... 还有 {len(duplicates)-5} 组重复")
        else:
            print(f"✅ 未发现重复图片")

        return len(image_files)

    def validate_label_format(self, label_path):
        """验证标注格式"""
        errors = []
        instances = 0

        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()

                # 检查格式（5值）
                if len(parts) != 5:
                    errors.append(f"Line {line_num}: 格式错误，应为5值，实际{len(parts)}值")
                    continue

                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # 检查类别ID范围（0-8）
                    if class_id < 0 or class_id > 8:
                        errors.append(f"Line {line_num}: 类别ID {class_id} 超出范围 [0-8]")
                        self.stats['invalid_class_id'].append(str(label_path))

                    # 检查边界框坐标（0-1范围）
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                        errors.append(f"Line {line_num}: 中心坐标超出范围 ({x_center}, {y_center})")
                        self.stats['invalid_bbox'].append(str(label_path))

                    if not (0 < width <= 1 and 0 < height <= 1):
                        errors.append(f"Line {line_num}: 尺寸超出范围 ({width}, {height})")
                        self.stats['invalid_bbox'].append(str(label_path))

                    # 统计类别分布
                    if 0 <= class_id <= 8:
                        self.stats['class_distribution'][class_id] += 1
                        instances += 1

                except ValueError as e:
                    errors.append(f"Line {line_num}: 数值解析错误 - {e}")

        except Exception as e:
            errors.append(f"文件读取错误: {e}")

        return errors, instances

    def validate_split(self, split='train'):
        """验证单个划分"""
        print(f"\n{'='*60}")
        print(f"验证 {split} 集...")
        print(f"{'='*60}")

        images_dir = self.dataset_path / 'images' / split
        labels_dir = self.dataset_path / 'labels' / split

        if not images_dir.exists():
            print(f"⚠️  图片目录不存在: {images_dir}")
            return
        if not labels_dir.exists():
            print(f"⚠️  标注目录不存在: {labels_dir}")
            return

        # 获取所有图片和标注
        image_files = {f.stem: f for f in images_dir.glob('*.jpg')}
        image_files.update({f.stem: f for f in images_dir.glob('*.png')})
        image_files.update({f.stem: f for f in images_dir.glob('*.jpeg')})

        label_files = {f.stem: f for f in labels_dir.glob('*.txt')}

        print(f"图片数: {len(image_files)}")
        print(f"标注数: {len(label_files)}")

        # 检查匹配
        image_stems = set(image_files.keys())
        label_stems = set(label_files.keys())

        missing_labels = image_stems - label_stems
        missing_images = label_stems - image_stems

        if missing_labels:
            print(f"⚠️  缺少标注: {len(missing_labels)} 个")
            self.stats['missing_labels'].extend([str(image_files[s]) for s in list(missing_labels)[:5]])

        if missing_images:
            print(f"⚠️  缺少图片: {len(missing_images)} 个")
            self.stats['missing_images'].extend([str(label_files[s]) for s in list(missing_images)[:5]])

        # 验证标注格式
        print(f"\n验证标注格式...")
        valid_count = 0
        invalid_count = 0
        total_instances = 0

        for stem in label_stems:
            if stem not in image_stems:
                continue

            label_path = label_files[stem]
            errors, instances = self.validate_label_format(label_path)
            total_instances += instances

            if errors:
                invalid_count += 1
                self.stats['invalid_format'].append({
                    'file': str(label_path),
                    'errors': errors[:3]  # 只保留前3个错误
                })
            else:
                valid_count += 1

        print(f"✅ 有效标注: {valid_count}")
        print(f"❌ 无效标注: {invalid_count}")
        print(f"📊 总实例数: {total_instances}")

        # 保存划分统计
        self.stats['split_stats'][split] = {
            'images': len(image_files),
            'labels': len(label_files),
            'instances': total_instances,
            'missing_labels': len(missing_labels),
            'missing_images': len(missing_images),
            'invalid_labels': invalid_count
        }

        self.stats['total_images'] += len(image_files)
        self.stats['total_labels'] += len(label_files)
        self.stats['total_instances'] += total_instances

    def generate_report(self):
        """生成验证报告"""
        print(f"\n{'='*60}")
        print("生成验证报告...")
        print(f"{'='*60}")

        report_path = self.dataset_path / 'validation_report_v11.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("9类材质数据集验证报告\n")
            f.write("="*60 + "\n\n")

            # 总体统计
            f.write("## 总体统计\n\n")
            f.write(f"总图片数: {self.stats['total_images']}\n")
            f.write(f"总标注数: {self.stats['total_labels']}\n")
            f.write(f"总实例数: {self.stats['total_instances']}\n\n")

            # 划分统计
            f.write("## 划分统计\n\n")
            for split, stats in self.stats['split_stats'].items():
                f.write(f"{split}:\n")
                f.write(f"  图片: {stats['images']}\n")
                f.write(f"  标注: {stats['labels']}\n")
                f.write(f"  实例: {stats['instances']}\n")
                f.write(f"  缺少标注: {stats['missing_labels']}\n")
                f.write(f"  缺少图片: {stats['missing_images']}\n")
                f.write(f"  无效标注: {stats['invalid_labels']}\n\n")

            # 类别分布
            f.write("## 类别分布（9类）\n\n")
            class_names = ['Plastic', 'Glass', 'Metal', 'Paper', 'Fiber',
                          'Foam', 'Rubber', 'Electronics', 'Other']
            for class_id in range(9):
                count = self.stats['class_distribution'][class_id]
                percentage = count / self.stats['total_instances'] * 100 if self.stats['total_instances'] > 0 else 0
                f.write(f"{class_id}: {class_names[class_id]:12s} - {count:6d} ({percentage:5.2f}%)\n")
            f.write("\n")

            # 问题统计
            f.write("## 问题统计\n\n")
            f.write(f"重复图片: {len(self.stats['duplicates'])}\n")
            f.write(f"无效格式: {len(self.stats['invalid_format'])}\n")
            f.write(f"无效类别ID: {len(set(self.stats['invalid_class_id']))}\n")
            f.write(f"无效边界框: {len(set(self.stats['invalid_bbox']))}\n")
            f.write(f"缺少标注: {len(self.stats['missing_labels'])}\n")
            f.write(f"缺少图片: {len(self.stats['missing_images'])}\n\n")

            # 详细错误（前10个）
            if self.stats['invalid_format']:
                f.write("## 格式错误示例（前10个）\n\n")
                for item in self.stats['invalid_format'][:10]:
                    f.write(f"文件: {item['file']}\n")
                    for error in item['errors']:
                        f.write(f"  - {error}\n")
                    f.write("\n")

        print(f"✅ 报告已保存: {report_path}")

    def run(self):
        """运行完整验证"""
        print("="*60)
        print("9类材质数据集验证工具")
        print("="*60)
        print(f"数据集: {self.dataset_path}")
        print(f"类别数: 9 (0-8)")
        print("="*60)

        # 验证每个划分
        for split in ['train', 'valid', 'test']:
            # 检测重复
            self.find_duplicates(split)
            # 验证格式
            self.validate_split(split)

        # 生成报告
        self.generate_report()

        # 打印摘要
        print(f"\n{'='*60}")
        print("验证摘要")
        print(f"{'='*60}")
        print(f"✅ 总图片数: {self.stats['total_images']}")
        print(f"✅ 总实例数: {self.stats['total_instances']}")
        print(f"❌ 重复图片: {len(self.stats['duplicates'])}")
        print(f"❌ 无效标注: {len(self.stats['invalid_format'])}")
        print(f"⚠️  缺少标注: {len(self.stats['missing_labels'])}")
        print(f"⚠️  缺少图片: {len(self.stats['missing_images'])}")
        print(f"\n{'='*60}")
        print("验证完成！")
        print(f"{'='*60}")
        print(f"详细报告: {self.dataset_path}/validation_report_v11.txt")
        print(f"{'='*60}")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("用法: python validate_material_9class.py <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    validator = Material9ClassValidator(dataset_path)
    validator.run()
