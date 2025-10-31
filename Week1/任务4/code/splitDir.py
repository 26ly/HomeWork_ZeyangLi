import os
import shutil
import random
from pathlib import Path


def split_dataset(source_dir, target_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    将源数据集按比例分割为训练集、验证集和测试集

    参数:
        source_dir: 源数据目录路径
        target_dir: 目标目录路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    """

    # 检查比例总和是否为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "比例总和必须为1"

    # 创建目标目录结构
    splits = ['train', 'val', 'test']
    for split in splits:
        for class_name in os.listdir(source_dir):
            class_dir = os.path.join(target_dir, split, class_name)
            os.makedirs(class_dir, exist_ok=True)

    # 处理每个类别的图片
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)

        # 跳过非目录文件
        if not os.path.isdir(class_path):
            continue

        print(f"处理类别: {class_name}")

        # 获取该类别所有图片文件
        image_files = []
        for file in os.listdir(class_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_files.append(file)

        # 随机打乱图片列表
        random.shuffle(image_files)
        total_images = len(image_files)

        # 计算各集合的数量
        train_count = int(total_images * train_ratio)
        val_count = int(total_images * val_ratio)
        test_count = total_images - train_count - val_count

        print(f"  总图片数: {total_images}")
        print(f"  训练集: {train_count}, 验证集: {val_count}, 测试集: {test_count}")

        # 分割图片
        train_files = image_files[:train_count]
        val_files = image_files[train_count:train_count + val_count]
        test_files = image_files[train_count + val_count:]

        # 复制文件到对应目录
        for file in train_files:
            src = os.path.join(class_path, file)
            dst = os.path.join(target_dir, 'train', class_name, file)
            shutil.copy2(src, dst)

        for file in val_files:
            src = os.path.join(class_path, file)
            dst = os.path.join(target_dir, 'val', class_name, file)
            shutil.copy2(src, dst)

        for file in test_files:
            src = os.path.join(class_path, file)
            dst = os.path.join(target_dir, 'test', class_name, file)
            shutil.copy2(src, dst)


def main():
    # 配置路径
    source_directory = "C:\\Users\\17765\\Downloads\\人工智能26赛季招新考核\\具身智能方向考核题\\题目4数据集\\armor_8c_new"
    target_directory = "C:\\Users\\17765\\mmpretrain\\projects\\RoboM\\DataSet"

    # 检查源目录是否存在
    if not os.path.exists(source_directory):
        print(f"错误: 源目录 '{source_directory}' 不存在!")
        return

    # 创建目标目录
    os.makedirs(target_directory, exist_ok=True)

    # 设置随机种子以确保可重复性
    random.seed(42)

    # 执行数据集分割
    print("开始分割数据集...")
    split_dataset(source_directory, target_directory)
    print("数据集分割完成!")

    # 显示最终目录结构
    print("\n生成的目录结构:")
    for root, dirs, files in os.walk(target_directory):
        level = root.replace(target_directory, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # 只显示前5个文件
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... 还有 {len(files) - 5} 个文件")


if __name__ == "__main__":
    main()