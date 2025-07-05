import os
import numpy as np
from PIL import Image
from tqdm import tqdm  # 引入tqdm用于显示进度条
import json
import random

# 设置Hugging Face镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset, DatasetDict

# --- 1. 初始化和目录设置 ---
print("正在加载 Hugging Face 数据集 'refoundd/NailongClassification'...")
dataset = load_dataset("refoundd/NailongClassification")

output_dir = "./yolo_dataset"
train_img_dir = os.path.join(output_dir, "train", "images")
train_lbl_dir = os.path.join(output_dir, "train", "labels")
val_img_dir = os.path.join(output_dir, "val", "images")
val_lbl_dir = os.path.join(output_dir, "val", "labels")

os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(train_lbl_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

# --- 2. 类别映射与数据集分割 ---

# 从训练集中提取所有唯一的类别名称并排序，确保每次运行顺序一致
all_labels = sorted(set(item["label"] for item in dataset["train"]))
class_map = {label: idx for idx, label in enumerate(all_labels)}

# 保存类别映射文件 (class_map.txt)，这对于推理时显示类别名称很有用
class_map_path = os.path.join(output_dir, "class_map.txt")
with open(class_map_path, "w", encoding="utf-8") as f:
    for label in all_labels:
        f.write(f"{label}\n")
print(f"类别映射文件已保存至: {class_map_path}")
print(f"共找到 {len(all_labels)} 个类别: {all_labels}")


# 如果数据集中没有预设的验证集，则从训练集中按比例分割
split_ratio = 0.9  # 90% 训练, 10% 验证
if "validation" not in dataset and "val" not in dataset:
    print("数据集中未找到验证集，正在从训练集中分割...")
    # 使用固定的随机种子，确保每次分割结果一致
    train_val_split = dataset["train"].train_test_split(
        test_size=1 - split_ratio, shuffle=True, seed=42
    )
    dataset = DatasetDict(
        {"train": train_val_split["train"], "validation": train_val_split["test"]}
    )
    print(
        f"数据集已分割: 训练集 {len(dataset['train'])} 样本, 验证集 {len(dataset['validation'])} 样本"
    )
else:
    # 如果存在 'val' 或 'validation'，统一使用 'validation' 作为键
    if "val" in dataset and "validation" not in dataset:
        dataset["validation"] = dataset.pop("val")
    print(
        f"使用预设的数据集: 训练集 {len(dataset['train'])} 样本, 验证集 {len(dataset['validation'])} 样本"
    )


# --- 3. 核心转换函数 ---


def convert_to_yolo(split_data, split_name):
    """
    将Hugging Face数据集的单个分割转换为YOLO格式。

    Args:
        split_data (Dataset): 'train' 或 'validation' 的数据集对象。
        split_name (str): "train" 或 "val"。
    """
    # 使用tqdm包装循环以显示进度条
    print(f"\n正在转换 '{split_name}' 数据集...")
    for i, item in tqdm(
        enumerate(split_data), total=len(split_data), desc=f"Converting {split_name}"
    ):
        img = item["image"]

        # --- 图像处理 ---
        # 统一图像模式为RGB，这是JPEG格式的要求
        if img.mode == "RGBA":
            # 为RGBA图像创建一个白色背景板进行合并，去掉透明通道
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # 使用Alpha通道作为蒙版
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # 保存图像文件
        img_path = os.path.join(
            output_dir, split_name, "images", f"{split_name}_{i:05d}.jpg"
        )
        img.save(img_path, format="JPEG", quality=95)

        # --- 标签处理 (核心修改) ---
        # 获取图片的实际宽度和高度
        img_w, img_h = img.size

        # 获取类别ID
        # 注意：Hugging Face数据集中的label可以直接是整数ID
        class_id = item["label"]

        # 1. 定义覆盖整张图的像素级边界框 (x_min, y_min, x_max, y_max)
        # 对于分类任务，边界框就是图片本身
        bbox_pixels = (0, 0, img_w, img_h)
        x_min, y_min, x_max, y_max = bbox_pixels

        # 2. 根据YOLO公式，将像素坐标转换为归一化的中心点坐标和宽高
        # x_center = (x_min + x_max) / 2 / img_w
        # y_center = (y_min + y_max) / 2 / img_h
        # width    = (x_max - x_min) / img_w
        # height   = (y_max - y_min) / img_h
        x_center_norm = ((x_min + x_max) / 2) / img_w
        y_center_norm = ((y_min + y_max) / 2) / img_h
        width_norm = (x_max - x_min) / img_w
        height_norm = (y_max - y_min) / img_h

        # 对于覆盖整图的情况，以上计算结果将永远是 (0.5, 0.5, 1.0, 1.0)
        # 但这样写逻辑更清晰，也更容易适配真正的边界框数据

        # 创建并写入标签文件
        label_path = os.path.join(
            output_dir, split_name, "labels", f"{split_name}_{i:05d}.txt"
        )
        class_id = class_map[item["label"]]
        with open(label_path, "w", encoding="utf-8") as f:
            f.write(
                f"{class_id} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n"
            )


# --- 4. 执行转换 ---
convert_to_yolo(dataset["train"], "train")
convert_to_yolo(dataset["validation"], "val")


# --- 5. 创建YOLO数据集配置文件 (dataset.yaml) ---
yaml_path = os.path.join(output_dir, "dataset.yaml")
with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(f"# YOLOv8 dataset configuration file\n")
    f.write(f"# Path to the root directory of the dataset\n")
    f.write(f"path: {os.path.abspath(output_dir)}\n\n")  # 使用绝对路径更稳妥
    f.write(f"# Train/validation images directories\n")
    f.write(f"train: train/images\n")
    f.write(f"val: val/images\n\n")
    f.write(f"# Number of classes\n")
    f.write(f"nc: {len(all_labels)}\n\n")
    f.write(f"# Class names\n")
    f.write(f"names: {list(class_map.keys())}\n")  # 直接使用class_map的键列表

print(f"\n转换完成！YOLO格式的数据集已保存在 '{output_dir}' 目录下。")
print(f"YOLO配置文件已创建: {yaml_path}")
