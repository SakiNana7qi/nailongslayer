"""
存放所有配置常量。
"""

import os

# --- 模型和路径配置 ---
MODEL_PATH = "runs/detect/whole_image_detection/weights/best.onnx"
CLASS_MAP_FILE = "NaiLong.v5i.yolov8-obb/class_map.txt"

# --- 推理参数 ---
IMGSZ = 640
CONF_THRESHOLD = 0.45
IOU_THRESHOLD = 0.2

# --- 性能配置 ---
# 用于图像切片预处理的线程数
NUM_PREPROCESS_THREADS = os.cpu_count() or 4
