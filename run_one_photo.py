import os
import cv2
import numpy as np
import onnxruntime as ort
import time
import sys
import torch

# 确保您已安装 torchvision: pip install torchvision
from torchvision.ops import nms


# ---------------------------- 主要修改部分 ----------------------------
def process_detections(
    detections,
    tile_h,
    tile_w,
    pad_w,
    pad_h,
    scale,
    tile_x_offset,
    tile_y_offset,
    conf_threshold=0.25,
):
    """
    处理模型在单个图块上的输出，将检测框坐标精确地从模型输出空间转换到原始大图空间。
    这是本次修改的核心函数。

    参数:
        detections (np.array): 模型原始输出，形状如 [1, 84, 8400]。
        tile_h (int): 原始图块的高度。
        tile_w (int): 原始图块的宽度。
        pad_w (int): 填充的宽度。
        pad_h (int): 填充的高度。
        scale (float): 从原始图块到模型输入尺寸的缩放比例。
        tile_x_offset (int): 图块在原始大图中的左上角 x 坐标。
        tile_y_offset (int): 图块在原始大图中的左上角 y 坐标。
        conf_threshold (float): 置信度阈值。

    返回:
        一个包含 (box, score, class_id) 的列表，其中 box 是在大图坐标系下的 [x1, y1, x2, y2]。
    """
    # 将模型输出 [1, 84, 8400] 转置为 [8400, 84] 以方便处理
    detections = detections.transpose(0, 2, 1)[0]

    processed_results = []

    for det in detections:
        # 前4个值是 cx, cy, w, h，这些是在 640x640 输入空间中的坐标
        cx, cy, w, h = det[:4]

        # 后面是类别分数
        class_scores = det[4:]
        class_id = np.argmax(class_scores)
        max_score = class_scores[class_id]

        if max_score > conf_threshold:
            # --- 坐标逆向换算开始 ---
            # 1. 从模型输入空间 (imgsz x imgsz) 换算回带填充的图块空间
            box_in_padded_tile_x1 = cx - w / 2
            box_in_padded_tile_y1 = cy - h / 2
            box_in_padded_tile_x2 = cx + w / 2
            box_in_padded_tile_y2 = cy + h / 2

            # 2. 去掉填充 (padding) 的影响，得到在缩放后图块中的坐标
            # (减去加在左边和上边的padding)
            box_in_resized_tile_x1 = box_in_padded_tile_x1 - pad_w
            box_in_resized_tile_y1 = box_in_padded_tile_y1 - pad_h
            box_in_resized_tile_x2 = box_in_padded_tile_x2 - pad_w
            box_in_resized_tile_y2 = box_in_padded_tile_y2 - pad_h

            # 3. 去掉缩放 (scaling) 的影响，得到在原始图块中的坐标
            # (除以缩放比例)
            box_in_original_tile_x1 = box_in_resized_tile_x1 / scale
            box_in_original_tile_y1 = box_in_resized_tile_y1 / scale
            box_in_original_tile_x2 = box_in_resized_tile_x2 / scale
            box_in_original_tile_y2 = box_in_resized_tile_y2 / scale

            # 4. 加上图块在大图中的偏移量，得到在大图中的最终坐标
            global_x1 = box_in_original_tile_x1 + tile_x_offset
            global_y1 = box_in_original_tile_y1 + tile_y_offset
            global_x2 = box_in_original_tile_x2 + tile_x_offset
            global_y2 = box_in_original_tile_y2 + tile_y_offset

            # --- 坐标逆向换算结束 ---

            processed_results.append(
                {
                    "box": [global_x1, global_y1, global_x2, global_y2],
                    "score": float(max_score),
                    "class_id": class_id,
                }
            )

    return processed_results


def predict_large_image_with_tiling(
    model_path,
    source_path,
    class_names,
    imgsz=640,  # <-- 默认值修改为224，与您的目标一致
    conf_threshold=0.25,
    iou_threshold=0.45,
):
    """
    使用切图方式对大图像进行预测（优化版）。
    """
    if not os.path.isfile(source_path):
        raise ValueError(f"无效的源文件路径: {source_path}")

    print("正在加载模型...")
    session = ort.InferenceSession(
        model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    print(f"模型加载成功. 输入: {input_name}, 输出: {output_names}")

    large_image = cv2.imread(source_path)
    if large_image is None:
        print(f"无法读取图像: {source_path}")
        return

    orig_h, orig_w = large_image.shape[:2]
    print(f"成功读取图像: {os.path.basename(source_path)}, 尺寸: {orig_w}x{orig_h}")

    # 定义切图参数 (让重叠比例更直观)
    tile_size = imgsz
    overlap_ratio = 0.3  # 20% 重叠区域
    stride = int(tile_size * (1 - overlap_ratio))

    all_results = []

    print("开始切图推理...")
    start_time = time.time()

    # 使用 np.arange 可以更稳定地生成切图的起始坐标
    y_steps = np.arange(0, orig_h, stride)
    x_steps = np.arange(0, orig_w, stride)

    for y in y_steps:
        for x in x_steps:
            y_end = min(y + tile_size, orig_h)
            x_end = min(x + tile_size, orig_w)
            tile = large_image[y:y_end, x:x_end]

            tile_h, tile_w = tile.shape[:2]
            if tile_h == 0 or tile_w == 0:
                continue

            # --- 预处理图块 ( letterbox 方式 ) ---
            scale = min(imgsz / tile_h, imgsz / tile_w)
            new_h, new_w = int(tile_h * scale), int(tile_w * scale)

            resized_tile = cv2.resize(
                tile, (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )

            # 创建一个灰色的底板
            padded_tile = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)

            # 计算填充量
            pad_w = (imgsz - new_w) // 2
            pad_h = (imgsz - new_h) // 2

            # 将缩放后的图块粘贴到底板中心
            padded_tile[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized_tile

            # 转换为模型需要的格式: BGR->RGB, HWC->CHW, Normalization
            input_tensor = cv2.cvtColor(padded_tile, cv2.COLOR_BGR2RGB)
            input_tensor = input_tensor.transpose(2, 0, 1)  # HWC to CHW
            input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension
            input_tensor = input_tensor.astype(np.float32) / 255.0

            outputs = session.run(output_names, {input_name: input_tensor})

            # 处理并转换坐标
            results_in_tile = process_detections(
                detections=outputs[0],
                tile_h=tile_h,
                tile_w=tile_w,
                pad_w=pad_w,
                pad_h=pad_h,
                scale=scale,
                tile_x_offset=x,
                tile_y_offset=y,
                conf_threshold=conf_threshold,
            )
            all_results.extend(results_in_tile)

    inference_time = time.time() - start_time
    print(f"所有图块推理完成, 总耗时: {inference_time:.2f}秒")

    if not all_results:
        print("在整个图像中没有检测到任何物体。")
        return

    print(f"在应用NMS之前，共检测到 {len(all_results)} 个候选框。")

    # --- 使用 TorchVision NMS 进行后处理 ---
    # 按类别对所有检测结果进行分组
    detections_by_class = {}
    for res in all_results:
        class_id = res["class_id"]
        if class_id not in detections_by_class:
            detections_by_class[class_id] = []
        detections_by_class[class_id].append(res)

    final_boxes = []
    final_scores = []
    final_class_ids = []

    for class_id, detections in detections_by_class.items():
        boxes = torch.tensor([d["box"] for d in detections], dtype=torch.float32)
        scores = torch.tensor([d["score"] for d in detections], dtype=torch.float32)

        # 执行NMS
        keep_indices = nms(boxes, scores, iou_threshold)

        # 保存保留下来的结果
        for idx in keep_indices:
            det = detections[idx]
            final_boxes.append(det["box"])
            final_scores.append(det["score"])
            final_class_ids.append(det["class_id"])

    print(f"在应用NMS之后，剩下 {len(final_boxes)} 个物体。")

    # 绘制最终结果
    output_image = large_image.copy()
    for box, score, class_id in zip(final_boxes, final_scores, final_class_ids):
        x1, y1, x2, y2 = [int(p) for p in box]

        class_name = (
            class_names[class_id]
            if class_id < len(class_names)
            else f"Class {class_id}"
        )
        label = f"{class_name}: {score:.2f}"

        color = (0, 0, 255) if "nailong" in class_name else (0, 255, 0)

        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            output_image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
        )

    output_dir = "images/predict_results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "result_" + os.path.basename(source_path))
    cv2.imwrite(output_path, output_image)
    print(f"最终结果已保存至: {output_path}")


# ---------------------------- `if __name__ == "__main__"` 部分保持不变 ----------------------------
if __name__ == "__main__":
    # 配置路径 (根据您的实际路径修改)
    model_path = "runs/detect/whole_image_detection/weights/best.onnx"
    # 将 predict_source 修改为单张大图的路径
    predict_source = "./images/test/7.jpg"  # 替换为您的1260*26478图像路径

    class_map_file = "NaiLong.v5i.yolov8-obb/class_map.txt"
    try:
        with open(class_map_file, "r", encoding="utf-8") as f:
            # 读取所有行，并用 strip() 清除每行末尾的换行符和空白
            MY_CLASS_NAMES = [line.strip() for line in f.readlines()]
        print(f"成功从 {class_map_file} 加载了 {len(MY_CLASS_NAMES)} 个类别。")
        print("类别列表:", MY_CLASS_NAMES)
    except FileNotFoundError:
        print(f"[错误] 类别文件未找到: {class_map_file}")
        print("请确保该文件与您的Python脚本在同一目录下，或者提供正确的文件路径。")
        sys.exit(1)  # 找不到文件则退出程序

    print("\n进行预测...")
    predict_large_image_with_tiling(
        model_path=model_path,
        source_path=predict_source,
        class_names=MY_CLASS_NAMES,
        imgsz=640,  # <-- 明确指定为224
        conf_threshold=0.45,  # <--- 您可以调整这个置信度阈值
        iou_threshold=0.2,  # <-- NMS的IOU阈值
    )

    print("\n操作完成!")
