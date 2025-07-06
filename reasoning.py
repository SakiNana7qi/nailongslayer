"""
负责ONNX模型推理和相关的图像预处理、后处理。
"""

import onnxruntime as ort
import time
import cv2
import numpy as np
from multiprocessing.pool import ThreadPool
import torch
from torchvision.ops import nms


def _preprocess_tile(
    tile: np.ndarray, imgsz: int
) -> tuple[np.ndarray, float, int, int]:
    """
    (私有)预处理单个图块。
    返回: input_tensor, scale, pad_w, pad_h
    """
    tile_h, tile_w = tile.shape[:2]  # tile h w 通道数
    scale = min(imgsz / tile_h, imgsz / tile_w)
    new_h, new_w = int(tile_h * scale), int(tile_w * scale)  # 最长的一边 = imgsz

    resized_tile = cv2.resize(tile, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # 用 opencv 的 resize 用INTER_LINEAR插值算法缩放

    padded_tile = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    pad_w, pad_h = (imgsz - new_w) // 2, (imgsz - new_h) // 2
    padded_tile[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized_tile
    # 创建一个全是114灰度的画布，然后把缩放后的图放在中间

    input_tensor = cv2.cvtColor(padded_tile, cv2.COLOR_BGR2RGB)
    # OpenCV 默认 BGR（闹麻了），转成 RGB
    input_tensor = input_tensor.transpose(2, 0, 1)  # 转成 [3,640,640]，yolo 喜欢这个
    input_tensor = np.expand_dims(input_tensor, axis=0)  # 前面加一个 batch 维度
    input_tensor = input_tensor.astype(np.float32) / 255.0  # 0~255 -> 0~1

    return input_tensor, scale, pad_w, pad_h
    # 缩放参数别忘记传回去，以及画布在图像中的位置


def get_inference_results(
    session: ort.InferenceSession,
    image: np.ndarray,
    thread_pool: ThreadPool,
    imgsz: int = 640,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> list[dict]:
    """
    对单张图像执行切片推理，并返回检测结果列表。
    """

    t_total_start = time.perf_counter()

    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    orig_h, orig_w = image.shape[:2]
    tile_size = imgsz
    overlap_ratio = 0.2
    stride = int(tile_size * (1 - overlap_ratio))

    # 批量处理

    t_tile_prep_start = time.perf_counter()
    tiles_with_coords = []
    y_steps = np.arange(0, orig_h, stride)
    x_steps = np.arange(0, orig_w, stride)

    for y in y_steps:
        for x in x_steps:
            tile = image[y : min(y + tile_size, orig_h), x : min(x + tile_size, orig_w)]
            if tile.shape[0] > 0 and tile.shape[1] > 0:
                tiles_with_coords.append((tile, x, y, imgsz))
    # 这里在做一个带重叠的滑动窗口

    if not tiles_with_coords:
        return []

    t_tile_prep_end = time.perf_counter()

    t_preprocess_start = time.perf_counter()

    def process_single_tile(tile, x, y, size):
        input_tensor, scale, pad_w, pad_h = _preprocess_tile(tile, size)
        return input_tensor, scale, pad_w, pad_h, x, y

    # 使用线程池的 starmap 并行处理所有图块
    # starmap 可以接受一个元组列表作为参数，非常方便
    results = thread_pool.starmap(process_single_tile, tiles_with_coords)

    num_tiles = len(results)

    batch_tensor = np.empty((num_tiles, 3, imgsz, imgsz), dtype=np.float32)
    batch_params = []

    for i, item in enumerate(tiles_with_coords):
        input_tensor, scale, pad_w, pad_h = _preprocess_tile(item[0], imgsz)
        batch_tensor[i] = input_tensor
        batch_params.append((scale, pad_w, pad_h, item[1], item[2]))

    params_array = np.array(batch_params, dtype=np.float32)
    t_preprocess_end = time.perf_counter()

    # [N,3,640,640]
    t_run_start = time.perf_counter()
    outputs = session.run(output_names, {input_name: batch_tensor})  # ~60ms
    # [N,84,8400]
    t_run_end = time.perf_counter()

    t_postprocess_start = time.perf_counter()
    batch_dets = outputs[0].transpose(0, 2, 1)  # [N,4,8400]
    boxes = batch_dets[..., :4]
    # [N,8400,4]
    scores_all_classes = batch_dets[..., 4:]
    # [N,8400,80]
    scores = np.max(scores_all_classes, axis=-1)
    class_ids = np.argmax(scores_all_classes, axis=-1)
    # [N,8400,0]

    mask = scores > conf_threshold  # 筛选出置信度达到要求的

    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    filtered_class_ids = class_ids[mask]

    tile_indices = np.where(mask)[0]

    if tile_indices.size == 0:
        return []

    applied_params = params_array[tile_indices]

    scale = applied_params[:, 0]
    pad_w = applied_params[:, 1]
    pad_h = applied_params[:, 2]
    x_offset = applied_params[:, 3]
    y_offset = applied_params[:, 4]

    cx, cy, w, h = filtered_boxes.T
    x1 = (cx - w / 2 - pad_w.squeeze()) / scale.squeeze() + x_offset.squeeze()
    y1 = (cy - h / 2 - pad_h.squeeze()) / scale.squeeze() + y_offset.squeeze()
    x2 = (cx + w / 2 - pad_w.squeeze()) / scale.squeeze() + x_offset.squeeze()
    y2 = (cy + h / 2 - pad_h.squeeze()) / scale.squeeze() + y_offset.squeeze()
    # 还原成绝对坐标

    final_boxes = np.vstack((x1, y1, x2, y2)).T
    t_postprocess_end = time.perf_counter()

    t_nms_start = time.perf_counter()
    final_results = []
    # 按类别进行 NMS 非极大值抑制
    unique_class_ids = np.unique(filtered_class_ids)
    for class_id in unique_class_ids:
        class_mask = filtered_class_ids == class_id
        boxes_for_nms = torch.from_numpy(final_boxes[class_mask]).float()
        scores_for_nms = torch.from_numpy(filtered_scores[class_mask]).float()

        keep_indices = nms(boxes_for_nms, scores_for_nms, iou_threshold)

        keep_indices_np = keep_indices.cpu().numpy()

        kept_boxes = boxes_for_nms[keep_indices_np].cpu().numpy()
        kept_scores = scores_for_nms[keep_indices_np].cpu().numpy()

        for i in range(len(kept_boxes)):
            final_results.append(
                {
                    "box": kept_boxes[i].tolist(),
                    "score": kept_scores[i],
                    "class_id": class_id,
                }
            )
    t_nms_end = time.perf_counter()

    t_total_end = time.perf_counter()
    """
    if not hasattr(get_inference_results, "call_count"):
        get_inference_results.call_count = 0

    if get_inference_results.call_count % 10 == 0:
        print("--- get_inference_results 详细耗时 ---")
        print(
            f"  - (a) 图块准备: {(t_tile_prep_end - t_tile_prep_start) * 1000:.2f} ms"
        )
        print(
            f"  - (b) 批量预处理: {(t_preprocess_end - t_preprocess_start) * 1000:.2f} ms"
        )
        print(
            f"  - (d) session.run: {(t_run_end - t_run_start) * 1000:.2f} ms  <-- GPU部分"
        )
        print(
            f"  - (e) 批量后处理 (坐标转换): {(t_postprocess_end - t_postprocess_start) * 1000:.2f} ms"
        )
        print(f"  - (f) NMS: {(t_nms_end - t_nms_start) * 1000:.2f} ms")
        print(f"  -> 总计: {(t_total_end - t_total_start) * 1000:.2f} ms")

    get_inference_results.call_count += 1
    """
    return final_results
