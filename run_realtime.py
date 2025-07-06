import os
import cv2
import numpy as np
import onnxruntime as ort
import time
import sys
import torch
from torchvision.ops import nms
import mss
import pygetwindow as gw
import win32gui

import random

import ctypes
from ctypes import wintypes

from PyQt6.QtWidgets import QApplication, QWidget, QLabel
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QFont
from PyQt6.QtCore import Qt, QRect, QThread, pyqtSignal, QObject

import threading
from multiprocessing.pool import ThreadPool


def get_dpi_scale_factor(hwnd):
    """获取指定窗口的DPI缩放比例。"""
    try:
        dpi = ctypes.windll.user32.GetDpiForWindow(hwnd)
        # 标准DPI为96
        return dpi / 96.0
    except AttributeError:
        # 如果API不存在（例如在旧版Windows上），则回退
        try:
            hDC = ctypes.windll.user32.GetDC(hwnd)
            # LOGPIXELSX (88) 代表水平方向的DPI
            dpi = ctypes.windll.gdi32.GetDeviceCaps(hDC, 88)
            ctypes.windll.user32.ReleaseDC(hwnd, hDC)
            return dpi / 96.0
        except Exception:
            # 如果所有方法都失败，则返回1.0（无缩放）
            return 1.0


def preprocess_tile(tile, imgsz):
    """预处理单个图块，返回 input_tensor 和必要的换算参数。"""
    tile_h, tile_w = tile.shape[:2]
    scale = min(imgsz / tile_h, imgsz / tile_w)
    new_h, new_w = int(tile_h * scale), int(tile_w * scale)

    resized_tile = cv2.resize(tile, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded_tile = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    pad_w, pad_h = (imgsz - new_w) // 2, (imgsz - new_h) // 2
    padded_tile[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized_tile

    input_tensor = cv2.cvtColor(padded_tile, cv2.COLOR_BGR2RGB)
    input_tensor = input_tensor.transpose(2, 0, 1)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    input_tensor = input_tensor.astype(np.float32) / 255.0

    return input_tensor, scale, pad_w, pad_h


def get_inference_results(
    session, image, imgsz=640, conf_threshold=0.25, iou_threshold=0.45
):
    """
    对单张图像(如屏幕截图)执行完整的切片推理，并返回检测结果列表。
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

    if not tiles_with_coords:
        return []

    t_tile_prep_end = time.perf_counter()

    t_preprocess_start = time.perf_counter()

    def process_single_tile(tile, x, y, size):
        input_tensor, scale, pad_w, pad_h = preprocess_tile(tile, size)
        return input_tensor, scale, pad_w, pad_h, x, y

    # 使用线程池的 starmap 并行处理所有图块
    # starmap 可以接受一个元组列表作为参数，非常方便
    results = thread_pool.starmap(process_single_tile, tiles_with_coords)

    num_tiles = len(results)

    batch_tensor = np.empty((num_tiles, 3, imgsz, imgsz), dtype=np.float32)
    batch_params = []

    for i, item in enumerate(tiles_with_coords):
        input_tensor, scale, pad_w, pad_h = preprocess_tile(item[0], imgsz)
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
    batch_dets = outputs[0].transpose(0, 2, 1)
    boxes = batch_dets[..., :4]
    # [N,8400,4]
    scores_all_classes = batch_dets[..., 4:]
    # [N,8400,80]
    scores = np.max(scores_all_classes, axis=-1)
    class_ids = np.argmax(scores_all_classes, axis=-1)
    # [N,8400,0]

    mask = scores > conf_threshold

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

    final_boxes = np.vstack((x1, y1, x2, y2)).T
    t_postprocess_end = time.perf_counter()

    # --- NMS后处理 ---
    t_nms_start = time.perf_counter()
    final_results = []
    # 按类别进行NMS
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

    return final_results


class OverlayWindow(QWidget):
    def __init__(self, class_names):
        super().__init__()
        self.class_names = class_names
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_InputMethodTransparent)
        self.label = QLabel(self)
        self.was_visible = False

    def update_overlay(self, detections, geometry):
        # 这是一个槽(Slot)，用于接收工作线程发来的信号。
        window_left, window_top, window_width, window_height, dpi_scale = geometry

        window_left = int(window_left / dpi_scale)
        window_top = int(window_top / dpi_scale)

        # 移动并调整窗口大小以匹配目标窗口

        self.setGeometry(window_left, window_top, window_width, window_height)

        # 创建一个透明的 QPixmap 作为画布
        pixmap = QPixmap(window_width, window_height)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        for det in detections:
            box = det["box"]
            abs_x1, abs_y1, abs_x2, abs_y2 = [int(p) for p in box]
            rel_x1 = int(abs_x1 / dpi_scale)
            rel_y1 = int(abs_y1 / dpi_scale)
            w = int((abs_x2 - abs_x1) / dpi_scale)
            h = int((abs_y2 - abs_y1) / dpi_scale)
            pen = QPen(QColor(0, 255, 0, 200), 2)
            painter.setPen(pen)
            painter.drawRect(rel_x1, rel_y1, w, h)
            font = QFont("Arial", 10, QFont.Weight.Bold)
            painter.setFont(font)
            class_name = (
                self.class_names[det["class_id"]]
                if det["class_id"] < len(self.class_names)
                else f"ID {det['class_id']}"
            )
            label = f"{class_name}: {det['score']:.2f}"
            metrics = painter.fontMetrics()
            text_width = metrics.horizontalAdvance(label) + 8
            text_height = metrics.height()
            text_rect = QRect(rel_x1, rel_y1 - text_height, text_width, text_height)
            painter.setBrush(QColor(0, 255, 0, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(text_rect)
            color = QColor(255, 0, 0) if det["score"] > 0.8 else QColor(0, 0, 255, 128)
            painter.setPen(color)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, label)

        painter.end()

        self.label.setPixmap(pixmap)
        self.label.setGeometry(0, 0, geometry[2], geometry[3])
        if not self.isVisible() and (detections or self.was_visible):
            self.show()

        # 记录当前帧是否有检测结果，用于下一帧判断是否隐藏
        self.was_visible = bool(detections)

    def hide_overlay(self):
        self.was_visible = False
        self.hide()


class Worker(QObject):
    # 定义信号
    detections_ready = pyqtSignal(list, tuple)
    window_inactive = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self, hwnd, model_path, imgsz, conf, iou):
        super().__init__()
        self.hwnd = hwnd
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self._is_running = True

    def run(self):
        print("工作线程已启动...")
        session = ort.InferenceSession(
            self.model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        sct = mss.mss()
        frame_count = 0
        while self._is_running:
            t_start = time.perf_counter()

            if not win32gui.IsWindow(self.hwnd) or win32gui.IsIconic(self.hwnd):
                self.window_inactive.emit()
                time.sleep(0.5)
                continue

            dpi_scale = get_dpi_scale_factor(self.hwnd)

            left_client, top_client, right_client, bottom_client = (
                win32gui.GetClientRect(self.hwnd)
            )
            client_width = right_client - left_client
            client_height = bottom_client - top_client

            screen_x_logical, screen_y_logical = win32gui.ClientToScreen(
                self.hwnd, (left_client, top_client)
            )
            monitor = {
                "top": screen_y_logical,
                "left": screen_x_logical,
                "width": client_width,
                "height": client_height,
            }

            if monitor["width"] <= 0 or monitor["height"] <= 0:
                time.sleep(0.1)
                continue

            t_grab_start = time.perf_counter()
            img = np.array(sct.grab(monitor))
            if img.size == 0:
                continue
            t_grab_end = time.perf_counter()

            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            t_inference_start = time.perf_counter()
            results = get_inference_results(
                session, img_bgr, self.imgsz, self.conf, self.iou
            )
            t_inference_end = time.perf_counter()

            geometry = (
                screen_x_logical,
                screen_y_logical,
                client_width,
                client_height,
                dpi_scale,
            )
            self.detections_ready.emit(results, geometry)

            # time.sleep(0.01)
            t_end = time.perf_counter()
            if frame_count % 10 == 0:
                print(
                    f"屏幕截图 (sct.grab): {(t_grab_end - t_grab_start) * 1000:.2f} ms"
                )
                print(
                    f"完整推理函数 (get_inference_results): {(t_inference_end - t_inference_start) * 1000:.2f} ms"
                )
                print(f"总循环时间: {(t_end - t_start) * 1000:.2f} ms")
            frame_count += 1

        sct.close()
        print("工作线程已停止。")
        self.finished.emit()

    def stop(self):
        self._is_running = False


def select_target_window():
    windows = []

    def callback(hwnd, L):
        if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
            windows.append((hwnd, win32gui.GetWindowText(hwnd)))

    win32gui.EnumWindows(callback, None)
    print("请选择要捕捉的窗口:")
    for i, (hwnd, title) in enumerate(windows):
        print(f"[{i}] {title}")
    try:
        choice = int(input("请输入窗口编号: "))
        if 0 <= choice < len(windows):
            return windows[choice][0]
    except (ValueError, IndexError):
        pass
    print("无效的选择。")
    return None


def run_realtime_detection(
    model_path, class_names, imgsz, conf_threshold, iou_threshold
):
    app = QApplication(sys.argv)

    hwnd = select_target_window()
    if not hwnd:
        return
    print(f"已选择窗口句柄: {hwnd}")

    # --- 设置线程 ---
    thread = QThread()
    worker = Worker(hwnd, model_path, imgsz, conf_threshold, iou_threshold)
    worker.moveToThread(thread)

    # --- 设置UI和信号槽连接 ---
    overlay = OverlayWindow(class_names)

    # 当线程启动时，开始运行worker的run方法
    thread.started.connect(worker.run)
    # 当worker完成工作后，退出线程
    worker.finished.connect(thread.quit)
    # 当worker对象被销毁时，也退出线程
    worker.destroyed.connect(thread.wait)

    # 核心连接：当worker发出detections_ready信号时，调用overlay的update_overlay方法
    worker.detections_ready.connect(overlay.update_overlay)
    worker.window_inactive.connect(overlay.hide_overlay)

    # 启动线程
    thread.start()

    print("主GUI线程已启动... 按Ctrl+C在终端中停止程序。")

    # 运行Qt事件循环
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("\n正在停止工作线程...")
        worker.stop()
        thread.quit()
        thread.wait()
        print("程序已退出。")


if __name__ == "__main__":
    MODEL_PATH = "runs/detect/whole_image_detection/weights/best.onnx"
    CLASS_MAP_FILE = "NaiLong.v5i.yolov8-obb/class_map.txt"
    IMGSZ = 640
    CONF_THRESHOLD = 0.45
    IOU_THRESHOLD = 0.2

    try:
        with open(CLASS_MAP_FILE, "r", encoding="utf-8") as f:
            MY_CLASS_NAMES = [line.strip() for line in f.readlines()]
        print(f"成功从 {CLASS_MAP_FILE} 加载了 {len(MY_CLASS_NAMES)} 个类别。")
    except FileNotFoundError:
        print(f"[错误] 类别文件未找到: {CLASS_MAP_FILE}")
        sys.exit(1)

    NUM_THREADS = os.cpu_count() or 4
    thread_pool = ThreadPool(processes=NUM_THREADS)
    print(f"已创建 {NUM_THREADS} 个线程的全局线程池。")

    # 启动实时检测
    run_realtime_detection(
        MODEL_PATH, MY_CLASS_NAMES, IMGSZ, CONF_THRESHOLD, IOU_THRESHOLD
    )

    # 程序结束时关闭线程池
    print("正在关闭线程池...")
    thread_pool.close()
    thread_pool.join()
    print("线程池已关闭。")
