"""
定义QObject工作线程，负责屏幕捕捉和调用推理，避免UI阻塞。
"""

import onnxruntime as ort
import time
from PyQt6.QtCore import QObject, pyqtSignal

from reasoning import get_inference_results
from myutils import get_dpi_scale_factor
import cv2
import numpy as np
import mss
import win32gui
from multiprocessing.pool import ThreadPool


class Worker(QObject):
    """在后台线程中执行所有耗时操作。"""

    detections_ready = pyqtSignal(list, tuple)
    window_inactive = pyqtSignal()
    finished = pyqtSignal()
    """
    detections_ready: 当一帧处理完成时，发射这个信号，并把检测结果 list 和窗口几何信息 tuple 发射出去
    window_inactive: 当发现目标窗口关闭或最小化时，发射这个信号。
    finished: 发射工作循环正常结束信号。
    """

    def __init__(self, hwnd, model_path, imgsz, conf, iou, num_threads):
        super().__init__()
        self.hwnd = hwnd
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.num_threads = num_threads
        self._is_running = True

    def run(self):
        """工作线程的主循环。"""

        print("工作线程已启动...")
        session = ort.InferenceSession(
            self.model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        sct = mss.mss()
        NUM_THREADS = self.num_threads
        thread_pool = ThreadPool(processes=NUM_THREADS)
        print(f"已创建 {NUM_THREADS} 个线程的全局线程池。")
        # 在工作线程内部创建和管理线程池
        frame_count = 0
        while self._is_running:
            t_start = time.perf_counter()

            if not win32gui.IsWindow(self.hwnd) or win32gui.IsIconic(self.hwnd):
                # 不存在或者被最小化，睡香香觉
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
            img = np.array(sct.grab(monitor))  # 截图器 BGRA
            if img.size == 0:
                continue
            t_grab_end = time.perf_counter()

            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            t_inference_start = time.perf_counter()
            results = get_inference_results(
                session, img_bgr, thread_pool, self.imgsz, self.conf, self.iou
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
            """
            if frame_count % 10 == 0:
                print(
                    f"屏幕截图 (sct.grab): {(t_grab_end - t_grab_start) * 1000:.2f} ms"
                )
                print(
                    f"完整推理函数 (get_inference_results): {(t_inference_end - t_inference_start) * 1000:.2f} ms"
                )
                print(f"总循环时间: {(t_end - t_start) * 1000:.2f} ms")
            frame_count += 1
            """

        sct.close()
        print("工作线程已停止。")
        # 程序结束时关闭线程池
        print("正在关闭线程池...")
        thread_pool.close()
        thread_pool.join()
        print("线程池已关闭。")
        self.finished.emit()

    def stop(self):
        """请求停止工作循环。"""
        self._is_running = False
