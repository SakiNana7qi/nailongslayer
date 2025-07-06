import onnxruntime as ort
import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QThread

import config
from myutils import select_target_window
from overlay import OverlayWindow
from worker import Worker
import win32gui


def run_realtime_detection():
    """主程序，设置并运行实时检测应用。"""
    # 加载 class_id 和名称
    try:
        with open(config.CLASS_MAP_FILE, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f.readlines()]
        print(f"成功从 {config.CLASS_MAP_FILE} 加载了 {len(class_names)} 个类别。")
    except FileNotFoundError:
        print(f"[Error] 类别文件未找到: {config.CLASS_MAP_FILE}")
        sys.exit(1)

    # 初始化Qt应用
    app = QApplication(sys.argv)

    # 选择目标窗口
    hwnd = select_target_window()
    if not hwnd:
        return
    print(f"已选择窗口句柄: {hwnd}, Title: '{win32gui.GetWindowText(hwnd)}'")

    # 设置线程
    thread = QThread()
    worker = Worker(
        hwnd,
        config.MODEL_PATH,
        config.IMGSZ,
        config.CONF_THRESHOLD,
        config.IOU_THRESHOLD,
        config.NUM_PREPROCESS_THREADS,
    )
    worker.moveToThread(thread)

    # 设置UI和信号槽连接
    overlay = OverlayWindow(class_names)

    # 当线程启动时，开始运行worker的run方法
    thread.started.connect(worker.run)
    # 当worker完成工作后，退出线程
    worker.finished.connect(thread.quit)
    # 当worker对象被销毁时，也退出线程
    worker.destroyed.connect(thread.wait)

    # 当worker发出detections_ready信号时，调用overlay的update_overlay方法
    worker.detections_ready.connect(overlay.update_overlay)
    worker.window_inactive.connect(overlay.hide_overlay)

    # 启动线程
    thread.start()

    print("主GUI线程已启动... 按Ctrl+C在终端中停止程序。")

    # 运行Qt事件循环并处理退出
    try:
        app.exec()
    finally:
        print("\n正在停止工作线程...")
        worker.stop()
        thread.quit()
        thread.wait()  # 确保主线程不会在线程完全死透前还没跑完
        print("程序已退出。")


if __name__ == "__main__":

    run_realtime_detection()
