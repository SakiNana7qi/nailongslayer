"""
定义PyQt6覆盖窗口类，用于在屏幕上绘制检测结果。
"""

from PyQt6.QtWidgets import QWidget, QLabel
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QFont
from PyQt6.QtCore import Qt, QRect


class OverlayWindow(QWidget):
    """一个透明的、置顶的窗口，用于绘制检测框。"""

    def __init__(self, class_names: list[str]):
        super().__init__()
        self.class_names = class_names
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint  # 无边框
            | Qt.WindowType.WindowStaysOnTopHint  # 置顶
            | Qt.WindowType.Tool  # 不会出现在任务栏里
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)  # 背景透明
        self.setAttribute(Qt.WidgetAttribute.WA_InputMethodTransparent)  # 鼠标穿透窗口
        self.label = QLabel(self)
        self.was_visible = False  # 记录上一帧覆盖层是否是可见的

    def update_overlay(self, detections: list[dict], geometry: tuple):
        """槽函数：接收检测结果并更新绘制内容。"""
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

        self.label.setPixmap(pixmap)  # 如果 isVisible 自动 show
        self.label.setGeometry(0, 0, geometry[2], geometry[3])
        if not self.isVisible() and (detections or self.was_visible):
            # 当前窗口被挡住了但是有检测结果还是显示（全屏检测）
            self.show()

        # 记录当前帧是否有检测结果，用于下一帧判断是否隐藏
        self.was_visible = bool(detections)

    def hide_overlay(self):
        """槽函数，当目标窗口不活动时隐藏覆盖层。"""
        self.was_visible = False
        self.hide()
