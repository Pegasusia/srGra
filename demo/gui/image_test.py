from PyQt5.QtWidgets import QDialog, QLabel, QHBoxLayout, QVBoxLayout, QWidget, QSizePolicy, QFrame
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PIL import Image
import os


class SplitImageWidget(QWidget):

    def __init__(self, low_res_path, high_res_path):
        super().__init__()
        self.low_res = QPixmap(low_res_path)
        self.high_res = QPixmap(high_res_path)
        self.split_position = 0
        self.setMouseTracking(True)

    def resizeEvent(self, event):
        self.split_position = self.width() // 2
        super().resizeEvent(event)

    # def paintEvent(self, event):

    #     print("LowRes Loaded:", not self.low_res.isNull())
    #     print("HighRes Loaded:", not self.high_res.isNull())

    #     if self.low_res.isNull() or self.high_res.isNull():
    #         return

    #     painter = QPainter(self)
    #     widget_rect = self.rect()
    #     half = self.split_position

    #     # 统一缩放为控件尺寸，确保完整显示（不裁切）
    #     low_scaled = self.low_res.scaled(widget_rect.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
    #     high_scaled = self.high_res.scaled(widget_rect.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

    #     # 左侧低分图
    #     painter.drawPixmap(QRect(0, 0, half, widget_rect.height()), low_scaled.copy(0, 0, half, widget_rect.height()))
    #     # 右侧高分图
    #     painter.drawPixmap(QRect(half, 0, widget_rect.width() - half, widget_rect.height()), high_scaled.copy(half, 0,widget_rect.width() - half, widget_rect.height()))

    #     pen = QPen(Qt.red, 2)
    #     painter.setPen(pen)
    #     painter.drawLine(half, 0, half, widget_rect.height())

    def paintEvent(self, event):
        if self.low_res.isNull() or self.high_res.isNull():
            return

        painter = QPainter(self)
        try:
            widget_rect = self.rect()
            half = self.split_position

            low_scaled = self.low_res.scaled(widget_rect.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            high_scaled = self.high_res.scaled(widget_rect.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

            painter.drawPixmap(
                QRect(0, 0, half, widget_rect.height()), low_scaled.copy(0, 0, half, widget_rect.height()))

            painter.drawPixmap(
                QRect(half, 0,
                      widget_rect.width() - half, widget_rect.height()),
                high_scaled.copy(half, 0,
                                 widget_rect.width() - half, widget_rect.height()))

            pen = QPen(Qt.red, 2)
            painter.setPen(pen)
            painter.drawLine(half, 0, half, widget_rect.height())
        finally:
            painter.end()

    def mouseMoveEvent(self, event):
        width = self.width()
        self.split_position = max(0, min(event.x(), width))
        self.update()


class ImageComparisonDialog(QDialog):

    def __init__(self, high_res_path, low_res_path):
        super().__init__()
        self.setWindowTitle("图像对比展示")
        self.resize(1200, 900)

        layout = QVBoxLayout()

        # 第一层：滑动对比图（居中 + 边框）
        compare_widget = SplitImageWidget(low_res_path, high_res_path)
        compare_widget.setMinimumHeight(500)
        compare_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        compare_frame = QFrame()
        compare_frame.setStyleSheet("border: 2px solid #aaa; background-color: #f4f4f4;")
        compare_layout = QVBoxLayout()
        compare_layout.setAlignment(Qt.AlignCenter)
        compare_layout.addWidget(compare_widget)
        compare_frame.setLayout(compare_layout)

        layout.addWidget(compare_frame)

        # 第二层+第三层合并为两列（图像 + 信息）
        row_layout = QHBoxLayout()
        for path, name in [(low_res_path, "低分辨率"), (high_res_path, "高分辨率")]:
            col_frame = QFrame()
            col_frame.setStyleSheet("border: 1px solid #aaa; background-color: #fff;")
            col_layout = QVBoxLayout()

            image_label = QLabel()
            pixmap = QPixmap(path)
            image_label.setPixmap(pixmap.scaled(400, 300, Qt.KeepAspectRatio))
            col_layout.addWidget(image_label)

            with Image.open(path) as img:
                w, h = img.size
            size_kb = os.path.getsize(path) / 1024
            info = QLabel(f"{name}图像\n分辨率: {w}x{h}\n大小: {size_kb:.2f} KB\n路径: {path}")
            info.setWordWrap(True)
            col_layout.addWidget(info)

            col_frame.setLayout(col_layout)
            row_layout.addWidget(col_frame)

        layout.addLayout(row_layout)
        self.setLayout(layout)


def display_images(input_image_path, output_image_path):
    dialog = ImageComparisonDialog(output_image_path, input_image_path)
    dialog.exec_()
