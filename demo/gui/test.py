from PyQt5.QtWidgets import QDialog, QLabel, QHBoxLayout, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PIL import Image
import os

class SplitImageWidget(QWidget):
    """用于显示高分辨率和低分辨率图像的对比"""

    def __init__(self, low_res_path, high_res_path):
        super().__init__()
        self.low_res = QPixmap(low_res_path)
        self.high_res = QPixmap(high_res_path)
        self.split_position = self.width() // 2
        self.setMouseTracking(True)

        self.setMinimumSize(1, 1)
        self.setSizePolicy(QWidget.sizePolicy(self))

    def paintEvent(self, event):
        painter = QPainter(self)
        width = self.width()
        height = self.height()
        half = self.split_position

        # low_scaled = self.low_res.scaled(width, height, Qt.KeepAspectRatioByExpanding)
        # high_scaled = self.high_res.scaled(width, height, Qt.KeepAspectRatioByExpanding)

        high_scaled = self.high_res.scaled(self.size(), Qt.KeepAspectRatio)
        low_scaled = self.low_res.scaled(self.size(), Qt.KeepAspectRatio)

        # 左侧显示低分辨率图像
        painter.drawPixmap(QRect(0, 0, half, height), low_scaled, QRect(0, 0, half, height))
        # 右侧显示高分辨率图像
        painter.drawPixmap(QRect(half, 0, width - half, height), high_scaled, QRect(half, 0, width - half, height))

        pen = QPen(Qt.red, 2)
        painter.setPen(pen)
        painter.drawLine(half, 0, half, height)

    def mouseMoveEvent(self, event):
        self.split_position = event.x()
        self.update()

class ImageComparisonDialog(QDialog):

    def __init__(self, high_res_path, low_res_path):
        super().__init__()
        self.setWindowTitle("图像对比展示")
        self.resize(1200, 900)

        layout = QVBoxLayout()

        # 第一层：滑动对比图
        compare_widget = SplitImageWidget(low_res_path, high_res_path)
        compare_widget.setMinimumHeight(400)
        # compare_widget.setFixedHeight(400)
        # compare_widget.setFixedWidth(800)
        layout.addWidget(compare_widget)

        # 第二+三层：图像和信息并列两列
        row_layout = QHBoxLayout()

        for path, name in [(low_res_path, "低分辨率"), (high_res_path, "高分辨率")]:
            col = QVBoxLayout()
            label = QLabel()
            pixmap = QPixmap(path).scaled(400, 300, Qt.KeepAspectRatio)
            label.setPixmap(pixmap)
            col.addWidget(label)

            with Image.open(path) as img:
                w, h = img.size
            size_kb = os.path.getsize(path) / 1024
            info_label = QLabel(f"{name}图像\n分辨率: {w}x{h}\n大小: {size_kb:.2f} KB\n路径: {path}")
            info_label.setWordWrap(True)
            col.addWidget(info_label)

            row_layout.addLayout(col)

        layout.addLayout(row_layout)
        self.setLayout(layout)

def display_images(input_image_path, output_image_path):
    dialog = ImageComparisonDialog(output_image_path, input_image_path)
    dialog.exec_()
