from PyQt5.QtWidgets import QDialog, QLabel, QHBoxLayout, QVBoxLayout, QWidget, QSizePolicy, QFrame, QPushButton
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PIL import Image
from logic.utils import open_output_file
import os


class SplitImageWidget(QWidget):
    """自定义控件：分割图像对比"""

    def __init__(self, low_res_path, high_res_path):
        super().__init__()
        self.low_res = QPixmap(low_res_path)
        self.high_res = QPixmap(high_res_path)
        self.split_position = 0
        self.setMouseTracking(True)

    def resizeEvent(self, event):
        self.split_position = self.width() // 2
        super().resizeEvent(event)

    def paintEvent(self, event):
        if self.low_res.isNull() or self.high_res.isNull():
            return

        painter = QPainter(self)
        try:
            # 根据控件宽高计算目标图像区域（居中 + 保留比例）
            container_rect = self.rect()
            target_size = self.low_res.size().scaled(container_rect.size(), Qt.KeepAspectRatio)
            target_x = (container_rect.width() - target_size.width()) // 2
            target_y = (container_rect.height() - target_size.height()) // 2
            target_rect = QRect(target_x, target_y, target_size.width(), target_size.height())

            half = max(0, min(self.split_position - target_x, target_size.width()))

            low_scaled = self.low_res.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            high_scaled = self.high_res.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # 左侧低分图
            painter.drawPixmap(QRect(target_x, target_y, half, target_size.height()),
                            low_scaled.copy(0, 0, half, target_size.height()))

            # 右侧高分图
            painter.drawPixmap(QRect(target_x + half, target_y, target_size.width() - half, target_size.height()),
                            high_scaled.copy(half, 0, target_size.width() - half, target_size.height()))

            # 分割线（在图像范围中）
            pen = QPen(Qt.red, 2)
            painter.setPen(pen)
            painter.drawLine(target_x + half, target_y, target_x + half, target_y + target_size.height())

        finally:
            painter.end()

    def mouseMoveEvent(self, event):
        width = self.width()
        self.split_position = max(0, min(event.x(), width))
        self.update()


class ImageComparisonDialog(QDialog):
    """图像对比展示对话框"""

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
            image_label.setPixmap(pixmap.scaled(400, 200, Qt.KeepAspectRatio))
            col_layout.addWidget(image_label)

            with Image.open(path) as img:
                w, h = img.size
            size_kb = os.path.getsize(path) / 1024
            info = QLabel(f"{name}图像\n分辨率: {w}x{h}\n大小: {size_kb:.2f} KB\n路径: {path}")
            info.setWordWrap(True)
            info.setStyleSheet("font-size: 18px; padding: 10px; ")
            col_layout.addWidget(info)

            col_frame.setLayout(col_layout)
            row_layout.addWidget(col_frame)

        layout.addLayout(row_layout)
        self.setLayout(layout)


        # 打开指定文件
        self.open_folder = QPushButton("打开指定文件夹")
        self.open_folder.setStyleSheet("padding: 12px; font-weight: bold; font-size: 22px;")
        self.open_folder.clicked.connect(lambda: open_output_file(high_res_path))
        self.open_folder.setShortcut(Qt.Key_Return)
        layout.addWidget(self.open_folder)


def display_images(input_image_path, output_image_path):
    """显示图像对比"""
    dialog = ImageComparisonDialog(output_image_path, input_image_path)
    dialog.exec_() # 阻塞式对话框
