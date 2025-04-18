import os
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QHBoxLayout
from PyQt5.QtGui import QPixmap
from PIL import Image


class ImageComparisonWindow(QDialog):

    def __init__(self, input_image_path, output_image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("图像对比")
        self.setGeometry(200, 200, 1200, 600)  # 设置窗口大小

        # 主布局
        layout = QVBoxLayout()

        # 图像显示布局
        image_layout = QHBoxLayout()

        # 显示输入图像
        input_label = QLabel("输入图片")
        input_pixmap = QPixmap(input_image_path)
        input_pixmap = input_pixmap.scaled(500, 400)  # 调整图像大小
        input_label.setPixmap(input_pixmap)
        image_layout.addWidget(input_label)

        # 显示生成图像
        output_label = QLabel("生成图片")
        output_pixmap = QPixmap(output_image_path)
        output_pixmap = output_pixmap.scaled(500, 400)  # 调整图像大小
        output_label.setPixmap(output_pixmap)
        image_layout.addWidget(output_label)

        layout.addLayout(image_layout)

        # 图像信息布局
        info_layout = QHBoxLayout()

        # 获取输入图像信息
        with Image.open(input_image_path) as img:
            input_width, input_height = img.size
        input_size = os.path.getsize(input_image_path) / 1024  # 文件大小 (KB)
        input_info_label = QLabel(f"输入图片信息: {input_width}x{input_height}, {input_size:.2f} KB")
        info_layout.addWidget(input_info_label)

        # 获取生成图像信息
        with Image.open(output_image_path) as img:
            output_width, output_height = img.size
        output_size = os.path.getsize(output_image_path) / 1024  # 文件大小 (KB)
        output_info_label = QLabel(f"生成图片信息: {output_width}x{output_height}, {output_size:.2f} KB")
        info_layout.addWidget(output_info_label)

        layout.addLayout(info_layout)

        self.setLayout(layout)