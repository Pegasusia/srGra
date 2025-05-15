from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap
from PIL import Image
import os


class ImageDisplay:
    """图像显示类 负责更新UI中的图像显示"""

    @staticmethod
    def update_image(label, path, info_label=None):
        """更新图像显示"""
        if not os.path.exists(path):
            return

        # 显示图像
        pixmap = QPixmap(path)
        pixmap = pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio)
        label.setPixmap(pixmap)

        # 更新信息标签
        if info_label:
            with Image.open(path) as img:
                w, h = img.size
            size = os.path.getsize(path) / 1024
            info_label.setText(f"{w}x{h}, {size:.2f} KB")