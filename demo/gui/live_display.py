from PyQt5.QtWidgets import QDialog, QLabel, QHBoxLayout, QVBoxLayout, QFrame
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer
from PIL import Image
import os


class LiveComparisonDialog(QDialog):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("实时图像对比展示")
        self.setMinimumSize(1000, 700)

        layout = QVBoxLayout()

        self.image_layout = QHBoxLayout()
        self.low_frame = self._build_image_info_frame("低分辨率")
        self.high_frame = self._build_image_info_frame("高分辨率")

        self.image_layout.addWidget(self.low_frame)
        self.image_layout.addWidget(self.high_frame)

        layout.addLayout(self.image_layout)
        self.setLayout(layout)

        # 缓存当前图像路径，避免重复加载
        self.current_low_path = ""
        self.current_high_path = ""

    def _build_image_info_frame(self, label_text):
        frame = QFrame()
        frame.setStyleSheet("border: 1px solid #aaa; background-color: #fff;")
        layout = QVBoxLayout()

        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setFixedSize(400, 300)
        layout.addWidget(image_label)

        info_label = QLabel(f"{label_text}图像")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        frame.setLayout(layout)

        frame.image_label = image_label
        frame.info_label = info_label
        return frame

    def update_images(self, low_path, high_path):
        if low_path == self.current_low_path and high_path == self.current_high_path:
            return  # 路径未变则跳过

        self.current_low_path = low_path
        self.current_high_path = high_path

        self._update_single_image(self.low_frame, low_path, "低分辨率")
        self._update_single_image(self.high_frame, high_path, "高分辨率")
        self.repaint()  # 强制刷新控件

    def _update_single_image(self, frame, path, label):
        if not os.path.exists(path):
            return

        pixmap = QPixmap(path)
        if pixmap.isNull():
            return

        scaled = pixmap.scaled(frame.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        frame.image_label.setPixmap(scaled)

        try:
            with Image.open(path) as img:
                w, h = img.size
            size_kb = os.path.getsize(path) / 1024
            frame.info_label.setText(f"{label}图像\n分辨率: {w}x{h}\n大小: {size_kb:.2f} KB\n路径: {path}")
        except:
            frame.info_label.setText(f"{label}图像\n(无法获取图像信息)")

