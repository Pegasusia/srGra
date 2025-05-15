import sys
import multiprocessing
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QPushButton, QTextEdit, QWidget, QFileDialog,
                             QLabel, QLineEdit, QComboBox, QHBoxLayout, QRadioButton, QButtonGroup, QAction,
                             QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from core.config_manager import ConfigManager
from core.log_handler import LogHandler
from core.inference_ctrl import InferenceController
from core.image_display import ImageDisplay


class InferenceGUI(QMainWindow):

    def __init__(self):
        super().__init__()
        # 初始化核心组件
        self.log_text = QTextEdit()
        self.log_handler = LogHandler(self.log_text)
        self.config_manager = ConfigManager(self.log_handler.emit)
        self.inference_ctrl = None

        # 初始化UI
        self._init_ui()
        self._setup_shortcuts()
        self.config_manager.load_config(self)
        self.inference_ctrl = InferenceController(self.log_handler, self.config_manager, self)

    def _init_ui(self):
        """完整UI布局"""
        self.setWindowTitle("超分辨率重建")
        self.setGeometry(100, 100, 1400, 900)

        # 输入类型选择
        input_type_layout = QHBoxLayout()
        self.input_type_label = QLabel("选择输入类型:")
        self.file_radio = QRadioButton("单图片")
        self.folder_radio = QRadioButton("文件夹")
        self.video_radio = QRadioButton("视频")
        self.file_radio.setChecked(True)
        self.input_type_group = QButtonGroup(self)
        self.input_type_group.addButton(self.file_radio)
        self.input_type_group.addButton(self.folder_radio)
        self.input_type_group.addButton(self.video_radio)
        input_type_layout.addWidget(self.input_type_label)
        input_type_layout.addWidget(self.file_radio)
        input_type_layout.addWidget(self.folder_radio)
        input_type_layout.addWidget(self.video_radio)

        # 模型选择
        model_selection_layout = QHBoxLayout()
        self.model_selection_label = QLabel("选择神经网络:")
        self.model_selection_combo = QComboBox()
        self.model_selection_combo.addItems(["ESRGAN", "EDSR", "BasicVSR"])
        model_selection_layout.addWidget(self.model_selection_label)
        model_selection_layout.addWidget(self.model_selection_combo)

        # 路径选择组件
        def create_path_row(label_text, line_edit, button_callback):
            layout = QHBoxLayout()
            label = QLabel(label_text)
            layout.addWidget(label)
            layout.addWidget(line_edit)
            button = QPushButton("浏览")
            button.clicked.connect(button_callback)
            layout.addWidget(button)
            return layout

        self.input_path = QLineEdit()
        self.model_path = QLineEdit()
        self.output_path = QLineEdit()

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.addLayout(input_type_layout)
        main_layout.addLayout(model_selection_layout)
        main_layout.addLayout(create_path_row("输入路径:", self.input_path, self.select_input_path))
        main_layout.addLayout(create_path_row("模型路径:", self.model_path, self.select_model_file))
        main_layout.addLayout(create_path_row("输出路径:", self.output_path, self.select_output_folder))

        # 图像显示区域
        image_layout = QHBoxLayout()
        self.input_image_label = QLabel()
        self.input_image_label.setFixedSize(600, 400)
        self.input_image_label.setStyleSheet("border: 1px solid black;")
        self.output_image_label = QLabel()
        self.output_image_label.setFixedSize(600, 400)
        self.output_image_label.setStyleSheet("border: 1px solid black;")
        image_layout.addWidget(self.input_image_label)
        image_layout.addWidget(self.output_image_label)

        # 图像信息
        info_layout = QHBoxLayout()
        self.input_image_info = QLabel("输入图片信息: ")
        self.output_image_info = QLabel("输出图片信息: ")
        info_layout.addWidget(self.input_image_info)
        info_layout.addWidget(self.output_image_info)

        # 控制按钮
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("开始处理 (Enter)")
        self.stop_button = QPushButton("停止处理")
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)

        # 整合布局
        main_layout.addWidget(self.log_text)
        main_layout.addLayout(image_layout)
        main_layout.addLayout(info_layout)
        main_layout.addLayout(button_layout)

        # 事件绑定
        self.start_button.clicked.connect(self.inference_ctrl.start_inference)
        self.stop_button.clicked.connect(self.inference_ctrl._reset_buttons)
        self.model_selection_combo.currentIndexChanged.connect(self.config_manager.save_config)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def _setup_shortcuts(self):
        """快捷键设置"""
        enter_action = QAction('Enter', self)
        enter_action.setShortcut(Qt.Key_Return)
        enter_action.triggered.connect(self.inference_ctrl.start_inference)
        self.addAction(enter_action)

        esc_action = QAction('Esc', self)
        esc_action.setShortcut(Qt.Key_Escape)
        esc_action.triggered.connect(QApplication.quit)
        self.addAction(esc_action)

    def select_input_path(self):
        """选择输入路径"""
        if self.file_radio.isChecked():
            path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片文件 (*.png *.jpg *.jpeg)")
        elif self.folder_radio.isChecked():
            path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频文件 (*.mp4 *.avi)")

        if path:
            self.input_path.setText(path)
            self.config_manager.save_config(self)

    def select_model_file(self):
        """选择模型文件"""
        path, _ = QFileDialog.getOpenFileName(self, "选择模型", "", "模型文件 (*.pth)")
        if path:
            self.model_path.setText(path)
            self.config_manager.save_config(self)

    def select_output_folder(self):
        """选择输出文件夹"""
        path = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if path:
            self.output_path.setText(path)
            self.config_manager.save_config(self)

    def display_images(self, input_path, output_path):
        """更新图像显示"""
        ImageDisplay.update_image(self.input_image_label, input_path, self.input_image_info)
        ImageDisplay.update_image(self.output_image_label, output_path, self.output_image_info)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    app = QApplication(sys.argv)
    window = InferenceGUI()
    window.show()
    sys.exit(app.exec_())