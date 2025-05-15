import os
import re
import subprocess
from PyQt5.QtWidgets import (QMainWindow, QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog, QVBoxLayout,
                             QHBoxLayout, QWidget, QComboBox, QRadioButton, QButtonGroup, QMessageBox, QAction)
from PyQt5.QtCore import Qt
from gui.log_signal import LogSignal
from gui.image_display import display_images
from logic.config_manager import ConfigManager
from logic.inference_runner import InferenceRunner
from logic.path_utils import contains_chinese, show_warning_chinese, show_warning_path


class InferenceGUI(QMainWindow):
    """主窗口类 负责创建和管理GUI界面"""


    def __init__(self):
        super().__init__()
        self.setWindowTitle("超分辨率重建系统")
        self.setGeometry(100, 100, 1400, 900)

        self.log_signal = LogSignal()
        self.log_signal.log_signal.connect(self.update_log)

        ConfigManager.ensure_userdata_folder()
        self.runner = InferenceRunner(self)
        self.process = None

        self.init_ui()
        self.load_config()
        self.create_shortcuts()

    def init_ui(self):
        """初始化UI组件"""
        layout = QVBoxLayout()

        self.file_radio = QRadioButton("单图片")
        self.folder_radio = QRadioButton("文件夹")
        self.video_radio = QRadioButton("视频")
        self.file_radio.setChecked(True)
        self.input_type_group = QButtonGroup()
        self.input_type_group.addButton(self.file_radio)
        self.input_type_group.addButton(self.folder_radio)
        self.input_type_group.addButton(self.video_radio)

        radio_layout = QHBoxLayout()
        radio_layout.addWidget(QLabel("选择输入类型:"))
        radio_layout.addWidget(self.file_radio)
        radio_layout.addWidget(self.folder_radio)
        radio_layout.addWidget(self.video_radio)
        layout.addLayout(radio_layout)

        self.model_selection_combo = QComboBox()
        self.model_selection_combo.addItems(["ESRGAN", "EDSR", "BasicVSR"])
        self.model_selection_combo.currentIndexChanged.connect(self.save_config)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("选择神经网络:"))
        model_layout.addWidget(self.model_selection_combo)
        layout.addLayout(model_layout)

        self.input_path, self.input_button = QLineEdit(), QPushButton("打开文件夹")
        self.input_button.clicked.connect(self.select_input_path)
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("输入路径:"))
        input_layout.addWidget(self.input_path)
        input_layout.addWidget(self.input_button)
        layout.addLayout(input_layout)

        self.model_path, self.model_button = QLineEdit(), QPushButton("打开文件夹")
        self.model_button.clicked.connect(self.select_model_file)
        model_path_layout = QHBoxLayout()
        model_path_layout.addWidget(QLabel("模型路径:"))
        model_path_layout.addWidget(self.model_path)
        model_path_layout.addWidget(self.model_button)
        layout.addLayout(model_path_layout)

        self.output_path, self.output_button = QLineEdit(), QPushButton("打开文件夹")
        self.output_button.clicked.connect(self.select_output_folder)
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("输出路径:"))
        output_layout.addWidget(self.output_path)
        output_layout.addWidget(self.output_button)
        layout.addLayout(output_layout)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        self.input_image_label = QLabel("输入图片")
        self.output_image_label = QLabel("生成图片")
        for label in (self.input_image_label, self.output_image_label):
            label.setFixedSize(600, 400)
            label.setStyleSheet("border: 1px solid black;")

        image_layout = QHBoxLayout()
        image_layout.addWidget(self.input_image_label)
        image_layout.addWidget(self.output_image_label)
        layout.addLayout(image_layout)

        self.input_image_info = QLabel("输入图片信息: ")
        self.output_image_info = QLabel("生成图片信息: ")
        image_info_layout = QHBoxLayout()
        image_info_layout.addWidget(self.input_image_info)
        image_info_layout.addWidget(self.output_image_info)
        layout.addLayout(image_info_layout)

        self.start_button = QPushButton("运行程序(Enter)")
        self.start_button.clicked.connect(self.start_inference)
        layout.addWidget(self.start_button)

        self.stop_button = QPushButton("取消运行")
        self.stop_button.clicked.connect(self.stop_inference)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def keyPressEvent(self, event):
        """ESC 退出程序"""
        if event.key() == Qt.Key_Escape:
            self.close()

    def create_shortcuts(self):
        """ENTER 启动程序"""
        enter_start = QAction('enter', self)
        enter_start.setShortcut(Qt.Key_Return)
        enter_start.triggered.connect(self.start_inference)
        self.addAction(enter_start)

    def log_message(self, level, message):
        """输出信息到日志框"""
        self.log_signal.log_signal.emit(f"[{level}] {message}")

    def update_log(self, message):
        """更新日志框"""
        self.log_text.append(message)
        self.log_text.ensureCursorVisible()

    def load_config(self):
        """加载配置文件"""
        config = ConfigManager.load()
        self.input_path.setText(config.get("input_path", ""))
        self.model_path.setText(config.get("model_path", ""))
        self.output_path.setText(config.get("output_folder", ""))
        index = self.model_selection_combo.findText(config.get("selected_model", "ESRGAN"))
        if index != -1:
            self.model_selection_combo.setCurrentIndex(index)

    def save_config(self):
        """保存配置文件"""
        config = {
            "input_path": self.input_path.text(),
            "model_path": self.model_path.text(),
            "output_folder": self.output_path.text(),
            "selected_model": self.model_selection_combo.currentText()
        }
        ConfigManager.save(config)

    def select_input_path(self):
        """选择输入文件或文件夹"""
        if self.file_radio.isChecked():
            file, _ = QFileDialog.getOpenFileName(self, "选择输入文件", self.input_path.text(), "Images (*.png *.jpg *.jpeg)")
            if file: self.input_path.setText(file)
        elif self.folder_radio.isChecked():
            folder = QFileDialog.getExistingDirectory(self, "选择输入文件夹", self.input_path.text())
            if folder: self.input_path.setText(folder)
        elif self.video_radio.isChecked():
            file, _ = QFileDialog.getOpenFileName(self, "选择视频", self.input_path.text(), "Videos (*.mp4 *.avi *.mkv)")
            if file: self.input_path.setText(file)
        self.save_config()

    def select_model_file(self):
        """选择模型文件"""
        file, _ = QFileDialog.getOpenFileName(self, "选择模型", self.model_path.text(), "PyTorch Model (*.pth)")
        if file:
            self.model_path.setText(file)
            self.save_config()

    def select_output_folder(self):
        """选择输出文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹", self.output_path.text())
        if folder:
            self.output_path.setText(folder)
            self.save_config()

    def start_inference(self):
        """开始推理"""

        # 清空日志框
        self.log_text.clear()

        # 载入模型
        input_path = self.input_path.text()
        model_path = self.model_path.text()
        output_path = self.output_path.text()

        # 检查路径
        if any(map(contains_chinese, [input_path, model_path, output_path])):
            show_warning_chinese()
            return
        if not all(map(os.path.exists, [input_path, model_path, output_path])):
            show_warning_path()
            return
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.runner.start(input_path, model_path, output_path)

    def update_log_queue(self, log_queue, process):
        """更新日志队列"""
        import time
        while process.is_alive() or not log_queue.empty():
            try:
                message = log_queue.get(timeout=0.1)
                self.log_message("INFO", message)
            except:
                time.sleep(0.05) # 避免CPU占用过高
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def stop_inference(self):
        """停止推理"""
        if self.process:
            self.process.terminate()
            self.log_message("INFO", "用户终止运行")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def run_inference(self, input_path, model_path, output_folder):
        """运行推理"""
        import subprocess, os
        model = self.model_selection_combo.currentText()
        if model == "ESRGAN":
            script = "inference/inference_esrgan.py"
            suffix = "_ESRGAN.png"
        elif model == "EDSR":
            script = "inference/inference_edsr.py"
            suffix = "_EDSR.png"
        elif model == "BasicVSR":
            script = "inference/inference_basicvsr.py"
            suffix = "_BasicVSR.png"
        else:
            self.log_message("ERROR", "未知模型")
            return

        if self.file_radio.isChecked():
            command = [
                "python", script, "--input_file", input_path, "--model_path", model_path, "--output", output_folder
            ]
            input_image = input_path
            output_image = os.path.join(output_folder, os.path.splitext(os.path.basename(input_path))[0] + suffix)
        elif self.folder_radio.isChecked():
            command = ["python", script, "--input", input_path, "--model_path", model_path, "--output", output_folder]
            files = sorted(os.listdir(input_path))
            input_image = os.path.join(input_path, files[0]) if files else None
            output_image = os.path.join(output_folder, os.path.splitext(files[0])[0] + suffix) if files else None
        else:
            command = [
                "python", script, "--input_path", input_path, "--model_path", model_path, "--save_path", output_folder
            ]
            input_image = input_path
            output_image = os.path.join(output_folder, os.path.splitext(os.path.basename(input_path))[0] + suffix)

        try:
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for line in self.process.stdout:
                self.log_message("INFO", line.strip())
            for line in self.process.stderr:
                self.log_message("ERROR", line.strip())
            self.process.wait()
            if self.process.returncode == 0:
                self.log_message("SUCCESS", "推理完成")
                display_images(self, input_image, output_image)
                self.open_output_folder(output_folder)
            else:
                self.log_message("ERROR", f"失败，返回码：{self.process.returncode}")
        except Exception as e:
            self.log_message("ERROR", f"推理出错: {str(e)}")
        finally:
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def open_output_folder(self, folder):
        """结束后自动打开输出文件夹"""
        try:
            if os.name == "nt":
                os.startfile(folder)
            else:
                subprocess.Popen(["open", folder])
        except:
            self.log_message("ERROR", "无法打开输出文件夹")